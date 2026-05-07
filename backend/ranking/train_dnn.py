import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sqlalchemy import select

from backend import models
from backend.database import SessionLocal
from backend.ranking.dnn_ranker import MODEL_DIR, MODEL_PATH, RankingDNN
from backend.ranking.feature_engineering import build_feature_vector, build_features, feature_names


def _build_label(behavior: models.UserBehavior) -> float:
    behavior_type = (behavior.behavior_type or "view").lower()
    if behavior_type == "collect":
        return 1.0
    if behavior_type == "like":
        return 0.9
    if behavior_type == "click":
        return 0.7
    if behavior_type == "view":
        return 0.4
    return 0.3


def _sample_negative_items(all_item_ids: list[int], positives: set[int], size: int) -> list[int]:
    candidates = [x for x in all_item_ids if x not in positives]
    if not candidates:
        return []
    if len(candidates) <= size:
        return candidates
    return random.sample(candidates, size)


def build_training_dataset(db, neg_ratio: int = 3) -> tuple[np.ndarray, np.ndarray]:
    all_items = list(db.scalars(select(models.Item.id)).all())
    if not all_items:
        return np.empty((0, len(feature_names())), dtype=np.float32), np.empty((0,), dtype=np.float32)

    behaviors = list(
        db.scalars(
            select(models.UserBehavior).order_by(models.UserBehavior.created_at.asc(), models.UserBehavior.id.asc())
        ).all()
    )
    if not behaviors:
        return np.empty((0, len(feature_names())), dtype=np.float32), np.empty((0,), dtype=np.float32)

    user_positive_map: dict[int, set[int]] = {}
    for behavior in behaviors:
        user_positive_map.setdefault(behavior.user_id, set()).add(behavior.item_id)

    x_rows: list[list[float]] = []
    y_rows: list[float] = []

    for behavior in behaviors:
        recall_score_pos = min(1.0, max(0.1, float(behavior.weight or 1.0) / 5.0))
        pos_features = build_features(
            db=db,
            user_id=behavior.user_id,
            item_id=behavior.item_id,
            recall_score=recall_score_pos,
        )
        x_rows.append(build_feature_vector(pos_features, recall_score=recall_score_pos))
        y_rows.append(_build_label(behavior))

        positives = user_positive_map.get(behavior.user_id, set())
        negative_items = _sample_negative_items(all_items, positives=positives, size=neg_ratio)
        for neg_item_id in negative_items:
            recall_score_neg = random.uniform(0.01, 0.25)
            neg_features = build_features(
                db=db,
                user_id=behavior.user_id,
                item_id=neg_item_id,
                recall_score=recall_score_neg,
            )
            x_rows.append(build_feature_vector(neg_features, recall_score=recall_score_neg))
            y_rows.append(0.0)

    x = np.asarray(x_rows, dtype=np.float32)
    y = np.asarray(y_rows, dtype=np.float32)
    return x, y


def _split_by_time(x: np.ndarray, y: np.ndarray, valid_ratio: float = 0.2):
    n = len(x)
    if n < 10:
        return x, y, x, y
    split_idx = int(n * (1 - valid_ratio))
    split_idx = max(1, min(split_idx, n - 1))
    return x[:split_idx], y[:split_idx], x[split_idx:], y[split_idx:]


def _compute_scaler(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def _normalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


def _evaluate(model: nn.Module, x_valid: torch.Tensor, y_valid: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(x_valid)
        loss = nn.BCEWithLogitsLoss()(logits, y_valid)
    return float(loss.item())


def train(epochs: int, batch_size: int, learning_rate: float, neg_ratio: int, seed: int) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    db = SessionLocal()
    try:
        x, y = build_training_dataset(db=db, neg_ratio=neg_ratio)
    finally:
        db.close()

    if len(x) == 0:
        raise RuntimeError("训练数据为空，请先准备用户行为数据")

    x_train_raw, y_train, x_valid_raw, y_valid = _split_by_time(x, y)
    mean, std = _compute_scaler(x_train_raw)
    x_train = _normalize(x_train_raw, mean, std)
    x_valid = _normalize(x_valid_raw, mean, std)

    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    x_valid_tensor = torch.tensor(x_valid, dtype=torch.float32)
    y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32)

    input_dim = x_train_tensor.shape[1]
    model = RankingDNN(input_dim=input_dim, hidden_dims=(128, 64, 32), dropout=0.2)

    pos_cnt = float((y_train_tensor > 0.5).sum().item())
    neg_cnt = float((y_train_tensor <= 0.5).sum().item())
    pos_weight = torch.tensor([max(neg_cnt / max(pos_cnt, 1.0), 1.0)], dtype=torch.float32)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_loss = float("inf")
    best_state = None
    patience = 3
    no_improve = 0

    for _ in range(epochs):
        model.train()
        indices = np.random.permutation(len(x_train_tensor))
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            xb = x_train_tensor[batch_idx]
            yb = y_train_tensor[batch_idx]

            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        valid_loss = _evaluate(model, x_valid_tensor, y_valid_tensor)
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is None:
        best_state = model.state_dict()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, MODEL_PATH)

    scaler_payload = {
        "mean": mean.astype(np.float32).tolist(),
        "std": std.astype(np.float32).tolist(),
    }
    (MODEL_DIR / "scaler.json").write_text(json.dumps(scaler_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    version = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    meta_payload = {
        "version": version,
        "created_at": datetime.utcnow().isoformat(),
        "input_dim": int(input_dim),
        "feature_names": feature_names(),
        "hidden_dims": [128, 64, 32],
        "dropout": 0.2,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "neg_ratio": neg_ratio,
        "train_size": int(len(x_train)),
        "valid_size": int(len(x_valid)),
        "best_valid_loss": float(best_loss),
    }
    (MODEL_DIR / "meta.json").write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "version": version,
        "model_path": str(MODEL_PATH),
        "best_valid_loss": float(best_loss),
        "train_size": int(len(x_train)),
        "valid_size": int(len(x_valid)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DNN ranking model")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--neg-ratio", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        neg_ratio=args.neg_ratio,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
