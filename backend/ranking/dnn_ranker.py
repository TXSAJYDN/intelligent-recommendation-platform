import json
import threading
from pathlib import Path

import torch
import torch.nn as nn

from backend.ranking.feature_engineering import build_feature_vector, feature_names


MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "ranking" / "dnn"
MODEL_PATH = MODEL_DIR / "model.pt"
META_PATH = MODEL_DIR / "meta.json"
SCALER_PATH = MODEL_DIR / "scaler.json"


class RankingDNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...] = (64, 32), dropout: float = 0.15):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = hidden
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


_state_lock = threading.Lock()
_model: RankingDNN | None = None
_meta: dict | None = None
_scaler_mean: list[float] | None = None
_scaler_std: list[float] | None = None


def _safe_baseline_score(features: dict[str, float], recall_score: float) -> float:
    x = (
        0.45 * recall_score
        + 0.2 * features.get("match_interest", 0.0)
        + 0.2 * min(features.get("popularity", 0.0) / 150.0, 1.0)
        + 0.15 * features.get("freshness", 0.0)
    )
    if x < 0:
        return 0.0
    if x > 1:
        return 1.0
    return float(x)


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_vector(vec: list[float]) -> list[float]:
    if not _scaler_mean or not _scaler_std:
        return vec
    if len(_scaler_mean) != len(vec) or len(_scaler_std) != len(vec):
        return vec
    out: list[float] = []
    for value, mean, std in zip(vec, _scaler_mean, _scaler_std):
        if std <= 1e-12:
            out.append(float(value - mean))
        else:
            out.append(float((value - mean) / std))
    return out


def _build_model(input_dim: int, meta: dict) -> RankingDNN:
    hidden_dims = meta.get("hidden_dims", [64, 32])
    dropout = float(meta.get("dropout", 0.15))
    hidden_tuple = tuple(int(x) for x in hidden_dims)
    model = RankingDNN(input_dim=input_dim, hidden_dims=hidden_tuple, dropout=dropout)
    return model


def load_dnn_model(force_reload: bool = False) -> bool:
    global _model, _meta, _scaler_mean, _scaler_std

    if _model is not None and not force_reload:
        return True

    with _state_lock:
        if _model is not None and not force_reload:
            return True

        if not MODEL_PATH.exists():
            _model = None
            _meta = None
            _scaler_mean = None
            _scaler_std = None
            return False

        try:
            meta = _load_json(META_PATH)
            scaler = _load_json(SCALER_PATH)

            names = meta.get("feature_names") or feature_names()
            input_dim = int(meta.get("input_dim") or len(names))
            model = _build_model(input_dim=input_dim, meta=meta)

            state = torch.load(MODEL_PATH, map_location="cpu")
            model.load_state_dict(state)
            model.eval()

            _model = model
            _meta = meta
            _scaler_mean = [float(x) for x in scaler.get("mean", [])]
            _scaler_std = [float(x) for x in scaler.get("std", [])]
            return True
        except Exception:
            _model = None
            _meta = None
            _scaler_mean = None
            _scaler_std = None
            return False


def get_dnn_model_version() -> str:
    if _meta and _meta.get("version"):
        return str(_meta.get("version"))
    return "baseline"


def get_dnn_status() -> dict:
    loaded = _model is not None
    return {
        "loaded": loaded,
        "model_path": str(MODEL_PATH),
        "version": get_dnn_model_version(),
        "feature_dim": int(_meta.get("input_dim")) if _meta and _meta.get("input_dim") else len(feature_names()),
        "meta": _meta or {},
    }


def predict_score(feature_vector: list[float]) -> float:
    if _model is None and not load_dnn_model(force_reload=False):
        return 0.0

    model = _model
    if model is None:
        return 0.0

    vec = _normalize_vector(feature_vector)
    x = torch.tensor([vec], dtype=torch.float32)
    with torch.no_grad():
        logit = model(x)
        prob = torch.sigmoid(logit).item()
    return float(prob)


def score_with_dnn(features: dict[str, float], recall_score: float) -> float:
    vector = build_feature_vector(features=features, recall_score=recall_score)
    if load_dnn_model(force_reload=False):
        score = predict_score(vector)
        if score > 0:
            return score
    return _safe_baseline_score(features=features, recall_score=recall_score)
