import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend import models
from backend.services.embedding_service import deserialize_vector


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return vector
    return vector / norm


def _build_user_interest_vector(db: Session, user_id: int) -> tuple[np.ndarray | None, set[int]]:
    behaviors = (
        db.query(models.UserBehavior)
        .filter(models.UserBehavior.user_id == user_id)
        .order_by(models.UserBehavior.created_at.desc())
        .limit(200)
        .all()
    )
    if not behaviors:
        return None, set()

    item_weight: dict[int, float] = {}
    for behavior in behaviors:
        item_weight[behavior.item_id] = item_weight.get(behavior.item_id, 0.0) + float(behavior.weight)

    interacted_ids = set(item_weight.keys())
    emb_stmt = select(models.ItemEmbedding).where(models.ItemEmbedding.item_id.in_(interacted_ids))
    emb_rows = list(db.scalars(emb_stmt).all())
    if not emb_rows:
        return None, interacted_ids

    weighted_sum = None
    total_weight = 0.0
    for emb in emb_rows:
        weight = item_weight.get(emb.item_id, 1.0)
        vec = deserialize_vector(emb.embedding)
        if weighted_sum is None:
            weighted_sum = vec * weight
        else:
            weighted_sum = weighted_sum + vec * weight
        total_weight += weight

    if weighted_sum is None or total_weight <= 1e-12:
        return None, interacted_ids

    user_vec = _normalize(weighted_sum / total_weight)
    return user_vec.astype(np.float32), interacted_ids


def recall_embedding(db: Session, user_id: int, top_k: int = 50) -> list[tuple[int, float]]:
    user_vec, interacted_ids = _build_user_interest_vector(db=db, user_id=user_id)
    if user_vec is None:
        return []

    all_embs = list(db.scalars(select(models.ItemEmbedding)).all())
    scores: list[tuple[int, float]] = []
    for emb in all_embs:
        if emb.item_id in interacted_ids:
            continue
        item_vec = deserialize_vector(emb.embedding)
        item_vec = _normalize(item_vec.astype(np.float32))
        score = float(np.dot(user_vec, item_vec))
        if score > 0:
            scores.append((emb.item_id, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]
