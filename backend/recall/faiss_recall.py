import faiss
import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend import models
from backend.recall.embedding_recall import _build_user_interest_vector
from backend.services.embedding_service import deserialize_vector


def recall_faiss(db: Session, user_id: int, top_k: int = 50) -> list[tuple[int, float]]:
    user_vec, interacted_ids = _build_user_interest_vector(db=db, user_id=user_id)
    if user_vec is None:
        return []

    emb_rows = list(db.scalars(select(models.ItemEmbedding)).all())
    candidate_ids: list[int] = []
    vectors: list[np.ndarray] = []

    for emb in emb_rows:
        if emb.item_id in interacted_ids:
            continue
        vec = deserialize_vector(emb.embedding).astype(np.float32)
        candidate_ids.append(emb.item_id)
        vectors.append(vec)

    if not vectors:
        return []

    matrix = np.vstack(vectors).astype(np.float32)
    faiss.normalize_L2(matrix)

    query = user_vec.reshape(1, -1).astype(np.float32)
    faiss.normalize_L2(query)

    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)

    k = min(top_k, len(candidate_ids))
    distances, indices = index.search(query, k)

    result: list[tuple[int, float]] = []
    for score, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue
        item_id = candidate_ids[int(idx)]
        result.append((item_id, float(score)))
    return result
