from collections import defaultdict

from datetime import datetime
from sqlalchemy.orm import Session

from backend import crud, models
from backend.cache.redis_client import cache_client
from backend.ranking.dnn_ranker import get_dnn_model_version, score_with_dnn
from backend.ranking.feature_engineering import build_features
from backend.ranking.gbdt_ranker import score_with_gbdt
from backend.ranking.lr_ranker import score_with_lr
from backend.recall.embedding_recall import recall_embedding
from backend.recall.faiss_recall import recall_faiss
from backend.recall.item_cf import recall_item_cf
from backend.recall.user_cf import recall_user_cf


def invalidate_user_recommendation_cache(user_id: int) -> int:
    prefix = f"rec:user:{user_id}:"
    return cache_client.delete_by_prefix(prefix)


def _merge_recall_scores(*recalls: list[tuple[int, float]]) -> dict[int, float]:
    merged: dict[int, float] = defaultdict(float)
    for recall in recalls:
        for item_id, score in recall:
            merged[item_id] += score
    return merged


def recommend_for_user(
    db: Session,
    user_id: int,
    strategy: str = "hybrid",
    limit: int = 20,
    end_time: datetime = None,
    cache_ttl_seconds: int = 90,
) -> list[dict]:
    model_version = get_dnn_model_version()
    cache_key = f"rec:user:{user_id}:{strategy}:{model_version}:{limit}"
    if end_time:
        cache_key += f":{end_time.isoformat()}"
    
    cached = cache_client.get_json(cache_key)
    if cached:
        return cached

    # 注意：这里的召回函数暂不支持 end_time 过滤，但为了离线评测简单化，我们先重点解决结果过滤问题
    usercf = recall_user_cf(db=db, user_id=user_id, top_k=100)
    itemcf = recall_item_cf(db=db, user_id=user_id, top_k=100)
    embedding = recall_embedding(db=db, user_id=user_id, top_k=100)
    faiss = recall_faiss(db=db, user_id=user_id, top_k=100)

    merged = _merge_recall_scores(usercf, itemcf, embedding, faiss)
    interacted = crud.item_interacted_by_user(db=db, user_id=user_id, before_time=end_time)

    if not merged:
        hot_items = crud.list_hot_items(db=db, limit=limit)
        result = [
            {
                "item_id": item.id,
                "title": item.title,
                "score": float(item.popularity),
                "reason": "冷启动热门推荐",
            }
            for item in hot_items
        ]
        cache_client.set_json(cache_key, result, ex=cache_ttl_seconds)
        return result

    ranked: list[tuple[int, float]] = []
    for item_id, recall_score in merged.items():
        if item_id in interacted:
            continue
        features = build_features(db=db, user_id=user_id, item_id=item_id, recall_score=recall_score)
        if strategy == "lr":
            score = score_with_lr(features, recall_score)
        elif strategy == "gbdt":
            score = score_with_gbdt(features, recall_score)
        elif strategy == "dnn":
            score = score_with_dnn(features, recall_score)
        else:
            lr_score = score_with_lr(features, recall_score)
            gbdt_score = score_with_gbdt(features, recall_score)
            dnn_score = score_with_dnn(features, recall_score)
            score = 0.2 * lr_score + 0.2 * gbdt_score + 0.6 * dnn_score
        ranked.append((item_id, score))

    ranked.sort(key=lambda x: x[1], reverse=True)
    selected = ranked[:limit]

    result = []
    for item_id, score in selected:
        item = db.get(models.Item, item_id)
        if not item:
            continue
        result.append(
            {
                "item_id": item.id,
                "title": item.title,
                "score": float(score),
                "reason": "综合召回+排序",
            }
        )

    if not result:
        hot_items = crud.list_hot_items(db=db, limit=limit)
        result = [
            {
                "item_id": item.id,
                "title": item.title,
                "score": float(item.popularity),
                "reason": "兜底热门推荐",
            }
            for item in hot_items
        ]

    cache_client.set_json(cache_key, result, ex=cache_ttl_seconds)
    return result
