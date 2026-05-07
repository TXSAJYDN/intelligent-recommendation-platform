from sqlalchemy.orm import Session

from backend import crud
from backend.cache.redis_client import cache_client


HOT_KEY = "hot:items"


def get_hot_items(db: Session, limit: int = 20):
    cached = cache_client.get_json(HOT_KEY)
    if cached:
        return cached[:limit]

    items = crud.list_hot_items(db=db, limit=limit)
    payload = [
        {
            "item_id": item.id,
            "title": item.title,
            "score": float(item.popularity),
            "reason": "热门内容",
        }
        for item in items
    ]
    cache_client.set_json(HOT_KEY, payload, ex=120)
    return payload
