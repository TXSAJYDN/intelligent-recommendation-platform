from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend import models


FEATURE_NAMES: list[str] = [
    "recall_score",
    "popularity_norm",
    "freshness",
    "match_interest",
    "behavior_weight",
    "user_behavior_count",
    "user_recent_7d_count",
    "item_ctr_like",
    "item_ctr_collect",
    "item_age_days_norm",
]


def feature_names() -> list[str]:
    return list(FEATURE_NAMES)


def _normalize_popularity(popularity: float) -> float:
    return min(max(popularity, 0.0) / 1000.0, 1.0)


def _item_age_days(item: models.Item, now: datetime) -> float:
    base_time = item.publish_time or item.created_at or now
    if base_time > now:
        return 0.0
    return float((now - base_time).days)


def build_features(db: Session, user_id: int, item_id: int, recall_score: float = 0.0) -> dict[str, float]:
    item = db.get(models.Item, item_id)
    if item is None:
        return {
            "recall_score": float(recall_score),
            "popularity": 0.0,
            "popularity_norm": 0.0,
            "freshness": 0.0,
            "match_interest": 0.0,
            "behavior_weight": 0.0,
            "user_behavior_count": 0.0,
            "user_recent_7d_count": 0.0,
            "item_ctr_like": 0.0,
            "item_ctr_collect": 0.0,
            "item_age_days_norm": 1.0,
        }

    now = datetime.utcnow()
    user = db.get(models.User, user_id)

    interests = set()
    if user and user.interests:
        interests = {x.strip().lower() for x in user.interests.split(",") if x.strip()}

    item_tags = set()
    if item.tags:
        item_tags = {x.strip().lower() for x in item.tags.split(",") if x.strip()}
    match_interest = 1.0 if interests & item_tags else 0.0

    user_behaviors = list(
        db.scalars(select(models.UserBehavior).where(models.UserBehavior.user_id == user_id)).all()
    )
    user_behavior_count = float(len(user_behaviors))
    recent_boundary = now - timedelta(days=7)
    user_recent_7d_count = float(sum(1 for x in user_behaviors if x.created_at and x.created_at >= recent_boundary))

    user_item_behaviors = [x for x in user_behaviors if x.item_id == item_id]
    if user_item_behaviors:
        behavior_weight = float(sum(max(float(x.weight), 0.0) for x in user_item_behaviors) / len(user_item_behaviors))
    else:
        behavior_weight = 0.0

    item_behaviors = list(
        db.scalars(select(models.UserBehavior).where(models.UserBehavior.item_id == item_id)).all()
    )
    total_item_behaviors = float(len(item_behaviors))
    like_count = float(sum(1 for x in item_behaviors if (x.behavior_type or "").lower() == "like"))
    collect_count = float(sum(1 for x in item_behaviors if (x.behavior_type or "").lower() == "collect"))

    if total_item_behaviors > 0:
        item_ctr_like = like_count / total_item_behaviors
        item_ctr_collect = collect_count / total_item_behaviors
    else:
        item_ctr_like = 0.0
        item_ctr_collect = 0.0

    popularity = float(item.popularity or 0)
    popularity_norm = _normalize_popularity(popularity)

    age_days = _item_age_days(item=item, now=now)
    item_age_days_norm = min(age_days / 365.0, 1.0)
    freshness = max(0.0, 1.0 - age_days / 30.0)

    return {
        "recall_score": float(recall_score),
        "popularity": popularity,
        "popularity_norm": popularity_norm,
        "freshness": freshness,
        "match_interest": match_interest,
        "behavior_weight": behavior_weight,
        "user_behavior_count": user_behavior_count,
        "user_recent_7d_count": user_recent_7d_count,
        "item_ctr_like": item_ctr_like,
        "item_ctr_collect": item_ctr_collect,
        "item_age_days_norm": item_age_days_norm,
    }


def build_feature_vector(features: dict[str, float], recall_score: float | None = None) -> list[float]:
    mapped = dict(features)
    if recall_score is not None:
        mapped["recall_score"] = float(recall_score)
    return [float(mapped.get(name, 0.0)) for name in FEATURE_NAMES]
