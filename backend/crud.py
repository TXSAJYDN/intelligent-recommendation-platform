from collections import Counter
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend import models
from backend.services.embedding_service import (
    MODEL_NAME,
    build_item_text,
    encode_text,
    serialize_vector,
)


BEHAVIOR_WEIGHTS = {
    "view": 1.0,
    "click": 2.0,
    "like": 3.0,
    "favorite": 5.0,
    "rating": 4.0,
}


def create_user(db: Session, username: str, password: str, age: int | None, gender: str | None, interests: str | None) -> models.User:
    user = models.User(
        username=username,
        password=password,
        age=age,
        gender=gender,
        interests=interests,
        group_name="A" if (hash(username) % 2 == 0) else "B",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def get_user_by_username(db: Session, username: str) -> models.User | None:
    return db.scalar(select(models.User).where(models.User.username == username))


def get_user(db: Session, user_id: int) -> models.User | None:
    return db.get(models.User, user_id)


def list_items(db: Session, skip: int = 0, limit: int = 20) -> list[models.Item]:
    stmt = select(models.Item).offset(skip).limit(limit)
    return list(db.scalars(stmt).all())


def create_item(db: Session, title: str, category: str | None, tags: str | None, description: str | None) -> models.Item:
    item = models.Item(
        title=title,
        category=category,
        tags=tags,
        description=description,
        publish_time=datetime.utcnow(),
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    upsert_item_embedding(db=db, item_id=item.id)
    return item


def upsert_item_embedding(db: Session, item_id: int) -> models.ItemEmbedding | None:
    item = db.get(models.Item, item_id)
    if item is None:
        return None

    item_text = build_item_text(
        title=item.title,
        category=item.category,
        tags=item.tags,
        description=item.description,
    )
    vector = encode_text(item_text)
    serialized = serialize_vector(vector)

    emb = db.scalar(select(models.ItemEmbedding).where(models.ItemEmbedding.item_id == item_id))
    if emb is None:
        emb = models.ItemEmbedding(
            item_id=item_id,
            embedding=serialized,
            model_name=MODEL_NAME,
        )
        db.add(emb)
    else:
        emb.embedding = serialized
        emb.model_name = MODEL_NAME
        emb.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(emb)
    return emb


def ensure_all_item_embeddings(db: Session) -> int:
    items = list(db.scalars(select(models.Item)).all())
    count = 0
    for item in items:
        upsert_item_embedding(db=db, item_id=item.id)
        count += 1
    return count


def list_item_embeddings(db: Session) -> list[models.ItemEmbedding]:
    return list(db.scalars(select(models.ItemEmbedding)).all())


def get_item(db: Session, item_id: int) -> models.Item | None:
    return db.get(models.Item, item_id)


def list_hot_items(db: Session, limit: int = 20) -> list[models.Item]:
    stmt = select(models.Item).order_by(models.Item.popularity.desc(), models.Item.id.desc()).limit(limit)
    return list(db.scalars(stmt).all())


def create_behavior(
    db: Session,
    user_id: int,
    item_id: int,
    behavior_type: str,
    rating: float | None = None,
    duration: int | None = None,
) -> models.UserBehavior:
    weight = BEHAVIOR_WEIGHTS.get(behavior_type, 1.0)
    behavior = models.UserBehavior(
        user_id=user_id,
        item_id=item_id,
        behavior_type=behavior_type,
        rating=rating,
        duration=duration,
        weight=weight,
    )
    db.add(behavior)

    item = db.get(models.Item, item_id)
    if item:
        item.popularity = int(item.popularity + weight)

    db.commit()
    db.refresh(behavior)
    return behavior


def list_user_behaviors(db: Session, user_id: int, limit: int = 100) -> list[models.UserBehavior]:
    stmt = (
        select(models.UserBehavior)
        .where(models.UserBehavior.user_id == user_id)
        .order_by(models.UserBehavior.created_at.desc())
        .limit(limit)
    )
    return list(db.scalars(stmt).all())


def upsert_profile(db: Session, user_id: int) -> models.UserProfile:
    behavior_stmt = (
        select(models.UserBehavior, models.Item)
        .join(models.Item, models.UserBehavior.item_id == models.Item.id)
        .where(models.UserBehavior.user_id == user_id)
    )
    rows = db.execute(behavior_stmt).all()

    category_counter: Counter[str] = Counter()
    tag_counter: Counter[str] = Counter()
    activity_score = 0.0
    last_active_time = None

    for behavior, item in rows:
        if item.category:
            category_counter[item.category] += 1
        if item.tags:
            for t in [x.strip() for x in item.tags.split(",") if x.strip()]:
                tag_counter[t] += 1
        activity_score += behavior.weight
        if (last_active_time is None) or (behavior.created_at > last_active_time):
            last_active_time = behavior.created_at

    favorite_categories = ",".join([k for k, _ in category_counter.most_common(3)]) if category_counter else None
    favorite_tags = ",".join([k for k, _ in tag_counter.most_common(5)]) if tag_counter else None

    profile = db.scalar(select(models.UserProfile).where(models.UserProfile.user_id == user_id))
    if profile is None:
        profile = models.UserProfile(
            user_id=user_id,
            favorite_categories=favorite_categories,
            favorite_tags=favorite_tags,
            activity_score=activity_score,
            last_active_time=last_active_time,
        )
        db.add(profile)
    else:
        profile.favorite_categories = favorite_categories
        profile.favorite_tags = favorite_tags
        profile.activity_score = activity_score
        profile.last_active_time = last_active_time

    db.commit()
    db.refresh(profile)
    return profile


def get_profile(db: Session, user_id: int) -> models.UserProfile | None:
    return db.scalar(select(models.UserProfile).where(models.UserProfile.user_id == user_id))


def item_interacted_by_user(db: Session, user_id: int, before_time: datetime | None = None) -> set[int]:
    stmt = select(models.UserBehavior.item_id).where(models.UserBehavior.user_id == user_id)
    if before_time:
        stmt = stmt.where(models.UserBehavior.created_at < before_time)
    return set(db.scalars(stmt).all())


def get_system_stats(db: Session) -> dict:
    users = db.scalar(select(func.count(models.User.id))) or 0
    items = db.scalar(select(func.count(models.Item.id))) or 0
    behaviors = db.scalar(select(func.count(models.UserBehavior.id))) or 0
    return {"users": users, "items": items, "behaviors": behaviors}
