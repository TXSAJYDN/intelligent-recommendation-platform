from collections import defaultdict

from sqlalchemy.orm import Session

from backend import models


def recall_item_cf(db: Session, user_id: int, top_k: int = 50) -> list[tuple[int, float]]:
    behaviors = db.query(models.UserBehavior).all()

    user_items: dict[int, set[int]] = defaultdict(set)
    item_users: dict[int, set[int]] = defaultdict(set)

    for b in behaviors:
        user_items[b.user_id].add(b.item_id)
        item_users[b.item_id].add(b.user_id)

    target_items = user_items.get(user_id, set())
    if not target_items:
        return []

    scores: dict[int, float] = defaultdict(float)
    for item in target_items:
        users_of_item = item_users.get(item, set())
        for other_item, users in item_users.items():
            if other_item in target_items:
                continue
            inter = len(users_of_item & users)
            union = len(users_of_item | users)
            if union == 0:
                continue
            sim = inter / union
            if sim > 0:
                scores[other_item] += sim

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]
