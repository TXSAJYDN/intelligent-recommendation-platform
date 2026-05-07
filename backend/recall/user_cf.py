from collections import defaultdict

from sqlalchemy.orm import Session

from backend import models


def recall_user_cf(db: Session, user_id: int, top_k: int = 50) -> list[tuple[int, float]]:
    user_behaviors = db.query(models.UserBehavior).all()

    user_items: dict[int, set[int]] = defaultdict(set)
    for b in user_behaviors:
        user_items[b.user_id].add(b.item_id)

    target_items = user_items.get(user_id, set())
    if not target_items:
        return []

    candidates: dict[int, float] = defaultdict(float)
    for other_user, items in user_items.items():
        if other_user == user_id:
            continue
        inter = len(target_items & items)
        union = len(target_items | items)
        if union == 0:
            continue
        sim = inter / union
        if sim <= 0:
            continue
        for item_id in items - target_items:
            candidates[item_id] += sim

    ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]
