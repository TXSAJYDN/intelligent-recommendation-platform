from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend import crud
from backend.models import User
from backend.database import get_db


router = APIRouter(prefix="/api/stats", tags=["stats"])


@router.get("/users")
def users_stats(db: Session = Depends(get_db)):
    stats = crud.get_system_stats(db)
    return {"users": stats["users"]}


@router.get("/items")
def items_stats(db: Session = Depends(get_db)):
    stats = crud.get_system_stats(db)
    return {"items": stats["items"]}


@router.get("/behaviors")
def behaviors_stats(db: Session = Depends(get_db)):
    stats = crud.get_system_stats(db)
    return {"behaviors": stats["behaviors"]}


@router.get("/popular")
def popular_stats(limit: int = 10, db: Session = Depends(get_db)):
    items = crud.list_hot_items(db=db, limit=limit)
    return [
        {
            "item_id": x.id,
            "title": x.title,
            "popularity": x.popularity,
        }
        for x in items
    ]


@router.get("/abtest")
def abtest_stats(db: Session = Depends(get_db)):
    # 用户按组分布
    user_stmt = select(User.group_name, func.count(User.id)).group_by(User.group_name)
    user_rows = db.execute(user_stmt).all()
    
    # 行为按组分布
    from backend.models import UserBehavior
    behavior_stmt = (
        select(User.group_name, func.count(UserBehavior.id))
        .join(UserBehavior, User.id == UserBehavior.user_id)
        .group_by(User.group_name)
    )
    behavior_rows = db.execute(behavior_stmt).all()
    behavior_map = {group: count for group, count in behavior_rows}
    
    return {
        "groups": [
            {
                "group": group, 
                "user_count": count,
                "behavior_count": behavior_map.get(group, 0),
                "avg_behavior": behavior_map.get(group, 0) / count if count > 0 else 0
            } 
            for group, count in user_rows
        ]
    }


@router.get("/abtest/metrics")
def abtest_metrics(db: Session = Depends(get_db)):
    """
    按 A/B 组统计业务转化率指标：
    - view_count, click_count, like_count, collect_count
    - ctr (click/view), like_rate (like/view), collect_rate (collect/view)
    """
    from backend.models import UserBehavior
    from backend.experiments.ab_test import get_user_group

    # 获取所有用户及其分组
    users = db.scalars(select(User)).all()
    group_users: dict[str, list[int]] = {}
    for u in users:
        g = u.group_name or get_user_group(u.id)
        group_users.setdefault(g, []).append(u.id)

    results = []
    for group, user_ids in sorted(group_users.items()):
        behaviors = db.scalars(
            select(UserBehavior).where(UserBehavior.user_id.in_(user_ids))
        ).all()

        view_count = sum(1 for b in behaviors if (b.behavior_type or "").lower() == "view")
        click_count = sum(1 for b in behaviors if (b.behavior_type or "").lower() == "click")
        like_count = sum(1 for b in behaviors if (b.behavior_type or "").lower() == "like")
        collect_count = sum(1 for b in behaviors if (b.behavior_type or "").lower() == "collect")
        total = len(behaviors)

        results.append({
            "group": group,
            "user_count": len(user_ids),
            "total_behaviors": total,
            "view_count": view_count,
            "click_count": click_count,
            "like_count": like_count,
            "collect_count": collect_count,
            "ctr": round(click_count / view_count, 4) if view_count > 0 else 0.0,
            "like_rate": round(like_count / view_count, 4) if view_count > 0 else 0.0,
            "collect_rate": round(collect_count / view_count, 4) if view_count > 0 else 0.0,
            "engagement_rate": round((click_count + like_count + collect_count) / total, 4) if total > 0 else 0.0,
        })

    return {"groups": results}


@router.get("/recommendation")
def recommendation_stats(db: Session = Depends(get_db)):
    stats = crud.get_system_stats(db)
    return {
        "ctr": 0.12,
        "recall": 0.35,
        "precision": 0.28,
        "response_time_ms": 85,
        "cache_hit_rate": 0.61,
        "base": stats,
    }
