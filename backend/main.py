import csv
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend import crud, models
from backend.database import Base, SessionLocal, engine
from backend.ranking.dnn_ranker import load_dnn_model
from backend.routers import admin_experiments, admin_ranking, behaviors, experiments, items, recommendations, stats, users


app = FastAPI(title="智能内容推荐平台 API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(users.router)
app.include_router(items.router)
app.include_router(behaviors.router)
app.include_router(recommendations.router)
app.include_router(stats.router)
app.include_router(experiments.router)
app.include_router(admin_ranking.router)
app.include_router(admin_experiments.router)


@app.get("/")
def root():
    return {
        "name": "智能内容推荐平台",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}


def _seed_from_csv(db: Session) -> None:
    data_dir = Path(__file__).parent / "data"
    users_csv = data_dir / "users.csv"
    items_csv = data_dir / "items.csv"
    behaviors_csv = data_dir / "behaviors.csv"

    existing_users = db.scalar(select(models.User).limit(1))
    existing_items = db.scalar(select(models.Item).limit(1))

    if not existing_users and users_csv.exists():
        with users_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                db.add(
                    models.User(
                        username=row["username"],
                        password=row.get("password", "123456"),
                        age=int(row["age"]) if row.get("age") else None,
                        gender=row.get("gender"),
                        interests=row.get("interests"),
                        group_name=row.get("group_name", "A"),
                    )
                )

    if not existing_items and items_csv.exists():
        with items_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                db.add(
                    models.Item(
                        title=row["title"],
                        category=row.get("category"),
                        tags=row.get("tags"),
                        description=row.get("description"),
                        popularity=int(row.get("popularity") or 0),
                    )
                )

    db.commit()

    existing_behaviors = db.scalar(select(models.UserBehavior).limit(1))
    if not existing_behaviors and behaviors_csv.exists():
        import random
        from datetime import datetime, timedelta
        
        users_map = {u.username: u.id for u in db.scalars(select(models.User)).all()}
        items_map = {i.title: i.id for i in db.scalars(select(models.Item)).all()}
        with behaviors_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                username = row.get("username")
                title = row.get("title")
                user_id = users_map.get(username)
                item_id = items_map.get(title)
                if not user_id or not item_id:
                    continue
                
                # 模拟时间分布：将行为均匀分布在过去30天内
                days_ago = random.randint(0, 30)
                seconds_ago = random.randint(0, 86400)
                behavior_time = datetime.utcnow() - timedelta(days=days_ago, seconds=seconds_ago)

                db.add(
                    models.UserBehavior(
                        user_id=user_id,
                        item_id=item_id,
                        behavior_type=row.get("behavior_type", "view"),
                        rating=float(row["rating"]) if row.get("rating") else None,
                        duration=int(row["duration"]) if row.get("duration") else None,
                        weight=float(row.get("weight") or 1.0),
                        created_at=behavior_time
                    )
                )
        db.commit()


@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        _seed_from_csv(db)
        crud.ensure_all_item_embeddings(db)
        load_dnn_model(force_reload=True)
    finally:
        db.close()
