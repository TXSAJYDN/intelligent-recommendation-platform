from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend import crud, schemas
from backend.database import get_db
from backend.experiments.ab_test import strategy_by_group
from backend.services.hot_service import get_hot_items
from backend.services.recommendation_service import recommend_for_user


router = APIRouter(prefix="/api", tags=["recommendations"])


def _ensure_user(db: Session, user_id: int):
    user = crud.get_user(db, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="用户不存在")
    return user


@router.get("/recommend/guess/{user_id}", response_model=schemas.RecommendationResponse)
def guess_like(user_id: int, limit: int = 20, db: Session = Depends(get_db)):
    user = _ensure_user(db, user_id)
    strategy = strategy_by_group(user.group_name)
    items = recommend_for_user(db=db, user_id=user_id, strategy=strategy, limit=limit, cache_ttl_seconds=120)
    return {"user_id": user_id, "strategy": strategy, "items": items}


@router.get("/recommend/realtime/{user_id}", response_model=schemas.RecommendationResponse)
def realtime_recommend(user_id: int, limit: int = 20, db: Session = Depends(get_db)):
    _ensure_user(db, user_id)
    items = recommend_for_user(db=db, user_id=user_id, strategy="hybrid", limit=limit, cache_ttl_seconds=30)
    return {"user_id": user_id, "strategy": "hybrid", "items": items}


@router.get("/recommend/hot")
def hot_recommend(limit: int = 20, db: Session = Depends(get_db)):
    items = get_hot_items(db=db, limit=limit)
    return {"strategy": "hot", "items": items}


@router.get("/recommend/usercf/{user_id}", response_model=schemas.RecommendationResponse)
def usercf_recommend(user_id: int, limit: int = 20, db: Session = Depends(get_db)):
    _ensure_user(db, user_id)
    items = recommend_for_user(db=db, user_id=user_id, strategy="lr", limit=limit, cache_ttl_seconds=120)
    return {"user_id": user_id, "strategy": "usercf-lr", "items": items}


@router.get("/recommend/itemcf/{user_id}", response_model=schemas.RecommendationResponse)
def itemcf_recommend(user_id: int, limit: int = 20, db: Session = Depends(get_db)):
    _ensure_user(db, user_id)
    items = recommend_for_user(db=db, user_id=user_id, strategy="gbdt", limit=limit, cache_ttl_seconds=120)
    return {"user_id": user_id, "strategy": "itemcf-gbdt", "items": items}


@router.get("/recommend/faiss/{user_id}", response_model=schemas.RecommendationResponse)
def faiss_recommend(user_id: int, limit: int = 20, db: Session = Depends(get_db)):
    _ensure_user(db, user_id)
    items = recommend_for_user(db=db, user_id=user_id, strategy="dnn", limit=limit, cache_ttl_seconds=120)
    return {"user_id": user_id, "strategy": "faiss-dnn", "items": items}


@router.get("/recommend/ranking/{user_id}", response_model=schemas.RecommendationResponse)
def ranking_recommend(user_id: int, model: str = "hybrid", limit: int = 20, db: Session = Depends(get_db)):
    _ensure_user(db, user_id)
    if model not in {"hybrid", "lr", "gbdt", "dnn"}:
        raise HTTPException(status_code=400, detail="model 必须是 hybrid/lr/gbdt/dnn")
    items = recommend_for_user(db=db, user_id=user_id, strategy=model, limit=limit, cache_ttl_seconds=120)
    return {"user_id": user_id, "strategy": model, "items": items}
