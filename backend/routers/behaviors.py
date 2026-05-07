from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend import crud, schemas
from backend.database import get_db
from backend.services.profile_service import refresh_profile
from backend.services.recommendation_service import invalidate_user_recommendation_cache


router = APIRouter(prefix="/api", tags=["behaviors"])


@router.post("/behaviors", response_model=schemas.BehaviorOut)
def create_behavior(payload: schemas.BehaviorCreate, db: Session = Depends(get_db)):
    if crud.get_user(db, payload.user_id) is None:
        raise HTTPException(status_code=404, detail="用户不存在")
    if crud.get_item(db, payload.item_id) is None:
        raise HTTPException(status_code=404, detail="内容不存在")

    behavior = crud.create_behavior(
        db=db,
        user_id=payload.user_id,
        item_id=payload.item_id,
        behavior_type=payload.behavior_type,
        rating=payload.rating,
        duration=payload.duration,
    )
    refresh_profile(db=db, user_id=payload.user_id)
    invalidate_user_recommendation_cache(user_id=payload.user_id)
    return behavior


@router.post("/behaviors/realtime", response_model=schemas.BehaviorOut)
def create_behavior_realtime(payload: schemas.BehaviorCreate, db: Session = Depends(get_db)):
    return create_behavior(payload=payload, db=db)


@router.get("/behaviors/{user_id}", response_model=list[schemas.BehaviorOut])
def list_behaviors(user_id: int, limit: int = 100, db: Session = Depends(get_db)):
    if crud.get_user(db, user_id) is None:
        raise HTTPException(status_code=404, detail="用户不存在")
    return crud.list_user_behaviors(db=db, user_id=user_id, limit=limit)
