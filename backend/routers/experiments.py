from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend import crud
from backend.database import get_db
from backend.experiments.ab_test import strategy_by_group


router = APIRouter(prefix="/api/experiments", tags=["experiments"])


@router.get("/group/{user_id}")
def get_user_group(user_id: int, db: Session = Depends(get_db)):
    user = crud.get_user(db, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="用户不存在")
    strategy = strategy_by_group(user.group_name)
    return {"user_id": user_id, "group": user.group_name, "strategy": strategy}
