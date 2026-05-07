from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend import crud, schemas
from backend.database import get_db


router = APIRouter(prefix="/api", tags=["users"])


@router.post("/register", response_model=schemas.UserOut)
def register(payload: schemas.UserCreate, db: Session = Depends(get_db)):
    exists = crud.get_user_by_username(db, payload.username)
    if exists:
        raise HTTPException(status_code=400, detail="用户名已存在")
    user = crud.create_user(
        db=db,
        username=payload.username,
        password=payload.password,
        age=payload.age,
        gender=payload.gender,
        interests=payload.interests,
    )
    return user


@router.post("/login", response_model=schemas.UserOut)
def login(payload: schemas.UserLogin, db: Session = Depends(get_db)):
    user = crud.get_user_by_username(db, payload.username)
    if user is None or user.password != payload.password:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    return user


@router.get("/users/{user_id}", response_model=schemas.UserOut)
def get_user(user_id: int, db: Session = Depends(get_db)):
    user = crud.get_user(db, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="用户不存在")
    return user


@router.get("/users/{user_id}/profile", response_model=schemas.UserProfileOut)
def get_user_profile(user_id: int, db: Session = Depends(get_db)):
    user = crud.get_user(db, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="用户不存在")

    profile = crud.upsert_profile(db, user_id)
    return profile
