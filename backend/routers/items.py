from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend import crud, schemas
from backend.database import get_db


router = APIRouter(prefix="/api", tags=["items"])


@router.post("/items", response_model=schemas.ItemOut)
def create_item(payload: schemas.ItemCreate, db: Session = Depends(get_db)):
    item = crud.create_item(
        db=db,
        title=payload.title,
        category=payload.category,
        tags=payload.tags,
        description=payload.description,
    )
    return item


@router.get("/items", response_model=list[schemas.ItemOut])
def list_items(skip: int = 0, limit: int = 20, db: Session = Depends(get_db)):
    return crud.list_items(db=db, skip=skip, limit=limit)


@router.get("/items/popular")
def get_popular(limit: int = 20, db: Session = Depends(get_db)):
    return [
        {
            "item_id": x.id,
            "title": x.title,
            "score": float(x.popularity),
            "reason": "热门内容",
        }
        for x in crud.list_hot_items(db=db, limit=limit)
    ]


@router.get("/items/{item_id}/similar")
def similar_items(item_id: int, db: Session = Depends(get_db)):
    item = crud.get_item(db, item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="内容不存在")

    category = item.category
    if not category:
        return []

    items = [x for x in crud.list_items(db=db, skip=0, limit=100) if x.id != item_id and x.category == category]
    return [
        {
            "item_id": x.id,
            "title": x.title,
            "score": 1.0,
            "reason": "同类内容",
        }
        for x in items[:20]
    ]


@router.get("/items/{item_id}", response_model=schemas.ItemOut)
def get_item(item_id: int, db: Session = Depends(get_db)):
    item = crud.get_item(db, item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="内容不存在")
    return item
