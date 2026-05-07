from sqlalchemy.orm import Session

from backend import crud


def refresh_profile(db: Session, user_id: int):
    return crud.upsert_profile(db=db, user_id=user_id)
