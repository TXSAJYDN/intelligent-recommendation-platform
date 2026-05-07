from datetime import datetime

from pydantic import BaseModel, Field


class UserCreate(BaseModel):
    username: str
    password: str
    age: int | None = None
    gender: str | None = None
    interests: str | None = None


class UserLogin(BaseModel):
    username: str
    password: str


class UserOut(BaseModel):
    id: int
    username: str
    age: int | None = None
    gender: str | None = None
    interests: str | None = None
    group_name: str
    created_at: datetime

    class Config:
        from_attributes = True


class ItemCreate(BaseModel):
    title: str
    category: str | None = None
    tags: str | None = None
    description: str | None = None


class ItemOut(BaseModel):
    id: int
    title: str
    category: str | None = None
    tags: str | None = None
    description: str | None = None
    popularity: int
    publish_time: datetime | None = None
    created_at: datetime

    class Config:
        from_attributes = True


class BehaviorCreate(BaseModel):
    user_id: int
    item_id: int
    behavior_type: str = Field(description="click/view/like/favorite/rating")
    rating: float | None = None
    duration: int | None = None


class BehaviorOut(BaseModel):
    id: int
    user_id: int
    item_id: int
    behavior_type: str
    rating: float | None = None
    duration: int | None = None
    weight: float
    created_at: datetime

    class Config:
        from_attributes = True


class UserProfileOut(BaseModel):
    user_id: int
    favorite_categories: str | None = None
    favorite_tags: str | None = None
    activity_score: float
    last_active_time: datetime | None = None

    class Config:
        from_attributes = True


class RecommendationItem(BaseModel):
    item_id: int
    title: str
    score: float
    reason: str | None = None


class RecommendationResponse(BaseModel):
    user_id: int
    strategy: str
    items: list[RecommendationItem]
