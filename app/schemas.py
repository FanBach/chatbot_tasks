# app/schemas.py
from datetime import datetime, timezone
from typing import Optional, List

from pydantic import BaseModel, Field, EmailStr, validator


# ------------ User schemas ------------
class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6, max_length=256)


class UserOut(BaseModel):
    id: int
    email: EmailStr
    created_at: datetime

    class Config:
        from_attributes = True


class UserMeOut(BaseModel):
    id: int
    email: EmailStr
    created_at: datetime
    is_admin: bool

    class Config:
        from_attributes = True


class ChangePasswordIn(BaseModel):
    old_password: str = Field(min_length=6, max_length=256)
    new_password: str = Field(min_length=6, max_length=256)


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


# ------------ Task schemas ------------
class TaskBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    end_date: Optional[datetime] = None
    status: Optional[str] = Field(default="todo")

    @validator("end_date")
    def ensure_tzaware(cls, v):
        if v and v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v

    @validator("status")
    def validate_status(cls, v):
        allowed = {"todo", "doing", "done"}
        if v is not None and v not in allowed:
            raise ValueError(f"status must be one of {allowed}")
        return v


class TaskCreate(TaskBase):
    pass


class TaskUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    end_date: Optional[datetime] = None
    status: Optional[str] = None

    @validator("end_date")
    def ensure_tzaware(cls, v):
        if v and v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v

    @validator("status")
    def validate_status(cls, v):
        if v is None:
            return v
        allowed = {"todo", "doing", "done"}
        if v not in allowed:
            raise ValueError(f"status must be one of {allowed}")
        return v


class TaskOut(BaseModel):
    id: int
    name: str
    created_at: datetime
    updated_at: datetime
    end_date: Optional[datetime]
    status: str

    class Config:
        from_attributes = True


# ------------ Chat & Summary schemas ------------
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)


class ChatTask(BaseModel):
    id: int
    name: str
    end_date: Optional[datetime]
    status: str

    class Config:
        from_attributes = True


class ChatResponse(BaseModel):
    answer: str
    used_tasks: List[ChatTask]


class ChatMessageOut(BaseModel):
    role: str
    content: str
    created_at: datetime

    class Config:
        from_attributes = True


class TaskRisk(BaseModel):
    id: int
    name: str
    end_date: Optional[datetime]
    risk_level: str
    status: str

    class Config:
        from_attributes = True


class TaskSummary(BaseModel):
    total: int
    overdue: int
    upcoming: int
    no_deadline: int
    high_risk: int
    medium_risk: int
    low_risk: int
    details: List[TaskRisk]


class AdminOverview(BaseModel):
    total_users: int
    total_tasks: int
    avg_tasks_per_user: float


class PlanOut(BaseModel):
    plan: str
    used_tasks: List[ChatTask]
