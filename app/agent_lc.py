# app/routes.py
from datetime import datetime, timezone, timedelta
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from .db import get_db
from .models import User, Task, ChatMessage
from .schemas import (
    UserCreate, UserOut, UserMeOut, ChangePasswordIn, Token,
    TaskCreate, TaskUpdate, TaskOut,
    TaskSummary, PlanOut, ChatRequest, ChatResponse,
    ChatMessageOut, AdminOverview, ChatTask,
)
from .security import (
    verify_password, get_password_hash, create_access_token,
    get_current_user, ADMIN_EMAIL,
)
from .ai import summarize_tasks, ask_llm_plan
from .agent import save_chat_message
from .agent_lc import agent_executor  # LangChain AgentExecutor

router = APIRouter()


# -------- Auth & User --------
@router.post("/auth/register", response_model=UserOut, status_code=201)
def register(user_in: UserCreate, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == user_in.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    user = User(email=user_in.email, hashed_password=get_password_hash(user_in.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.post("/auth/login", response_model=Token)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect email or password")

    token = create_access_token({"sub": str(user.id)})
    return Token(access_token=token)


@router.get("/me", response_model=UserMeOut)
def get_me(current_user: User = Depends(get_current_user)):
    return UserMeOut(
        id=current_user.id,
        email=current_user.email,
        created_at=current_user.created_at,
        is_admin=current_user.email == ADMIN_EMAIL,
    )


@router.post("/auth/change_password")
def change_password(
    payload: ChangePasswordIn,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not verify_password(payload.old_password, current_user.hashed_password):
        raise HTTPException(status_code=400, detail="Old password is incorrect")
    current_user.hashed_password = get_password_hash(payload.new_password)
    db.commit()
    return {"detail": "Password updated successfully"}


# -------- Tasks --------
@router.post("/tasks", response_model=TaskOut, status_code=201)
def create_task(
    task_in: TaskCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    task = Task(
        name=task_in.name,
        end_date=task_in.end_date,
        status=task_in.status or "todo",
        owner_id=current_user.id,
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


@router.get("/tasks", response_model=List[TaskOut])
def list_tasks(
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    q = (
        db.query(Task)
        .filter(Task.owner_id == current_user.id)
        .order_by(Task.created_at.desc())
    )
    tasks = q.offset(skip).limit(limit).all()
    return tasks


@router.get("/tasks/{task_id}", response_model=TaskOut)
def get_task(
    task_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    task = (
        db.query(Task)
        .filter(Task.id == task_id, Task.owner_id == current_user.id)
        .first()
    )
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@router.put("/tasks/{task_id}", response_model=TaskOut)
def update_task(
    task_id: int,
    task_in: TaskUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    task = (
        db.query(Task)
        .filter(Task.id == task_id, Task.owner_id == current_user.id)
        .first()
    )
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task_in.name is not None:
        task.name = task_in.name
    if task_in.end_date is not None:
        task.end_date = task_in.end_date
    if task_in.status is not None:
        task.status = task_in.status
    task.updated_at = datetime.now(timezone.utc)

    db.commit()
    db.refresh(task)
    return task


@router.delete("/tasks/{task_id}", status_code=204)
def delete_task(
    task_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    task = (
        db.query(Task)
        .filter(Task.id == task_id, Task.owner_id == current_user.id)
        .first()
    )
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    db.delete(task)
    db.commit()
    return None


@router.post("/tasks/{task_id}/clarify", response_model=TaskOut)
def clarify_task(
    task_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    from .ai import clarify_task_name

    task = (
        db.query(Task)
        .filter(Task.id == task_id, Task.owner_id == current_user.id)
        .first()
    )
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    new_name = clarify_task_name(task)
    task.name = new_name
    task.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(task)
    return task


@router.get("/tasks/summary", response_model=TaskSummary)
def get_task_summary(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    tasks = (
        db.query(Task)
        .filter(Task.owner_id == current_user.id)
        .order_by(Task.created_at.desc())
        .all()
    )
    return summarize_tasks(tasks)


@router.get("/tasks/reminders", response_model=List[TaskOut])
def get_reminders(
    days: int = 3,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    now = datetime.now(timezone.utc)
    end = now + timedelta(days=days)
    tasks = (
        db.query(Task)
        .filter(
            Task.owner_id == current_user.id,
            Task.end_date != None,  # noqa: E711
            Task.end_date >= now,
            Task.end_date <= end,
        )
        .order_by(Task.end_date.asc())
        .all()
    )
    return tasks


@router.get("/tasks/plan/today", response_model=PlanOut)
def plan_today(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    tasks = (
        db.query(Task)
        .filter(
            Task.owner_id == current_user.id,
            Task.status != "done",
        )
        .order_by(Task.end_date.asc().nullslast())
        .limit(50)
        .all()
    )
    plan_text = ask_llm_plan(tasks)
    used_tasks = [
        ChatTask(id=t.id, name=t.name, end_date=t.end_date, status=t.status)
        for t in tasks
    ]
    return PlanOut(plan=plan_text, used_tasks=used_tasks)


# -------- Chat & History --------
@router.get("/chat/history", response_model=List[ChatMessageOut])
def get_chat_history(
    limit: int = 20,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # Lấy message mới nhất trước rồi đảo lại nếu frontend muốn hiển thị cũ -> mới
    msgs = (
        db.query(ChatMessage)
        .filter(ChatMessage.user_id == current_user.id)
        .order_by(ChatMessage.created_at.desc())
        .limit(limit)
        .all()
    )
    return list(reversed(msgs))


@router.post("/tasks/chat", response_model=ChatResponse)
def chat_about_tasks(
    payload: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # Lưu message của user
    save_chat_message(db, current_user, "user", payload.question)

    # Gọi LangChain Agent: agent sẽ tự quyết định
    # - Có cần gọi tool CRUD task không (create/update/delete/list)
    # - Hay chỉ trả lời tư vấn
    result = agent_executor.invoke(
        {
            "input": payload.question,
            "user_id": current_user.id,
        }
    )
    answer = result["output"]

    # Lưu message của bot
    save_chat_message(db, current_user, "bot", answer)

    # Lấy danh sách task hiện tại (ví dụ 50 task gần nhất) trả về cho UI
    tasks = (
        db.query(Task)
        .filter(Task.owner_id == current_user.id)
        .order_by(Task.created_at.desc())
        .limit(50)
        .all()
    )
    used_tasks = [
        ChatTask(
            id=t.id,
            name=t.name,
            end_date=t.end_date,
            status=t.status,
        )
        for t in tasks
    ]
    return ChatResponse(answer=answer, used_tasks=used_tasks)


# -------- Admin --------
@router.get("/admin/overview", response_model=AdminOverview)
def admin_overview(
    db: Session = Depends(get_db), current_user: User = Depends(get_current_user)
):
    if current_user.email != ADMIN_EMAIL:
        raise HTTPException(status_code=403, detail="Not authorized")

    total_users = db.query(User).count()
    total_tasks = db.query(Task).count()
    avg = float(total_tasks) / float(total_users) if total_users > 0 else 0.0
    return AdminOverview(
        total_users=total_users,
        total_tasks=total_tasks,
        avg_tasks_per_user=round(avg, 2),
    )
