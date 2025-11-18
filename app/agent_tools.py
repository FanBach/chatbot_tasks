# app/agent_tools.py
from datetime import datetime, timezone
from typing import Optional

from langchain_core.tools import tool
from sqlalchemy.orm import Session

from .db import SessionLocal
from .models import Task


def _parse_deadline(deadline: Optional[str]) -> Optional[datetime]:
    if not deadline:
        return None
    # chấp nhận dạng "2025-11-20" hoặc ISO full
    if "T" in deadline:
        dt = datetime.fromisoformat(deadline)
    else:
        dt = datetime.fromisoformat(deadline + "T00:00:00")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


@tool
def list_tasks_tool(user_id: int, limit: int = 20) -> str:
    """
    Liệt kê các task của user (theo user_id). Dùng khi cần xem danh sách task để tư vấn.
    """
    db: Session = SessionLocal()
    try:
        tasks = (
            db.query(Task)
            .filter(Task.owner_id == user_id)
            .order_by(Task.created_at.desc())
            .limit(limit)
            .all()
        )
        if not tasks:
            return "User hiện chưa có task nào."
        lines = []
        for t in tasks:
            end_str = t.end_date.isoformat() if t.end_date else "không có deadline"
            lines.append(
                f"[ID {t.id}] {t.name} | status={t.status} | deadline={end_str}"
            )
        return "\n".join(lines)
    finally:
        db.close()


@tool
def create_task_tool(
    user_id: int,
    name: str,
    deadline: Optional[str] = None,
    status: str = "todo",
) -> str:
    """
    Tạo task mới cho user với tên, deadline (string, có thể bỏ trống) và status.
    """
    db: Session = SessionLocal()
    try:
        end_date = _parse_deadline(deadline)
        task = Task(
            name=name,
            end_date=end_date,
            status=status or "todo",
            owner_id=user_id,
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        dl = task.end_date.isoformat() if task.end_date else "không có deadline"
        return f"Đã tạo task [ID {task.id}] '{task.name}' (deadline: {dl}, status={task.status})."
    finally:
        db.close()


@tool
def update_task_status_tool(user_id: int, task_id: int, new_status: str) -> str:
    """
    Đổi trạng thái task (todo/doing/done) cho user.
    """
    db: Session = SessionLocal()
    try:
        task = (
            db.query(Task)
            .filter(Task.id == task_id, Task.owner_id == user_id)
            .first()
        )
        if not task:
            return f"Không tìm thấy task ID {task_id} thuộc user."
        task.status = new_status
        task.updated_at = datetime.now(timezone.utc)
        db.commit()
        return f"Đã đổi trạng thái task [ID {task.id}] thành '{task.status}'."
    finally:
        db.close()


@tool
def delete_task_tool(user_id: int, task_id: int) -> str:
    """
    Xoá một task của user theo ID.
    """
    db: Session = SessionLocal()
    try:
        task = (
            db.query(Task)
            .filter(Task.id == task_id, Task.owner_id == user_id)
            .first()
        )
        if not task:
            return f"Không tìm thấy task ID {task_id} thuộc user."
        db.delete(task)
        db.commit()
        return f"Đã xoá task [ID {task_id}]."
    finally:
        db.close()
