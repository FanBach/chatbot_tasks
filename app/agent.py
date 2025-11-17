# app/agent.py
from datetime import datetime, timezone
from typing import Optional
import re

from sqlalchemy.orm import Session

from .models import User, Task, ChatMessage


def handle_chat_action(question: str, user: User, db: Session) -> Optional[str]:
    """
    Rule-based mini agent:
    - 'tạo task ... deadline YYYY-MM-DD'
    - 'xoá task X' / 'xóa task X' / 'delete task X'
    - 'đổi tên task X thành Y' / 'rename task X to Y'
    - 'đánh dấu task X thành done' / 'mark task X as done'
    - 'dời deadline task X sang YYYY-MM-DD' / 'move deadline task X to ...'
    """
    q_raw = question.strip()
    q = q_raw.lower()

    # Tạo task
    if q.startswith("tạo task") or q.startswith("create task"):
        parts = q_raw.split(" ", 2)
        if len(parts) < 3:
            return (
                "Format tạo task chưa rõ. Ví dụ: 'tạo task Học FastAPI deadline 2025-11-20'."
            )
        tail = q_raw.split(" ", 2)[2].strip()

        if "deadline" in tail.lower():
            idx = tail.lower().find("deadline")
            name = tail[:idx].strip()
            deadline_str = tail[idx + len("deadline") :].strip()
        else:
            name = tail
            deadline_str = ""

        if not name:
            return (
                "Tên task trống. Hãy dùng format: 'tạo task Học FastAPI deadline 2025-11-20'."
            )

        end_date = None
        if deadline_str:
            try:
                if "t" in deadline_str.lower():
                    end_date = datetime.fromisoformat(deadline_str)
                else:
                    end_date = datetime.fromisoformat(deadline_str + "T00:00:00")
                if end_date.tzinfo is None:
                    end_date = end_date.replace(tzinfo=timezone.utc)
            except Exception:
                end_date = None

        task = Task(name=name, end_date=end_date, status="todo", owner_id=user.id)
        db.add(task)
        db.commit()
        db.refresh(task)
        deadline_msg = (
            task.end_date.isoformat() if task.end_date else "không có deadline cụ thể"
        )
        return f"Đã tạo task mới: [ID {task.id}] {task.name} (deadline: {deadline_msg})."

    # Xoá task
    m = re.search(r"(xoá|xóa|delete)\s+task\s+(\d+)", q)
    if m:
        task_id = int(m.group(2))
        task = (
            db.query(Task)
            .filter(Task.id == task_id, Task.owner_id == user.id)
            .first()
        )
        if not task:
            return f"Không tìm thấy task ID {task_id} của bạn."
        db.delete(task)
        db.commit()
        return f"Đã xoá task [ID {task_id}]."

    # Đổi tên task
    m = re.search(r"đổi tên task\s+(\d+)\s+thành\s+(.+)", q_raw, re.IGNORECASE)
    if not m:
        m = re.search(r"rename\s+task\s+(\d+)\s+to\s+(.+)", q_raw, re.IGNORECASE)
    if m:
        task_id = int(m.group(1))
        new_name = m.group(2).strip()
        task = (
            db.query(Task)
            .filter(Task.id == task_id, Task.owner_id == user.id)
            .first()
        )
        if not task:
            return f"Không tìm thấy task ID {task_id} của bạn."
        if not new_name:
            return "Tên mới trống, không thể cập nhật."
        task.name = new_name
        task.updated_at = datetime.now(timezone.utc)
        db.commit()
        return f"Đã đổi tên task [ID {task_id}] thành: {new_name}"

    # Đánh dấu status
    m = re.search(
        r"(đánh dấu|mark)\s+task\s+(\d+)\s+(thành|as)\s+(todo|doing|done)",
        q,
        re.IGNORECASE,
    )
    if m:
        task_id = int(m.group(2))
        new_status = m.group(4).lower()
        task = (
            db.query(Task)
            .filter(Task.id == task_id, Task.owner_id == user.id)
            .first()
        )
        if not task:
            return f"Không tìm thấy task ID {task_id} của bạn."
        task.status = new_status
        task.updated_at = datetime.now(timezone.utc)
        db.commit()
        return f"Đã đánh dấu task [ID {task_id}] là '{new_status}'."

    # Dời deadline
    m = re.search(
        r"(dời|đổi|move)\s+deadline\s+task\s+(\d+)\s+(sang|to)\s+(.+)",
        q_raw,
        re.IGNORECASE,
    )
    if m:
        task_id = int(m.group(2))
        deadline_str = m.group(4).strip()
        task = (
            db.query(Task)
            .filter(Task.id == task_id, Task.owner_id == user.id)
            .first()
        )
        if not task:
            return f"Không tìm thấy task ID {task_id} của bạn."

        try:
            if "t" in deadline_str.lower():
                new_deadline = datetime.fromisoformat(deadline_str)
            else:
                new_deadline = datetime.fromisoformat(deadline_str + "T00:00:00")
            if new_deadline.tzinfo is None:
                new_deadline = new_deadline.replace(tzinfo=timezone.utc)
        except Exception:
            return "Không parse được deadline mới. Hãy dùng format YYYY-MM-DD hoặc YYYY-MM-DDTHH:MM."

        task.end_date = new_deadline
        task.updated_at = datetime.now(timezone.utc)
        db.commit()
        return f"Đã cập nhật deadline task [ID {task_id}] thành: {new_deadline.isoformat()}"

    return None


def save_chat_message(db: Session, user: User, role: str, content: str) -> None:
    msg = ChatMessage(user_id=user.id, role=role, content=content)
    db.add(msg)
    db.commit()
