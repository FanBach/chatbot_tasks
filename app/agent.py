# app/agent.py
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

from .models import User, Task, ChatMessage
from .config import OPENAI_API_KEY


# =====================================
# 1. SCHEMA CHUẨN CHO ACTION
# =====================================
class TaskAction(BaseModel):
    action: str = Field(
        description=(
            "Hành động: create_task | delete_task | rename_task | "
            "update_deadline | change_status | none"
        )
    )
    task_id: Optional[int] = None
    task_name: Optional[str] = None
    new_name: Optional[str] = None
    deadline: Optional[str] = None
    new_status: Optional[str] = None


# =====================================
# 2. LLM STRUCTURED OUTPUT
# =====================================
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=OPENAI_API_KEY,
    temperature=0,
)

def parse_action(user_input: str) -> TaskAction:
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
Bạn là AI phân tích lệnh task.

Bạn phải trả về JSON theo schema:
- action: create_task / delete_task / rename_task / update_deadline / change_status / none
- task_id: số hoặc null
- task_name: string hoặc null
- new_name: string hoặc null
- deadline: string hoặc null
- new_status: string hoặc null

KHÔNG dùng chữ thừa. CHỈ trả JSON.
"""
        ),
        ("human", "{input}")
    ])

    chain = prompt | llm.with_structured_output(TaskAction)
    return chain.invoke({"input": user_input})



# =====================================
# 3. THỰC THI ACTION LÊN DATABASE
# =====================================
def handle_chat_action(question: str, user: User, db: Session) -> Optional[str]:
    action: TaskAction = parse_action(question)

    # ---- NONE ----
    if action.action == "none":
        return None

    # ---- CREATE TASK ----
    if action.action == "create_task":
        if not action.task_name:
            return "Bạn muốn tạo task gì?"
        name = action.task_name.strip()

        end_date = None
        if action.deadline:
            try:
                if "T" in action.deadline:
                    end_date = datetime.fromisoformat(action.deadline)
                else:
                    end_date = datetime.fromisoformat(action.deadline + "T00:00:00")
                if end_date.tzinfo is None:
                    end_date = end_date.replace(tzinfo=timezone.utc)
            except:
                pass

        task = Task(
            name=name,
            end_date=end_date,
            status="todo",
            owner_id=user.id
        )
        db.add(task)
        db.commit()
        db.refresh(task)

        dl = task.end_date.isoformat() if task.end_date else "không có deadline"
        return f"Đã tạo task mới: [ID {task.id}] {task.name} (deadline: {dl})."

    # ---- DELETE ----
    if action.action == "delete_task":
        if not action.task_id:
            return "Bạn muốn xóa task nào?"
        task = (
            db.query(Task)
            .filter(Task.id == action.task_id, Task.owner_id == user.id)
            .first()
        )
        if not task:
            return f"Không tìm thấy task ID {action.task_id}."
        db.delete(task)
        db.commit()
        return f"Đã xóa task [ID {action.task_id}]."

    # ---- RENAME ----
    if action.action == "rename_task":
        if not action.task_id or not action.new_name:
            return "Thiếu task_id hoặc tên mới."
        task = (
            db.query(Task)
            .filter(Task.id == action.task_id, Task.owner_id == user.id)
            .first()
        )
        if not task:
            return f"Không tìm thấy task ID {action.task_id}."
        task.name = action.new_name.strip()
        task.updated_at = datetime.now(timezone.utc)
        db.commit()
        return f"Đã đổi tên task [ID {task.id}] thành '{task.name}'."

    # ---- UPDATE DEADLINE ----
    if action.action == "update_deadline":
        if not action.task_id or not action.deadline:
            return "Cần task_id và deadline mới."
        task = (
            db.query(Task)
            .filter(Task.id == action.task_id, Task.owner_id == user.id)
            .first()
        )
        if not task:
            return f"Không tìm thấy task ID {action.task_id}."

        try:
            if "T" in action.deadline:
                new_dl = datetime.fromisoformat(action.deadline)
            else:
                new_dl = datetime.fromisoformat(action.deadline + "T00:00:00")
            if new_dl.tzinfo is None:
                new_dl = new_dl.replace(tzinfo=timezone.utc)
            task.end_date = new_dl
        except:
            return "Deadline không đúng định dạng."

        task.updated_at = datetime.now(timezone.utc)
        db.commit()
        return f"Đã cập nhật deadline task [ID {task.id}] thành {task.end_date.isoformat()}."

    # ---- CHANGE STATUS ----
    if action.action == "change_status":
        if not action.task_id or not action.new_status:
            return "Cần task_id và trạng thái mới."
        task = (
            db.query(Task)
            .filter(Task.id == action.task_id, Task.owner_id == user.id)
            .first()
        )
        if not task:
            return f"Không tìm thấy task ID {action.task_id}."
        task.status = action.new_status.strip()
        task.updated_at = datetime.now(timezone.utc)
        db.commit()
        return f"Đã đổi trạng thái task [ID {task.id}] thành '{task.status}'."

    return None


# =====================================
# 4. SAVE CHAT HISTORY
# =====================================
def save_chat_message(db: Session, user: User, role: str, content: str) -> None:
    msg = ChatMessage(user_id=user.id, role=role, content=content)
    db.add(msg)
    db.commit()
