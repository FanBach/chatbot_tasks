# app/ai.py
from datetime import datetime, timedelta, timezone
from typing import List, Optional
import math
import os

from sqlalchemy.orm import Session

from .models import Task, User
from .schemas import TaskSummary, TaskRisk
from .config import OPENAI_API_KEY

try:
    from openai import OpenAI

    openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except Exception:
    openai_client = None


def compute_risk(task: Task) -> str:
    if not task.end_date:
        return "no_deadline"
    now = datetime.now(timezone.utc)
    delta = (task.end_date - now).total_seconds()
    if delta < 0:
        return "overdue"
    days = delta / 86400.0
    if days <= 1:
        return "high"
    elif days <= 3:
        return "medium"
    elif days <= 7:
        return "low"
    else:
        return "low"


def summarize_tasks(tasks: List[Task]) -> TaskSummary:
    total = len(tasks)
    overdue = 0
    upcoming = 0
    no_deadline = 0
    high_risk = 0
    medium_risk = 0
    low_risk = 0
    details: List[TaskRisk] = []

    now = datetime.now(timezone.utc)

    for t in tasks:
        risk = compute_risk(t)
        if not t.end_date:
            no_deadline += 1
        else:
            if t.end_date < now:
                overdue += 1
            else:
                upcoming += 1

        if risk in ("overdue", "high"):
            high_risk += 1
        elif risk == "medium":
            medium_risk += 1
        elif risk == "low":
            low_risk += 1

        details.append(
            TaskRisk(
                id=t.id,
                name=t.name,
                end_date=t.end_date,
                risk_level=risk,
                status=t.status,
            )
        )

    return TaskSummary(
        total=total,
        overdue=overdue,
        upcoming=upcoming,
        no_deadline=no_deadline,
        high_risk=high_risk,
        medium_risk=medium_risk,
        low_risk=low_risk,
        details=details,
    )


def build_tasks_context(tasks: List[Task]) -> str:
    if not tasks:
        return "Không có task nào cho user."
    lines = []
    for t in tasks:
        end_str = t.end_date.isoformat() if t.end_date else "không có deadline"
        risk = compute_risk(t)
        lines.append(
            f"- [ID {t.id}] {t.name} (deadline: {end_str}, status: {t.status}, risk: {risk})"
        )
    return "\n".join(lines)


def cosine_sim(a, b) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def embed_text(text: str):
    if not openai_client:
        return None
    resp = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return resp.data[0].embedding


def get_relevant_tasks(question: str, user: User, db: Session) -> List[Task]:
    q_lower = question.lower()
    base_q = db.query(Task).filter(Task.owner_id == user.id)

    now = datetime.now(timezone.utc)
    today = now.date()
    if "hôm nay" in q_lower:
        start = datetime(today.year, today.month, today.day, tzinfo=timezone.utc)
        end = start + timedelta(days=1)
        base_q = base_q.filter(Task.end_date >= start, Task.end_date < end)
    elif "tuần này" in q_lower:
        start = datetime(today.year, today.month, today.day, tzinfo=timezone.utc)
        end = start + timedelta(days=7)
        base_q = base_q.filter(Task.end_date >= start, Task.end_date < end)

    tasks = base_q.order_by(Task.created_at.desc()).limit(100).all()
    if not tasks:
        tasks = (
            db.query(Task)
            .filter(Task.owner_id == user.id)
            .order_by(Task.created_at.desc())
            .limit(50)
            .all()
        )

    if not openai_client or not tasks:
        return tasks

    try:
        q_emb = embed_text(question)
        if not q_emb:
            return tasks

        scored = []
        for t in tasks:
            text = t.name
            if t.end_date:
                text += f" (deadline {t.end_date.isoformat()})"
            t_emb = embed_text(text)
            s = cosine_sim(q_emb, t_emb) if t_emb else 0.0
            scored.append((s, t))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_k = [t for s, t in scored[:20]]
        return top_k if top_k else tasks
    except Exception:
        return tasks


def ask_llm(question: str, tasks: List[Task]) -> str:
    context = build_tasks_context(tasks)
    summary = summarize_tasks(tasks)

    if not openai_client:
        if not tasks:
            return (
                f"(Không dùng LLM) Không tìm thấy task nào để trả lời câu hỏi: '{question}'."
            )
        top_names = [t.name for t in tasks[:5]]
        return (
            f"(Không dùng LLM) Bạn hiện có {len(tasks)} task. "
            f"Overdue: {summary.overdue}, upcoming: {summary.upcoming}. "
            f"Một vài task: {', '.join(top_names)}"
        )

    system_prompt = (
        "Bạn là trợ lý quản lý task cá nhân. "
        "Bạn nhận danh sách task (có deadline + status + mức độ risk) và câu hỏi của user. "
        "- Hãy trả lời ngắn gọn, rõ ràng, dùng bullet points nếu phù hợp.\n"
        "- Nếu user hỏi về kế hoạch / ưu tiên, hãy đề xuất thứ tự làm việc.\n"
        "- Nếu câu hỏi không thể trả lời từ dữ liệu task, hãy nói rõ: "
        "'Không có trong dữ liệu task.'\n"
        "- Luôn bám sát thông tin task, không bịa ra task mới."
    )

    summary_text = (
        f"Tổng số task: {summary.total}, "
        f"overdue: {summary.overdue}, "
        f"sắp tới hạn: {summary.upcoming}, "
        f"không có deadline: {summary.no_deadline}, "
        f"risk cao: {summary.high_risk}, "
        f"risk TB: {summary.medium_risk}, "
        f"risk thấp: {summary.low_risk}."
    )

    user_prompt = (
        f"Danh sách task của user:\n{context}\n\n"
        f"Summary: {summary_text}\n\n"
        f"Câu hỏi: {question}"
    )

    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=500,
        )
        return resp.choices[0].message.content
    except Exception as e:
        if not tasks:
            return f"(Lỗi gọi LLM: {e}) Hiện không có task nào trong dữ liệu."
        top_names = [t.name for t in tasks[:5]]
        return (
            f"(Lỗi gọi LLM: {e}) Tạm thời mình chỉ có thể cho bạn biết bạn có {len(tasks)} task: "
            + ", ".join(top_names)
        )


def ask_llm_plan(tasks: List[Task]) -> str:
    context = build_tasks_context(tasks)
    if not openai_client:
        if not tasks:
            return "Không có task nào để lập kế hoạch. Hãy thêm task trước."
        top_names = [t.name for t in tasks[:5]]
        return (
            "Không dùng LLM, nhưng gợi ý: hãy ưu tiên các task gần deadline và có risk cao.\n"
            "Ví dụ, bắt đầu với: " + ", ".join(top_names)
        )

    system_prompt = (
        "Bạn là trợ lý lập kế hoạch trong ngày. "
        "Dựa trên các task (tên + deadline + status + risk), "
        "hãy tạo lịch làm việc cho hôm nay theo từng khung giờ (từ 8h đến 17h). "
        "Chia nhỏ thời gian hợp lý, ưu tiên task overdue / risk cao trước, "
        "và xen kẽ các nhiệm vụ nhẹ."
    )

    user_prompt = (
        "Dưới đây là danh sách task của user:\n"
        f"{context}\n\n"
        "Hãy tạo kế hoạch chi tiết cho hôm nay với dạng:\n"
        "- 08:00 - 09:00: ...\n"
        "- 09:00 - 10:00: ...\n"
        "..."
    )

    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=600,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"(Lỗi gọi LLM khi lập kế hoạch: {e})"


def clarify_task_name(task: Task) -> str:
    if not openai_client:
        return task.name

    system_prompt = (
        "Bạn là trợ lý viết lại tiêu đề công việc. "
        "Hãy viết lại tên task cho rõ ràng, cụ thể, dễ hiểu hơn, "
        "không vượt quá một câu ngắn."
    )
    user_prompt = f"Tên task hiện tại: '{task.name}'. Hãy viết lại tốt hơn."

    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=60,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return task.name
