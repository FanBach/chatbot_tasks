from datetime import datetime, timedelta, timezone
from typing import Optional, List
import math
import os
import re

from dotenv import load_dotenv

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field, EmailStr, validator
from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    ForeignKey,
    create_engine,
    text,
)
from sqlalchemy.orm import relationship, sessionmaker, declarative_base, Session
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# ---------------------- Load .env ----------------------
load_dotenv()

# ---------------------- OpenAI Client (optional) ----------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
try:
    from openai import OpenAI

    openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except Exception:
    openai_client = None

# ---------------------- Config & Setup ----------------------
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://app_user:StrongPass_123!@localhost:5432/fastapi_tasks",
)
JWT_SECRET = os.getenv("JWT_SECRET", "dev_secret_change_me")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "60"))

ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@example.com")

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

app = FastAPI(title="Tasks API with Smart AI Agent", version="3.0.0")

# Serve static files (UI)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root():
    return FileResponse("static/index.html")


# ---------------------- Models ----------------------
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    tasks = relationship("Task", back_populates="owner", cascade="all, delete-orphan")
    chats = relationship(
        "ChatMessage", back_populates="owner", cascade="all, delete-orphan"
    )


class Task(Base):
    __tablename__ = "tasks"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    end_date = Column(DateTime(timezone=True), nullable=True)
    # status: todo / doing / done
    status = Column(String(20), nullable=False, default="todo")

    owner_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    owner = relationship("User", back_populates="tasks")


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), index=True)
    role = Column(String(10), nullable=False)  # "user" or "bot"
    content = Column(String, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    owner = relationship("User", back_populates="chats")


# ---------------------- Schemas ----------------------
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


# --------- Chat & Summary Schemas ---------
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


# ---------------------- Utils ----------------------
def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    Base.metadata.create_all(bind=engine)


def run_migrations():
    """Migration nhẹ nhàng cho cột status & bảng chat_messages."""
    with engine.connect() as conn:
        # Thêm cột status nếu chưa có
        try:
            conn.execute(
                text(
                    "ALTER TABLE tasks ADD COLUMN status VARCHAR(20) NOT NULL DEFAULT 'todo'"
                )
            )
        except Exception:
            # có thể đã tồn tại rồi, bỏ qua
            pass


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=JWT_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)


def get_current_user(
    db: Session = Depends(get_db), token: str = Depends(oauth2_scheme)
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id: Optional[int] = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.id == int(user_id)).first()
    if user is None:
        raise credentials_exception
    return user


# ---------------------- Smart Helpers ----------------------
def compute_risk(task: Task) -> str:
    """Heuristic risk: overdue / high / medium / low / no_deadline."""
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
    """Cosine similarity giữa 2 vector."""
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def embed_text(text: str):
    """Dùng OpenAI embedding nếu có key, nếu không thì None."""
    if not openai_client:
        return None
    resp = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return resp.data[0].embedding


def get_relevant_tasks(question: str, user: User, db: Session) -> List[Task]:
    """
    RAG mini with semantic search:
      - Luôn filter theo owner_id
      - Nếu có OpenAI key -> embedding + cosine similarity để chọn top-k task
      - Nếu không -> filter đơn giản + recent tasks
    """
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

    # Semantic search
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
            if t_emb:
                s = cosine_sim(q_emb, t_emb)
            else:
                s = 0.0
            scored.append((s, t))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_k = [t for s, t in scored[:20]]
        return top_k if top_k else tasks
    except Exception:
        return tasks


def ask_llm(question: str, tasks: List[Task]) -> str:
    """
    Gọi OpenAI GPT-4o-mini nếu có key, nếu không thì fallback trả lời heuristics.
    Thêm cả summary & risk vào prompt để AI gợi ý ưu tiên.
    """
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
                {
                    "role": "user",
                    "content": user_prompt,
                },
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
    """Sinh kế hoạch làm việc trong ngày từ danh sách task."""
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
    """Dùng LLM để viết lại tên task rõ ràng hơn."""
    if not openai_client:
        return task.name  # không làm gì

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


# ---------------------- Simple Agent (rule-based) ----------------------
def handle_chat_action(question: str, user: User, db: Session) -> Optional[str]:
    """
    Rule-based mini agent:
    Hỗ trợ:
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

        task = Task(
            name=name, end_date=end_date, status="todo", owner_id=user.id
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        deadline_msg = (
            task.end_date.isoformat()
            if task.end_date
            else "không có deadline cụ thể"
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

    # Chưa nhận dạng được action đặc biệt
    return None


def save_chat_message(db: Session, user: User, role: str, content: str) -> None:
    msg = ChatMessage(user_id=user.id, role=role, content=content)
    db.add(msg)
    db.commit()


# ---------------------- Auth & User Routes ----------------------
@app.post("/auth/register", response_model=UserOut, status_code=201)
def register(user_in: UserCreate, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == user_in.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    user = User(email=user_in.email, hashed_password=get_password_hash(user_in.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@app.post("/auth/login", response_model=Token)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect email or password")

    token = create_access_token({"sub": str(user.id)})
    return Token(access_token=token)


@app.get("/me", response_model=UserMeOut)
def get_me(current_user: User = Depends(get_current_user)):
    return UserMeOut(
        id=current_user.id,
        email=current_user.email,
        created_at=current_user.created_at,
        is_admin=current_user.email == ADMIN_EMAIL,
    )


@app.post("/auth/change_password")
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


# ---------------------- Task Routes ----------------------
@app.post("/tasks", response_model=TaskOut, status_code=201)
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


@app.get("/tasks", response_model=List[TaskOut])
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


@app.get("/tasks/{task_id}", response_model=TaskOut)
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


@app.put("/tasks/{task_id}", response_model=TaskOut)
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


@app.delete("/tasks/{task_id}", status_code=204)
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


# Clarify tên task bằng AI
@app.post("/tasks/{task_id}/clarify", response_model=TaskOut)
def clarify_task(
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

    new_name = clarify_task_name(task)
    task.name = new_name
    task.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(task)
    return task


# ---------------------- Summary & Reminders ----------------------
@app.get("/tasks/summary", response_model=TaskSummary)
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


@app.get("/tasks/reminders", response_model=List[TaskOut])
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


# AI Daily Plan
@app.get("/tasks/plan/today", response_model=PlanOut)
def plan_today(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    now = datetime.now(timezone.utc)
    # Chỉ lấy task chưa done & không quá hạn quá xa
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


# ---------------------- Chat & History ----------------------
@app.get("/chat/history", response_model=List[ChatMessageOut])
def get_chat_history(
    limit: int = 20,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    msgs = (
        db.query(ChatMessage)
        .filter(ChatMessage.user_id == current_user.id)
        .order_by(ChatMessage.created_at.asc())
        .limit(limit)
        .all()
    )
    return msgs


@app.post("/tasks/chat", response_model=ChatResponse)
def chat_about_tasks(
    payload: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # Lưu câu hỏi vào history
    save_chat_message(db, current_user, "user", payload.question)

    # 0. Thử xem câu hỏi là action (agent) không
    action_msg = handle_chat_action(payload.question, current_user, db)
    if action_msg:
        # Lưu câu trả lời agent
        save_chat_message(db, current_user, "bot", action_msg)
        tasks = (
            db.query(Task)
            .filter(Task.owner_id == current_user.id)
            .order_by(Task.created_at.desc())
            .limit(50)
            .all()
        )
        used_tasks = [
            ChatTask(id=t.id, name=t.name, end_date=t.end_date, status=t.status)
            for t in tasks
        ]
        return ChatResponse(answer=action_msg, used_tasks=used_tasks)

    # 1. RAG: lấy task liên quan
    tasks = get_relevant_tasks(payload.question, current_user, db)

    # 2. Gọi LLM / fallback
    answer = ask_llm(payload.question, tasks)

    # 3. Lưu câu trả lời
    save_chat_message(db, current_user, "bot", answer)

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


# ---------------------- Admin ----------------------
@app.get("/admin/overview", response_model=AdminOverview)
def admin_overview(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
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


# ---------------------- Lifecycle & Health ----------------------
@app.on_event("startup")
def on_startup():
    create_tables()
    run_migrations()


@app.get("/healthz")
def healthz():
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}
