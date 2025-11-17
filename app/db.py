# app/db.py
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base, Session

from .config import DATABASE_URL

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    from . import models  # ensure models are imported
    Base.metadata.create_all(bind=engine)


def run_migrations():
    """Migration nhẹ nhàng cho cột status & bảng chat_messages (nếu cần)."""
    with engine.connect() as conn:
        try:
            conn.execute(
                text(
                    "ALTER TABLE tasks ADD COLUMN status VARCHAR(20) NOT NULL DEFAULT 'todo'"
                )
            )
        except Exception:
            # cột đã tồn tại thì bỏ qua
            pass
