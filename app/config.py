# app/config.py
import os
from datetime import timedelta
from dotenv import load_dotenv

load_dotenv()

# Database
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://app_user:StrongPass_123!@localhost:5432/fastapi_tasks",
)

# JWT
JWT_SECRET = os.getenv("JWT_SECRET", "dev_secret_change_me")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "60"))

ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@example.com")

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
