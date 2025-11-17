# main.py
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.db import create_tables, run_migrations
from app.routes import router as api_router

app = FastAPI(title="Tasks API with Smart AI Agent", version="3.0.0")

# static UI
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root():
    return FileResponse("static/index.html")


@app.on_event("startup")
def on_startup():
    create_tables()
    run_migrations()


@app.get("/healthz")
def healthz():
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}


# include all API routes
app.include_router(api_router)
