## Quick orientation

This small repo is a single-file FastAPI monolith implementing JWT auth and a Tasks CRUD API.

Key files
- `hello.py` — full application (models, schemas, auth, routes) in a single module. Read this first.
- `docker-compose.yml` — brings up Postgres and pgAdmin for integration testing.
- `dec12_18features.csv` — a dataset in the repo, not referenced by the app.

Big-picture architecture
- Monolithic FastAPI app: HTTP endpoints -> Pydantic schemas -> SQLAlchemy models -> DB.
- Auth: JWT tokens generated in `create_access_token` and validated in `get_current_user` (`jose.jwt`).
- DB: configured via `DATABASE_URL` env var in `hello.py`. By default the code points at Postgres; docstring mentions SQLite but the DB engine uses the env var.
- Lifecycle: `create_tables()` runs at startup (see `@app.on_event("startup")`).

How to run (Windows PowerShell examples)
1) Create & activate venv:

```powershell
python -m venv venv
venv\Scripts\activate
```

2) Install dependencies used by the app (no requirements.txt is present):

```powershell
pip install fastapi uvicorn[standard] SQLAlchemy passlib python-jose[cryptography] python-dotenv
```

3) Run the app (note module name is `hello`, not `main`):

```powershell
uvicorn hello:app --reload
```

4) Use Docker Compose for a Postgres-backed dev DB:

```powershell
docker-compose up -d
# then set DATABASE_URL to point to the Postgres instance before starting the app
```

Environment variables the app reads (via `dotenv` in `hello.py`):
- `DATABASE_URL` — SQLAlchemy URL (Postgres default in file). Example:

```
DATABASE_URL=postgresql+psycopg2://postgres:postgres@localhost:5432/fastapi_tasks
JWT_SECRET=my_super_secret_key
JWT_EXPIRE_MINUTES=60
```

Project-specific coding patterns & tips for AI agents
- Single-file layout: prefer small, focused edits to `hello.py`. Search symbol names (`Task`, `User`, `create_access_token`) to find all occurrences.
- Pydantic models use `class Config: from_attributes = True` (attribute-based conversion). Keep this style when adding response models.
- Password hashing deliberately uses `pbkdf2_sha256` instead of bcrypt to avoid Windows bcrypt backend issues — continue with that scheme for new password helpers.
- Datetimes are validated to be timezone-aware via validators in Task schemas; preserve timezone-aware semantics when adding date/time fields.
- The app calls `create_tables()` on startup. Adding new models requires restarting the server to have SQLAlchemy create new tables (or run migration tooling if you introduce Alembic).

Concrete code pointers
- Endpoints: `/auth/register`, `/auth/login`, `/tasks` (see route decorators in `hello.py`).
- JWT: token `sub` is set to `str(user.id)`; `get_current_user` expects that value to find the User in DB.
- DB engine: `engine = create_engine(DATABASE_URL, pool_pre_ping=True)`. If you switch to SQLite, add `connect_args={"check_same_thread": False}` as noted in comments of `hello.py`.

Editing rules for safe changes
- Keep behavior backward-compatible: don't change JWT claims, DB column names, or response shapes unless you update client code/tests.
- For schema changes that affect responses, update the Pydantic response models (they use `from_attributes`) and ensure columns exist in the SQLAlchemy model.

Missing artifacts & notes
- There is no `requirements.txt` or test suite in the repo. The top-level docstring in `hello.py` mentions `main:app` — use `hello:app` instead.
- If you add tests or CI, prefer simple pytest tests that exercise `/auth` and `/tasks` endpoints using the TestClient from FastAPI.

If anything here is unclear or you want the file to be more/less prescriptive (for example: enforce type/style rules, add example curl/PowerShell snippets, or include minimal test templates), tell me what to add or change.
