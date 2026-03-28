# Phase 5: Supabase Migration & Alembic - Context

**Gathered:** 2026-03-28
**Status:** Ready for planning

<domain>
## Phase Boundary

Migrate the backend database from Docker MySQL (aiomysql) to Supabase PostgreSQL (asyncpg), initialize Alembic for schema management, fix MySQL-specific model defaults/types for PostgreSQL compatibility, and remove all runtime ALTER TABLE hacks.

</domain>

<decisions>
## Implementation Decisions

### Supabase Connection & Local Dev
- **D-01:** Use Supabase CLI (`supabase start`) for local development. DATABASE_URL points to local PostgreSQL at `postgresql+asyncpg://postgres:postgres@localhost:54322/postgres`.
- **D-02:** Use direct connection (port 5432 for cloud, 54322 for local CLI). No PgBouncer pooler — avoids `statement_cache_size=0` workaround. App has single worker + API server, not serverless.
- **D-03:** Remove the mysql:// prefix conversion logic in `database.py`. Accept DATABASE_URL as-is (must be a valid SQLAlchemy async URL).

### Migration Strategy
- **D-04:** Initialize Alembic with `alembic revision --autogenerate` to create baseline migration from existing SQLAlchemy models. No hand-written DDL.
- **D-05:** No data migration needed — fresh schema on Supabase. This is a dev project with no production data to preserve.
- **D-06:** Alembic reads DATABASE_URL from environment (via `env.py`), not from a hardcoded `alembic.ini` value.

### Enum & Type Handling
- **D-07:** Keep native PostgreSQL ENUMs for UserRole, PipelineMode, ProcessingStatus. SQLAlchemy's `Enum()` already handles this. Values are stable and unlikely to change.
- **D-08:** Fix Boolean `server_default="0"` → `server_default=text("false")` for PostgreSQL compatibility. Also fix Integer `server_default="0"` → `server_default=text('0')` (PostgreSQL wants text-based defaults).
- **D-09:** TIMESTAMP columns with `func.now()` are already dialect-agnostic — no changes needed.

### Runtime ALTER TABLE Removal
- **D-10:** Remove all 3 ALTER TABLE blocks entirely (worker.py lines 39-43, analyses.py line 99). The `config` and `claimedBy` columns are already defined in models.py, so the Alembic baseline migration creates them.

### Claude's Discretion
- Exact Alembic directory structure (`backend/alembic/` vs `backend/api/alembic/`)
- Whether to add `alembic` as a script in backend/.env.example or just document in README
- `env.py` async configuration details (using SQLAlchemy async engine pattern)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Backend - Database
- `backend/api/database.py` — Current async engine creation with mysql:// prefix conversion. Must be rewritten for PostgreSQL.
- `backend/api/models.py` — 7 SQLAlchemy models. Lines 77, 82: MySQL-specific `server_default="0"` that must be fixed.
- `backend/api/config.py` — Settings class with DATABASE_URL env var.
- `backend/api/requirements.txt` — Has `aiomysql`, needs `asyncpg` + `alembic`.

### Backend - ALTER TABLE Hacks
- `backend/api/routers/worker.py` — Lines 39-43: runtime ALTER TABLE for `config` and `claimedBy` columns.
- `backend/api/routers/analyses.py` — Line 99: runtime ALTER TABLE for `config` column.

### Phase 4 Context
- `.planning/phases/04-manus-dependency-removal/04-CONTEXT.md` — Prior phase decisions (CORS config, env var patterns).

### Research
- `.planning/research/SUMMARY.md` — Stack context, identified 3 MySQL-specific patterns in models.py.
- `.planning/research/PITFALLS.md` — Pitfalls to avoid during migration.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `backend/api/config.py` Settings class: DATABASE_URL already read from env. Just needs the value to change.
- `backend/api/deps.py`: `get_db()` dependency injection — no changes needed if engine/session setup is compatible.

### Established Patterns
- Backend env vars: read via `os.getenv()` with defaults in Settings class
- All DB access goes through `async_session` from `database.py` → single point of change
- Models use `Mapped[]` + `mapped_column()` (modern SQLAlchemy 2.0 style)

### Integration Points
- `docker-compose.yml`: Currently starts MySQL. Will need updating or removal of MySQL service.
- `backend/.env.example`: Already has `DATABASE_URL` with MySQL placeholder — must update to PostgreSQL format.

</code_context>

<specifics>
## Specific Ideas

- Alembic `env.py` should use async engine pattern (SQLAlchemy 2.0 + asyncpg)
- `supabase/config.toml` may be needed for Supabase CLI local configuration
- Update `backend/.env.example` DATABASE_URL default to `postgresql+asyncpg://postgres:postgres@localhost:54322/postgres`
- Remove `cryptography` from requirements.txt if it was only needed for aiomysql SSL (check first)

</specifics>

<deferred>
## Deferred Ideas

- ForeignKey declarations for referential integrity (DB-F01 — future milestone)
- Timezone-aware TIMESTAMP columns (DB-F02 — future milestone)
- Native PostgreSQL enums vs VARCHAR+CHECK evaluation (DB-F03 — future milestone, decided native ENUMs for now)

</deferred>

---

*Phase: 05-supabase-migration-alembic*
*Context gathered: 2026-03-28*
