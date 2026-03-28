# Phase 5: Supabase Migration & Alembic - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md -- this log preserves the alternatives considered.

**Date:** 2026-03-28
**Phase:** 05-supabase-migration-alembic
**Areas discussed:** Supabase connection & local dev, Migration strategy, Enum & type handling, Runtime ALTER TABLE removal

---

## Supabase Connection & Local Dev

| Option | Description | Selected |
|--------|-------------|----------|
| Supabase CLI (Recommended) | Run `supabase start` locally -- local PostgreSQL, no cloud dependency for dev | Yes |
| Cloud Supabase only | Always connect to cloud project, simpler but requires internet | |
| Both (CLI local + cloud prod) | CLI for local dev, cloud for production | |

**User's choice:** Supabase CLI
**Notes:** User specifically requested Supabase CLI usage.

| Option | Description | Selected |
|--------|-------------|----------|
| Direct connection (Recommended) | Port 5432 direct, no pooler quirks, `postgresql+asyncpg://` URL | Yes |
| Connection pooler (PgBouncer) | Port 6543 via PgBouncer, needs statement_cache_size=0 workaround | |
| You decide | Claude picks based on usage pattern | |

**User's choice:** Direct connection
**Notes:** App has single worker + API server, not serverless -- direct connection is appropriate.

---

## Migration Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Autogenerate from models (Recommended) | `alembic revision --autogenerate` from SQLAlchemy models, clean slate | Yes |
| Manual baseline migration | Hand-write initial migration SQL | |
| You decide | Claude picks based on schema complexity | |

**User's choice:** Autogenerate from models

| Option | Description | Selected |
|--------|-------------|----------|
| No data to migrate (Recommended) | Fresh start on Supabase, dev project with no production data | Yes |
| Yes, migrate existing data | Export MySQL data and import into Supabase | |

**User's choice:** No data to migrate
**Notes:** Dev project, fresh schema is sufficient.

---

## Enum & Type Handling

| Option | Description | Selected |
|--------|-------------|----------|
| Native PostgreSQL ENUMs (Recommended) | SQLAlchemy Enum() creates native PG ENUM types, type-safe | Yes |
| VARCHAR with CHECK constraints | Easier to modify values later, less type-safe | |
| You decide | Claude picks based on value stability | |

**User's choice:** Native PostgreSQL ENUMs

| Option | Description | Selected |
|--------|-------------|----------|
| Use text("false") (Recommended) | Change server_default="0" to server_default=text("false") | Yes |
| Use False literal | Use sa.false() dialect-agnostic approach | |
| You decide | Claude picks most idiomatic | |

**User's choice:** Use text("false")

---

## Runtime ALTER TABLE Removal

| Option | Description | Selected |
|--------|-------------|----------|
| Remove entirely (Recommended) | Delete all ALTER TABLE blocks, Alembic baseline creates the columns | Yes |
| Replace with Alembic check | Convert to proper Alembic migration that checks/adds columns | |
| You decide | Claude removes since models.py has columns defined | |

**User's choice:** Remove entirely
**Notes:** Columns already defined in models.py, Alembic baseline migration will create them.

---

## Claude's Discretion

- Alembic directory structure (backend/alembic/ vs backend/api/alembic/)
- Alembic env.py async configuration details
- Whether cryptography package is still needed after aiomysql removal

## Deferred Ideas

- ForeignKey declarations (DB-F01)
- Timezone-aware timestamps (DB-F02)
- Enum type evaluation for future changes (DB-F03)
