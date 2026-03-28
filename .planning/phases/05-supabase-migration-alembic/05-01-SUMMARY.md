---
phase: 05-supabase-migration-alembic
plan: 01
subsystem: database
tags: [asyncpg, postgresql, sqlalchemy, supabase]

# Dependency graph
requires: []
provides:
  - "asyncpg database driver replacing aiomysql for PostgreSQL connectivity"
  - "PostgreSQL-compatible server_default values using text() wrappers"
  - "Clean router code with no runtime ALTER TABLE hacks"
  - "alembic dependency available in requirements.txt"
affects: [05-02-alembic-baseline-migration]

# Tech tracking
tech-stack:
  added: [asyncpg, alembic]
  patterns: ["text() wrapper for all server_default string/numeric literals in SQLAlchemy models"]

key-files:
  created: []
  modified:
    - backend/api/database.py
    - backend/api/models.py
    - backend/api/requirements.txt
    - backend/api/routers/worker.py
    - backend/api/routers/analyses.py

key-decisions:
  - "Accept DATABASE_URL as-is with no prefix conversion -- caller must provide valid async scheme"
  - "Add alembic dependency now so Plan 02 can use it immediately"

patterns-established:
  - "server_default=text('value') pattern for all PostgreSQL-compatible model defaults"
  - "Schema changes via Alembic migrations only, never runtime ALTER TABLE"

requirements-completed: [DB-01, DB-02, DB-04]

# Metrics
duration: 3min
completed: 2026-03-28
---

# Phase 05 Plan 01: Database Driver Swap Summary

**Switched from aiomysql/MySQL to asyncpg/PostgreSQL with text()-wrapped model defaults and zero runtime ALTER TABLE statements**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-28T11:56:06Z
- **Completed:** 2026-03-28T11:58:52Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- database.py accepts DATABASE_URL as-is with no mysql prefix conversion
- 4 server_default values in models.py wrapped with text() for PostgreSQL compatibility (User.role, Analysis.status, Analysis.progress, Analysis.skipCache)
- requirements.txt has asyncpg>=0.29.0 replacing aiomysql>=0.2.0, plus alembic>=1.13.0
- Removed _ensure_analysis_columns function and all ALTER TABLE blocks from worker.py and analyses.py

## Task Commits

Each task was committed atomically:

1. **Task 1: Swap database driver to asyncpg and fix model defaults** - `aca1177` (feat)
2. **Task 2: Remove all runtime ALTER TABLE hacks from routers** - `a6e1f72` (fix)

## Files Created/Modified
- `backend/api/database.py` - Async engine creation, removed mysql:// prefix conversion
- `backend/api/models.py` - Added text() wrapper to 4 server_default values
- `backend/api/requirements.txt` - Replaced aiomysql with asyncpg, added alembic
- `backend/api/routers/worker.py` - Removed _ensure_analysis_columns and its call
- `backend/api/routers/analyses.py` - Removed ALTER TABLE block in create_analysis

## Decisions Made
- Accept DATABASE_URL as-is with no prefix conversion -- caller must provide valid async scheme (postgresql+asyncpg://)
- Added alembic>=1.13.0 to requirements.txt now so Plan 02 can use it immediately

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## Known Stubs

None - all changes are complete implementations with no placeholders.

## User Setup Required

None - no external service configuration required. Users must ensure DATABASE_URL env var uses postgresql+asyncpg:// scheme when connecting to Supabase.

## Next Phase Readiness
- Backend configured for asyncpg with PostgreSQL-compatible models
- Clean model definitions ready for Alembic autogenerate baseline (Plan 02)
- Zero ALTER TABLE statements in any router file

---
*Phase: 05-supabase-migration-alembic*
*Completed: 2026-03-28*
