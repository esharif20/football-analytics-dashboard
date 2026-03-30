---
phase: 08-database-redesign-time-series-tracks
plan: "01"
subsystem: database
tags: [supabase, postgresql, sqlalchemy, migrations, foreign-keys, rls, indexes]

# Dependency graph
requires:
  - phase: 05-supabase-migration
    provides: Alembic baseline migration creating all 7 tables in Supabase PostgreSQL
provides:
  - Supabase migration file with FK constraints, indexes, and permissive RLS
  - SQLAlchemy models with explicit ForeignKey declarations on all 8 FK columns
affects: [08-02, 08-03, backend-api]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Idempotent PostgreSQL migrations using DO $$ BEGIN ... EXCEPTION WHEN duplicate_object THEN NULL; END $$ blocks"
    - "Permissive RLS (USING true) as infrastructure placeholder before strict JWT-based row-ownership"
    - "SQLAlchemy ForeignKey with ondelete string matches migration ON DELETE action"

key-files:
  created:
    - supabase/migrations/20260330000001_phase8_schema_redesign.sql
  modified:
    - backend/api/models.py

key-decisions:
  - "Permissive RLS (USING true) rather than strict: existing auto-login flow has no JWT; strict RLS deferred until Supabase Auth JWT integration"
  - "Commentary.eventId uses ON DELETE SET NULL (not CASCADE) so commentary survives event deletion"
  - "Migration operates via ALTER TABLE on existing Alembic-created tables, not CREATE TABLE, to avoid conflicts with Alembic baseline"

patterns-established:
  - "Pattern 1: All FK column declarations in SQLAlchemy models use ForeignKey(table.id, ondelete=ACTION) matching migration constraint"
  - "Pattern 2: Composite index on (analysisId, frameNumber) for tracks table — template for time-series frame queries"

requirements-completed: [DB-R01, DB-R02, DB-R03, DB-R04]

# Metrics
duration: 5min
completed: 2026-03-30
---

# Phase 08 Plan 01: Schema Redesign Summary

**PostgreSQL FK constraints with ON DELETE CASCADE on all 7 relationships, 8 performance indexes, permissive RLS on users/videos/analyses, and SQLAlchemy ForeignKey declarations aligned to migration DDL**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-03-30T10:05:29Z
- **Completed:** 2026-03-30T10:10:00Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Created idempotent Supabase migration with 7 FK constraints (ON DELETE CASCADE) and 1 nullable FK (ON DELETE SET NULL for commentary.eventId)
- Added 8 performance indexes including composite idx_tracks_analysis_frame for time-series frame queries
- Enabled RLS on users, videos, analyses with permissive USING (true) policies as infrastructure placeholder
- Updated all 7 FK columns in SQLAlchemy models with explicit ForeignKey declarations matching migration DDL
- All 40 existing backend pytest tests pass after model changes

## Task Commits

Each task was committed atomically:

1. **Task 1: Write Supabase migration SQL** - `cb3a41f` (feat)
2. **Task 2: Add ForeignKey declarations to SQLAlchemy models** - `ad8799d` (feat)

## Files Created/Modified
- `supabase/migrations/20260330000001_phase8_schema_redesign.sql` - Complete DDL: FK constraints, 8 indexes, RLS enable + permissive policies
- `backend/api/models.py` - ForeignKey added to import and all 8 FK column declarations

## Decisions Made
- Permissive RLS (USING true) chosen over strict row-ownership: current auto-login flow produces no Supabase Auth JWT, so strict USING (auth.uid()::text = "openId") would block all queries. RLS infrastructure is in place; strict enforcement deferred to Supabase Auth JWT integration.
- Commentary.eventId uses ON DELETE SET NULL rather than CASCADE: commentary should survive if an individual event is deleted.
- Migration uses ALTER TABLE (not CREATE TABLE) since Alembic baseline already created the tables; idempotent DO blocks handle re-runs cleanly.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None - `python` not in PATH on this machine; used `python3` for pytest run. All 40 tests passed.

## User Setup Required
None - no external service configuration required. Migration file must be applied manually via `supabase db reset` or `supabase migration up` against the target Supabase project.

## Next Phase Readiness
- FK constraints and indexes provide the structural foundation for Plans 08-02 and 08-03
- SQLAlchemy models now accurately reflect database referential integrity
- No blockers for subsequent plans in Phase 08

---
*Phase: 08-database-redesign-time-series-tracks*
*Completed: 2026-03-30*
