---
milestone: v0.3
status: active
version: 1.1
---

# Roadmap

## Milestone v0.1 Objectives (Complete)

- Planning artifacts created and codebase mapped
- Development environments reproducible (frontend + backend)
- CV pipeline <-> API <-> frontend flow validated and gaps documented

## Milestone v0.2 Codebase Hardening & Supabase Migration

Remove all Manus platform dependencies, migrate from Docker MySQL to Supabase PostgreSQL, harden codebase with testing, linting, security, and SWE best practices -- all existing functionality preserved.

## Phase Breakdown

| Phase | Name | Goal | Requirements | Status |
|-------|------|------|--------------|--------|
| 4 | Manus Dependency Removal | Application runs with zero references to the Manus platform | MANUS-01, MANUS-02, MANUS-03, MANUS-04, DB-05 | complete |
| 5 | Supabase Migration & Alembic | Data persists to Supabase PostgreSQL with managed schema migrations | DB-01, DB-02, DB-03, DB-04 | complete |
| 6 | Frontend Decomposition & Code Quality | Codebase follows maintainable patterns with security guardrails | QUAL-01, QUAL-02, QUAL-03, QUAL-04, QUAL-05 | complete |
| 7 | Testing, Linting & CI | Every code change validated by automated tests, linting, and CI | TEST-01, TEST-02, TEST-03, QUAL-06, QUAL-07 | complete |
| 8 | Database Redesign & Time-Series Tracks | 1/3 | In Progress|  |

## Phase Details

### Phase 4: Manus Dependency Removal

**Goal:** Application runs with zero references to the Manus platform
**Requirements:** MANUS-01, MANUS-02, MANUS-03, MANUS-04, DB-05

**Success Criteria:**
1. Pipeline downloads ML models from URLs specified in env vars, not from manuscdn.com
2. Home page hero images load from local assets with no network requests to manuscdn.com
3. Auth persistence uses "football-dashboard-user" localStorage key (old key ignored)
4. A developer cloning the repo can copy .env.example and have all required env vars documented
5. CORS origins are configurable via CORS_ORIGINS env var (not hardcoded)

### Phase 5: Supabase Migration & Alembic

**Goal:** Data persists to Supabase PostgreSQL with managed schema migrations
**Requirements:** DB-01, DB-02, DB-03, DB-04

**Success Criteria:**
1. Application connects to Supabase PostgreSQL via asyncpg and all CRUD operations work
2. All 7 database tables use PostgreSQL-compatible types and defaults
3. Running `alembic upgrade head` on a fresh database creates the complete schema
4. No raw ALTER TABLE statements execute at runtime in any router

### Phase 6: Frontend Decomposition & Code Quality

**Goal:** Codebase follows maintainable patterns with security guardrails
**Requirements:** QUAL-01, QUAL-02, QUAL-03, QUAL-04, QUAL-05

**Success Criteria:**
1. Analysis page renders identically but Analysis.tsx is under 400 lines with sub-components in pages/analysis/
2. The dead base64 upload function no longer exists in api-local.ts
3. `next-themes` does not appear in package.json dependencies
4. Backend refuses to start in production mode if JWT_SECRET equals "dev-secret"
5. AutoLogin middleware only activates when LOCAL_DEV_MODE=true is explicitly set

### Phase 7: Testing, Linting & CI

**Goal:** Every code change validated by automated tests, linting, and CI
**Requirements:** TEST-01, TEST-02, TEST-03, QUAL-06, QUAL-07
**Plans:** 4/4 plans complete

Plans:
- [x] 07-01-PLAN.md — Backend ruff linting config + pytest HTTP endpoint smoke tests
- [x] 07-02-PLAN.md — Frontend ESLint + Prettier config + vitest unit tests
- [x] 07-03-PLAN.md — CI pipeline expansion: fix frontend cache bug + add backend job

**Success Criteria:**
1. `pytest` passes with tests covering health, upload, analysis, worker, and commentary endpoints
2. `vitest` passes with tests covering key frontend components and hooks
3. `ruff check backend/` passes with zero violations on the configured ruleset
4. `eslint frontend/src/` and `prettier --check frontend/src/` pass with zero violations
5. CI pipeline runs backend lint + test job and blocks merge on failure

### Phase 8: Database Redesign & Time-Series Tracks

**Goal:** Schema has referential integrity, proper indexes, RLS, and per-frame tracking data populates the tracks table
**Requirements:** DB-R01, DB-R02, DB-R03, DB-R04, DB-R05, DB-R06, DB-R07
**Plans:** 1/3 plans executed

Plans:
- [x] 08-01-PLAN.md — Supabase migration SQL (FK constraints, indexes, RLS) + SQLAlchemy ForeignKey declarations
- [ ] 08-02-PLAN.md — Pipeline export_tracks_json() function + call in all.py
- [ ] 08-03-PLAN.md — Worker POST /tracks endpoint + paginated GET /tracks + pipeline worker batched upload

**Success Criteria:**
1. All 7 application tables have foreign key constraints with ON DELETE CASCADE
2. Indexes exist on analysisId (events, tracks, statistics, commentary), videoId (analyses), userId (videos, analyses)
3. Running `supabase db reset` applies all migrations cleanly on a fresh DB
4. After a completed RunPod analysis, SELECT COUNT(*) FROM tracks WHERE analysis_id = {id} returns > 0
5. RLS policies exist: authenticated users can only SELECT/INSERT/UPDATE/DELETE their own rows in users, videos, analyses
6. SQLAlchemy models in backend/api/models.py match the migrated schema exactly
7. Backend API continues to pass all existing pytest tests after schema changes
