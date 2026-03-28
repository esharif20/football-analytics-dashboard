# Research Summary: Football Analytics Dashboard v0.2 Migration & Hardening

**Domain:** Database migration + codebase quality tooling
**Researched:** 2026-03-28
**Overall confidence:** HIGH

## Executive Summary

The migration from MySQL/aiomysql to Supabase PostgreSQL/asyncpg is a clean driver swap at the database.py level, but requires prerequisite fixes in models.py for PostgreSQL compatibility. Specifically: `server_default="0"` on Boolean/Integer columns must become `text("false")`/`text('0')`, `TIMESTAMP` should become `DateTime(timezone=True)`, and `JSON` should become `JSONB`. These are small changes but must happen BEFORE Alembic autogenerate runs, or the generated migration will bake in MySQL assumptions.

The existing architecture is well-layered for this migration. All routers use `Depends(get_db)` for database access (no direct engine imports), which means the driver swap is invisible to business logic. The two components that import `async_session` directly -- `auth.py` (middleware) and `ws.py` (WebSocket auth) -- work unchanged because the session interface is identical between aiomysql and asyncpg.

The Analysis.tsx decomposition (2358 lines, 25+ inline components) is the largest single change but is completely independent of the backend migration. The decomposition preserves the existing data flow: parent fetches all data via TanStack Query, children receive data as props. The `TeamColorsCtx` context already demonstrates the pattern for shared state.

The testing infrastructure has zero foundation -- no test DB, no async fixtures, no frontend unit tests. Building this from scratch requires careful attention to the auth middleware interaction: it imports `async_session` at module level, but when `DATABASE_URL` is unset, it falls back to a `FallbackUser` dataclass. This fallback path is the key to making tests work without a real database in the middleware layer.

## Key Findings

- **Database swap touches 3 files** (database.py, models.py, requirements.txt) + removing raw ALTER TABLE SQL from 2 routers. Zero changes to deps.py, auth.py, ws.py, storage.py, or any frontend code.
- **models.py has 3 MySQL-specific patterns** that break on PostgreSQL: Boolean `server_default="0"`, Integer `server_default="0"`, and `TIMESTAMP` type. All must be fixed before Alembic autogenerate.
- **Analysis.tsx decomposition** maps to ~30 new files but changes zero API calls or data flows. The wouter route resolves `Analysis/index.tsx` automatically.
- **Test fixture architecture** relies on FastAPI `dependency_overrides[get_db]` for routes and `FallbackUser` fallback for the auth middleware -- two independent paths that do not conflict.
- **SQLite works for 90% of unit tests** but cannot test `FOR UPDATE SKIP LOCKED` (worker/pending endpoint) or JSONB operations.

## Implications for Roadmap

Based on research, suggested phase structure:

1. **Database Migration (Changes 1 + 2)** - Everything depends on a working DB layer
   - Fix models.py for PG compatibility, rewrite database.py, init Alembic, remove ALTER TABLE hacks
   - Addresses: asyncpg swap, schema management, connection config
   - Avoids: Pitfall 2 (Boolean defaults), Pitfall 3 (autogenerate order)
   - Risk: Supabase pooler vs direct connection -- use port 5432

2. **Backend Testing (Change 4)** - Validates migration, catches regressions
   - Rewrite conftest.py with async fixtures, add httpx.AsyncClient, write core route tests
   - Addresses: zero test coverage for API endpoints
   - Avoids: Pitfall 4 (auth middleware conflict) by leveraging FallbackUser path
   - Depends on: Phase 1 (DB layer must be settled)

3. **Frontend Decomposition (Change 3)** - Independent, can run in parallel with Phase 1
   - Extract 25+ components from Analysis.tsx into directory structure
   - Addresses: 2358-line monolith blocking frontend testability
   - Avoids: Pitfall 6 (import path) by using directory index.tsx
   - No backend dependency

4. **Frontend Unit Testing (Change 5)** - Benefits from decomposition
   - Add vitest + Testing Library, write component tests
   - Addresses: zero frontend unit test coverage
   - Avoids: Pitfall 11 (test file collision) by separating *.test.tsx from *.spec.ts
   - Benefits from: Phase 3 (decomposed components are testable)

**Phase ordering rationale:**
- Phase 1 MUST come first: Alembic needs PG-compatible models, test fixtures need the DB driver decided
- Phase 2 depends on Phase 1: cannot write meaningful DB tests without knowing the driver
- Phase 3 is fully independent: zero backend files touched, can run in parallel with Phase 1
- Phase 4 benefits from Phase 3: testing a 2358-line monolith is nearly impossible vs testing individual components
- Phases 1+2 and Phase 3 can run on separate branches simultaneously

**Research flags for phases:**
- Phase 1: Verify Supabase SSL handling with asyncpg (may need `connect_args={"ssl": "require"}`)
- Phase 2: Decide SQLite-only vs SQLite+PG for test tiers
- Phase 3: Standard refactoring, no research needed
- Phase 4: Standard tooling setup, no research needed

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack (asyncpg, Alembic) | HIGH | Official SQLAlchemy docs, well-established patterns |
| Stack (testing tools) | HIGH | FastAPI official recommendation, Vite ecosystem standard |
| Architecture integration | HIGH | All integration points verified against actual codebase files |
| Model compatibility | HIGH | Every server_default, type, and raw SQL statement audited |
| Analysis.tsx decomposition | HIGH | All 25+ components catalogued with data dependencies |
| Auth middleware in tests | HIGH | Traced full code path through auth.py fallback logic |
| Pitfalls | HIGH | Supabase pooler issue confirmed by multiple community sources |

## Gaps to Address

- Supabase SSL configuration with asyncpg: may need `connect_args` depending on Supabase project settings
- Whether to use `native_enum=True` or `native_enum=False` for PostgreSQL enums -- tradeoff between DB enforcement and migration simplicity
- Exact vitest version compatibility with React 19 -- likely fine but unverified with current docs
- Security hardening details (JWT validation, CORS from env) deferred to implementation phase
