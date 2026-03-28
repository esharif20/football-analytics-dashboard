# Architecture Patterns

**Domain:** Football analytics dashboard -- Supabase migration and codebase hardening
**Researched:** 2026-03-28

## Current Architecture Overview

```
frontend/ (React 19 + Vite 6, wouter, TanStack Query)
  src/pages/Analysis.tsx   -- 2358-line monolith, 25+ inline components
  src/hooks/useWebSocket.ts
  src/lib/api-local.ts     -- REST client, all API calls

backend/api/ (FastAPI, async SQLAlchemy 2.0, aiomysql)
  database.py   -- engine creation, mysql:// prefix swap
  models.py     -- 7 tables: Users, Videos, Analyses, Events, Tracks, Statistics, Commentary
  deps.py       -- get_db() yields AsyncSession, get_current_user()
  auth.py       -- AutoLoginMiddleware (pure ASGI, reads from async_session directly)
  ws.py         -- in-memory subscriber registry, broadcast functions
  storage.py    -- local filesystem + ffmpeg re-encode
  routers/worker.py -- runtime ALTER TABLE hack, with_for_update(skip_locked=True)
  config.py     -- Settings class, reads from os.getenv

backend/pipeline/ (YOLOv8, ByteTrack -- OUT OF SCOPE for changes)
  worker.py     -- polls /api/worker/pending, runs subprocess, posts results back
```

### Data Flow (unchanged by migration)

```
Upload.tsx -> POST /api/upload/video -> storage_put() -> DB insert Video
           -> POST /api/analyses -> DB insert Analysis(status=pending)
           -> Navigate to /analysis/{id}, WS connects

Worker polls GET /api/worker/pending -> SELECT ... FOR UPDATE SKIP LOCKED
Worker runs pipeline subprocess
Worker POST /api/worker/upload-video (base64 video)
Worker POST /api/worker/analysis/{id}/complete (analytics JSON)
  -> strips large arrays, saves to Analysis, creates Statistic, broadcasts WS

Frontend re-fetches via TanStack Query, renders video + charts + stats
```

## Recommended Architecture After Migration

### Component Change Map

| Component | Responsibility | Communicates With | Changes Required |
|-----------|---------------|-------------------|------------------|
| `database.py` | Async engine + session factory | config.py, all routers via deps.py | **REWRITE**: swap aiomysql for asyncpg, new URL prefix logic |
| `models.py` | 7 SQLAlchemy ORM models | database.py (Base), all routers | **MODIFY**: TIMESTAMP -> DateTime, server_default fixes, Boolean defaults, JSON -> JSONB |
| `deps.py` | get_db(), get_current_user() | database.py, auth.py | **NO CHANGE**: already abstracts session creation |
| `auth.py` | AutoLoginMiddleware | database.py (async_session import) | **NO CHANGE**: uses async_session module-level import, still works |
| `ws.py` | WebSocket broadcast | database.py (async_session for auth) | **NO CHANGE**: in-memory registry, DB access only for auth check |
| `config.py` | Environment variable loading | .env file | **MODIFY**: DATABASE_URL default changes |
| `storage.py` | Local file storage | config.py | **NO CHANGE this milestone**: stays local filesystem |
| `worker.py` router | Worker endpoints, ALTER TABLE hack | models, deps, ws | **MODIFY**: remove `_ensure_analysis_columns()`, columns now in Alembic |
| `analyses.py` router | Analysis CRUD, also has ALTER TABLE | models, deps | **MODIFY**: remove duplicate `_ensure_analysis_columns()` call |
| `alembic/` | Schema migrations | models.py, database.py | **NEW**: entire directory, env.py, versions/ |
| `conftest.py` | Test fixtures | database.py, main.py | **REWRITE**: async fixtures with test DB, httpx.AsyncClient |
| `Analysis.tsx` | Results page monolith | api-local.ts, useWebSocket | **DECOMPOSE**: extract 25+ components into directory structure |
| `vitest.config.ts` | Frontend unit test config | vite.config.ts | **NEW**: vitest setup alongside existing Playwright |

---

## Change 1: Swap aiomysql for asyncpg in database.py

### What Changes

**Modified files:**
- `backend/api/database.py` -- URL prefix conversion logic (lines 7-9)
- `backend/api/config.py` -- DATABASE_URL handling
- `backend/api/requirements.txt` -- swap aiomysql for asyncpg, add psycopg2-binary

**Current URL conversion (database.py lines 7-9):**
```python
if _url.startswith("mysql://"):
    _url = _url.replace("mysql://", "mysql+aiomysql://", 1)
elif _url.startswith("mysql2://"):
    _url = _url.replace("mysql2://", "mysql+aiomysql://", 1)
```

**New URL conversion:**
```python
if _url.startswith("postgres://"):
    _url = _url.replace("postgres://", "postgresql+asyncpg://", 1)
elif _url.startswith("postgresql://"):
    _url = _url.replace("postgresql://", "postgresql+asyncpg://", 1)
```

Supabase connection strings use `postgres://` or `postgresql://` -- both need the `+asyncpg` driver suffix for SQLAlchemy async.

### Downstream Impact

| Downstream Component | Impact | Action |
|---------------------|--------|--------|
| `deps.py` | None -- async_session interface unchanged | No change |
| `auth.py` | None -- uses async_session import | No change |
| `ws.py` | None -- uses async_session import | No change |
| All routers | None -- all use `Depends(get_db)` | No change |
| Pipeline worker.py | None -- communicates via HTTP, not DB | No change |

### Model Compatibility Issues (models.py)

These SQLAlchemy constructs behave differently between MySQL and PostgreSQL. ALL must be fixed before the driver swap:

| Issue | Current (MySQL) | Required (PostgreSQL) | Affected Lines |
|-------|----------------|----------------------|----------------|
| `server_default="0"` on Boolean | MySQL accepts "0" as false | `server_default=text("false")` | Line 82 (skipCache) |
| `server_default="0"` on Integer | MySQL accepts string "0" | `server_default=text('0')` | Line 77 (progress) |
| `TIMESTAMP` column type | MySQL TIMESTAMP | Use `DateTime(timezone=True)` for clarity | All `createdAt`/`updatedAt`/`lastSignedIn` columns |
| `JSON` column type | MySQL JSON | PostgreSQL `JSONB` is better (indexable) | Lines 80, 111, 122-125, 153-158, 174 |
| `Enum(UserRole)` etc. | MySQL inline ENUM | PG creates separate TYPE object | Lines 43, 75-76 |

Types that work identically (no change needed): `Integer`, `String(N)`, `Text`, `Float`, `Boolean`, `autoincrement=True`, `func.now()`, `onupdate=func.now()`.

### Raw SQL in Routers

| Location | SQL | PG Compatible? | Action |
|----------|-----|---------------|--------|
| worker.py:39 | `ALTER TABLE analyses ADD COLUMN IF NOT EXISTS config JSON NULL` | Yes (PG 9.6+) | **Remove** -- Alembic manages schema |
| worker.py:43 | `ALTER TABLE analyses ADD COLUMN IF NOT EXISTS claimedBy VARCHAR(128) NULL` | Yes (PG 9.6+) | **Remove** -- Alembic manages schema |
| analyses.py:99 | Same as worker.py:39 | Yes | **Remove** |
| worker.py:65 | `.with_for_update(skip_locked=True)` | Yes (PG 9.5+) | **Keep** -- correct pattern |

### Engine Configuration for Supabase

Current: `create_async_engine(_url, echo=False, pool_pre_ping=True)`

Recommended additions:
- `pool_size=5, max_overflow=10` -- Supabase free tier has ~60 connection limit
- For direct connection (port 5432): no special args needed
- For pooler (port 6543): `connect_args={"statement_cache_size": 0}` -- REQUIRED to avoid prepared statement errors

---

## Change 2: Add Alembic Alongside Auto-Create Schema

### What Changes

**New files:**
- `backend/alembic.ini` -- Alembic config, points to migrations directory
- `backend/alembic/env.py` -- sync migration runner using psycopg2, imports Base metadata
- `backend/alembic/script.py.mako` -- migration template
- `backend/alembic/versions/001_initial_schema.py` -- baseline migration for all 7 tables

**Modified files:**
- `backend/api/routers/worker.py` -- remove `_ensure_analysis_columns()` (lines 36-46) and call at line 52
- `backend/api/routers/analyses.py` -- remove `_ensure_analysis_columns()` call (line 99 area)

### Architecture Decision: Sync Alembic with psycopg2

Alembic uses a **sync** psycopg2 connection, separate from the app's async asyncpg engine. Two connection strings:
- `alembic.ini`: `postgresql://user:pass@host:5432/db` (sync, psycopg2)
- `database.py`: `postgresql+asyncpg://user:pass@host:5432/db` (async, asyncpg)

This is the standard community pattern. Async Alembic env.py is fragile and unnecessary since migrations are CLI operations.

### Alembic env.py Wiring

```python
# alembic/env.py -- critical imports
from api.database import Base
from api.models import *  # noqa: F401,F403 -- register all models with Base.metadata
target_metadata = Base.metadata
```

The wildcard import ensures all 7 model classes register their tables in `Base.metadata`. Without it, `autogenerate` produces empty migrations.

### Migration Strategy

1. Fix models.py for PostgreSQL compatibility (Change 1 prerequisites)
2. `alembic init alembic/` -- scaffold
3. Configure env.py with psycopg2 URL and Base.metadata
4. `alembic revision --autogenerate -m "initial_schema"` -- captures all 7 tables + 3 enum types
5. Fresh Supabase DB: `alembic upgrade head`
6. Existing DB with tables: `alembic stamp head` (marks as current without running DDL)
7. Remove all `_ensure_analysis_columns()` code

### Directory Structure

```
backend/
  alembic.ini
  alembic/
    env.py              -- sync psycopg2, imports Base.metadata
    script.py.mako
    versions/
      001_initial_schema.py
  api/
    database.py         -- async asyncpg (Alembic does not touch this)
    models.py           -- source of truth for schema
```

---

## Change 3: Decompose Analysis.tsx Without Breaking Data Flow

### Current Structure (2358 lines, 1 file)

**Main exported component:** `Analysis()` (lines 129-843)
- Data fetching: 4 TanStack Query hooks (analysis, stats, events, commentary)
- State: activeTab, aiTab, filterTeam, selectedPlayer, realtimeProgress, videoRef
- WebSocket: useWebSocket hook for real-time progress

**25+ inline component functions** (never exported, share closure scope):
- UI helpers: `ChartTooltip`, `AnimatedSection`, `QuickStat`, `StatusBadge`, `ComingSoonCard`
- Processing: `ProcessingStatus`, `PipelinePerformanceCard`
- AI: `AICommentarySection` (has own useMutation), `EmptyCommentaryState`
- Visualizations: `PitchRadar`, `PlayerNode`, `HeatmapView`, `PassNetworkView`, `VoronoiView`, `BallTrajectoryDiagram`
- Stats: `PossessionDonut`, `TeamPerformanceRadar`, `StatsComparisonBar`, `StatRow`, `PlayerStatsTable`, `PlayerInteractionGraph`
- Tabs: `ModeSpecificTabs`, `EventTimeline`, `TeamShapeChart`, `DefensiveLineChart`, `PressingIntensityChart`

**1 React context:** `TeamColorsCtx` (line 88) -- provides team jersey colors derived from statistics

### Decomposition Strategy

**Critical constraint:** Data flows DOWN from the main `Analysis` component. No sub-component fetches its own data (except `AICommentarySection` which has a user-initiated mutation). This must be preserved.

**Proposed file structure:**
```
frontend/src/pages/Analysis/
  index.tsx                    -- main component: data fetching, state, layout (~300 lines)
  constants.ts                 -- PITCH_WIDTH, PITCH_HEIGHT, default colors
  context.ts                   -- TeamColorsCtx, useTeamColors()
  components/
    ChartTooltip.tsx
    AnimatedSection.tsx
    StatusBadge.tsx
    ProcessingStatus.tsx
    QuickStat.tsx
    StatRow.tsx
    PipelinePerformanceCard.tsx
    ComingSoonCard.tsx
    tabs/
      RadarTab.tsx             -- PitchRadar + PlayerNode
      StatsTab.tsx             -- PossessionDonut + TeamPerformanceRadar + StatsComparisonBar
      EventsTab.tsx            -- EventTimeline
      PlayersTab.tsx           -- PlayerStatsTable + PlayerInteractionGraph
      TacticalTab.tsx          -- TeamShapeChart + DefensiveLineChart + PressingIntensityChart
      AICommentaryTab.tsx      -- AICommentarySection + EmptyCommentaryState
      ModeSpecificTabs.tsx     -- tab router
    visualizations/
      PitchRadar.tsx
      HeatmapView.tsx
      PassNetworkView.tsx
      VoronoiView.tsx
      BallTrajectoryDiagram.tsx
      PossessionDonut.tsx
      TeamPerformanceRadar.tsx
      StatsComparisonBar.tsx
```

### Data Flow After Decomposition

```
Analysis/index.tsx
  |-- useQuery(analysisApi.get)     -> analysis object
  |-- useQuery(statisticsApi.list)  -> stats array
  |-- useQuery(eventsApi.list)      -> events array
  |-- useWebSocket(analysisId)      -> real-time progress
  |-- TeamColorsCtx.Provider
      |
      |-- <ProcessingStatus analysis={analysis} wsConnected={wsConnected} />
      |-- <ModeSpecificTabs mode={analysis.mode} activeTab={activeTab}>
      |     |-- <RadarTab trackingData={...} selectedPlayer={selectedPlayer} />
      |     |-- <StatsTab stats={stats} />
      |     |-- <EventsTab events={events} />
      |     |-- <PlayersTab analytics={...} events={events} />
      |     |-- <TacticalTab />
      |     |-- <AICommentaryTab analysisId={analysisId} />
      |
      |-- <video> element (stays in index.tsx -- ref needed for seek-to-event)
```

### Import Path Safety

The wouter route imports from `./pages/Analysis`. After decomposition:
- `Analysis.tsx` becomes `Analysis/index.tsx`
- Vite resolves `./pages/Analysis` to `./pages/Analysis/index.tsx` automatically
- Verify NO import has explicit `.tsx` extension (e.g., `from './pages/Analysis.tsx'`)

### What NOT to Change

- TanStack Query hooks stay in `index.tsx` -- do not distribute fetching
- WebSocket stays in `index.tsx` -- progress state passed as props
- `api-local.ts` untouched
- `useAuth` stays in `index.tsx`

---

## Change 4: Pytest Fixtures for Async FastAPI Testing

### Current State

- `conftest.py` -- only sys.path additions (15 lines)
- `test_tactical.py` -- tests commentary service (not API routes)
- No httpx.AsyncClient, no test DB, no dependency overrides

### Fixture Architecture

```python
# conftest.py -- key structure

# PRESERVE existing sys.path additions for test_tactical.py
import sys
from pathlib import Path
API_ROOT = Path(__file__).resolve().parent.parent
BACKEND_ROOT = API_ROOT.parent
for p in (str(API_ROOT), str(BACKEND_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

# NEW: async DB + client fixtures
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from api.database import Base
from api.deps import get_db
from api.main import app

@pytest_asyncio.fixture
async def db_engine():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()

@pytest_asyncio.fixture
async def db_session(db_engine):
    factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        yield session

@pytest_asyncio.fixture
async def client(db_session):
    async def override_get_db():
        yield db_session
    app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()
```

### Auth Middleware Interaction

The `AutoLoginMiddleware` imports `async_session` from database.py at module level. In tests:

1. `DATABASE_URL` is empty/unset -> `engine = None` (database.py line 13)
2. `async_session = None` (database.py line 15)
3. Middleware checks `if async_session is not None:` (auth.py line 48) -> skipped
4. Middleware creates `FallbackUser` (auth.py line 78)
5. Route handler uses overridden `get_db` -> test session

Both paths work independently. No conflict.

### SQLite Limitations

SQLite does not support: ENUM types (falls back to VARCHAR), JSONB (falls back to JSON text), `FOR UPDATE SKIP LOCKED` (raises error).

Tests for `worker/pending` (uses SKIP LOCKED) must run against PostgreSQL or mock the query. Unit tests with SQLite cover the other 90% of routes.

---

## Change 5: Vitest Alongside Playwright E2E

### Architecture

**New files:**
- `frontend/vitest.config.ts` -- test config
- `frontend/src/test/setup.ts` -- jsdom cleanup + jest-dom matchers
- `frontend/src/**/*.test.tsx` -- co-located component tests

**Existing (unchanged):**
- `playwright.config.ts` -- stays at project root
- `tests/e2e/*.spec.ts` -- Playwright E2E tests

### Test Boundary

| Layer | Tool | Pattern | Scope |
|-------|------|---------|-------|
| Unit | Vitest | `*.test.tsx` co-located | Individual components with props |
| Integration | Vitest + vi.mock | `*.test.tsx` in pages | Page components with mocked API |
| E2E | Playwright | `*.spec.ts` in tests/e2e | Full stack user flows |

### File Pattern Separation

- Vitest: `src/**/*.test.{ts,tsx}` -- inside frontend/src/
- Playwright: `tests/e2e/*.spec.ts` -- outside frontend/src/
- No collision possible with this pattern.

---

## Patterns to Follow

### Pattern 1: Dependency Injection via FastAPI Depends

All DB access in routers uses `Depends(get_db)`. This enables `dependency_overrides` in tests. The two exceptions (auth.py and ws.py importing `async_session` directly) are architecturally correct -- middleware and WebSocket handlers run outside DI scope.

### Pattern 2: Co-located Frontend Tests

`*.test.tsx` files live next to their component. Vitest discovers them automatically.

### Pattern 3: Alembic as Single Schema Authority

After Alembic setup, ALL schema changes go through `alembic revision --autogenerate`. No runtime ALTER TABLE. The `_ensure_analysis_columns()` hack is removed.

### Pattern 4: Props-Down Data Flow

Only `Analysis/index.tsx` performs data fetching. All children receive data as props. Prevents waterfalls, duplicates, and inconsistent states.

---

## Anti-Patterns to Avoid

### Anti-Pattern 1: Distributing Data Fetching

Moving `useQuery` into sub-components causes waterfalls and duplicates. Keep all fetching in `index.tsx`.

### Anti-Pattern 2: Testing Against Real Supabase

Slow, flaky, pollutes data. Use SQLite for unit tests, isolated PG for integration.

### Anti-Pattern 3: Dual Database Support

Do not keep MySQL URL conversion. This is a one-way migration. Remove MySQL code.

### Anti-Pattern 4: Production Migrations in App Startup

Running `alembic upgrade head` in FastAPI lifespan causes race conditions with multiple workers. Migrate in CI/CD.

### Anti-Pattern 5: Async Alembic env.py

Unnecessary complexity. Sync psycopg2 for Alembic CLI is the standard.

---

## Build Order (Dependency-Aware)

### Dependency Graph

```
(1) asyncpg swap + model fixes -----> (2) Alembic (needs PG-compatible models)
                                  \
                                   \-> (4) pytest fixtures (needs DB layer established)

(3) Analysis.tsx decomposition  (fully independent -- frontend only)

(5) vitest setup  (independent, easier after decomposition)
```

### Recommended Build Sequence

**Phase 1: Database Layer (Changes 1 + 2)**

Must be sequential within the phase:
1. Fix `models.py` -- TIMESTAMP, Boolean defaults, JSON -> JSONB
2. Rewrite `database.py` -- asyncpg URL conversion, pool config
3. Update `requirements.txt` -- remove aiomysql, add asyncpg + psycopg2-binary + alembic
4. `alembic init` + configure env.py + generate initial migration
5. Remove `_ensure_analysis_columns()` from worker.py and analyses.py
6. Smoke test against Supabase

**Phase 2: Backend Testing (Change 4)**

Depends on Phase 1 being complete:
1. Rewrite conftest.py (preserve sys.path for existing tests)
2. Add pytest-asyncio, httpx, aiosqlite to requirements
3. Write tests for critical paths
4. Add pytest to CI

**Phase 3: Frontend Decomposition (Change 3)**

Independent -- can run in parallel with Phases 1-2:
1. Create Analysis/ directory structure
2. Extract constants + context
3. Extract visualizations (pure rendering -- safest)
4. Extract tab containers
5. Wire up index.tsx
6. Verify Playwright E2E tests pass

**Phase 4: Frontend Unit Testing (Change 5)**

Benefits from Phase 3 (decomposed components are testable):
1. Add vitest + @testing-library to devDependencies
2. Create vitest.config.ts + setup.ts
3. Write unit tests for extracted components
4. Add vitest to CI

### Why This Order

- **1 before 2:** Alembic autogenerate reads models.py. Models must have PG types first.
- **1+2 before 4:** Test fixtures depend on the DB driver decision.
- **3 before 5:** A 2358-line monolith cannot be meaningfully unit tested. Decomposition unlocks testability.
- **3 is independent of 1+2:** Zero backend dependency. Can run in parallel.

---

## Scalability Considerations

| Concern | Current (MySQL Docker) | After Migration (Supabase) |
|---------|----------------------|---------------------------|
| Connection pooling | Unlimited local | Free tier: ~60 connections. pool_size=5, max_overflow=10 |
| File storage | Local filesystem | Still local this milestone |
| WebSocket | In-memory, single process | Still single process |
| Worker concurrency | SKIP LOCKED | Same on PostgreSQL |
| Schema management | Runtime ALTER TABLE | Alembic CLI (versioned, safe) |

## Sources

- SQLAlchemy 2.0 async docs -- HIGH confidence
- PostgreSQL ADD COLUMN IF NOT EXISTS (PG 9.6+) -- HIGH confidence
- PostgreSQL FOR UPDATE SKIP LOCKED (PG 9.5+) -- HIGH confidence
- FastAPI dependency_overrides -- HIGH confidence (official docs)
- Supabase pooler + asyncpg incompatibility -- MEDIUM confidence (training data, verify with current docs)
- Alembic sync vs async patterns -- HIGH confidence (standard community practice)
