# Phase 7: Testing, Linting & CI - Research

**Researched:** 2026-03-28
**Domain:** pytest, vitest, ruff, ESLint/Prettier, GitHub Actions
**Confidence:** HIGH

---

## Project Constraints (from CLAUDE.md)

- Do what has been asked; nothing more, nothing less
- NEVER create files unless absolutely necessary
- ALWAYS prefer editing an existing file to creating a new one
- Use `/tests` for test files
- NEVER commit secrets, credentials, or .env files
- ALWAYS run tests after making code changes
- ALWAYS verify build succeeds before committing
- Files under 500 lines

---

## 1. Current State (What Exists)

### Backend Tests

**Location:** `backend/api/tests/`
**Files:**
- `conftest.py` — sys.path fixture only; adds `backend/api/` and `backend/` to path. No DB fixtures, no TestClient fixture, no async session override.
- `test_tactical.py` — 30+ unit tests covering:
  - `_load_analytics_data` (commentary router helper): 4 tests
  - `GroundingFormatter` (tactical service): 16 tests
  - `TacticalAnalyzer` (tactical service): 8 tests
  - `StubProvider` / `get_provider` (llm_providers): 4 tests
  - All tests use `asyncio.run()` directly, not `pytest-asyncio`

**Location:** `backend/pipeline/test_analytics.py`
- Standalone test file at pipeline root (not in `tests/` subdir)
- Tests bbox_utils, kinematics, possession, events analytics modules
- Mocks `cv2`, `supervision`, `ultralytics`, `torch` at module level
- Uses `sys.path.insert` to add `src/` dir

**What is NOT tested:**
- No HTTP endpoint tests (no `TestClient` or `AsyncClient` usage anywhere)
- No health check test
- No upload endpoint test
- No analysis CRUD endpoint tests
- No worker endpoint tests (`/pending`, `/complete`, `/upload-video`)
- No auth endpoint tests
- No WebSocket tests

### Backend Dependencies

**`backend/api/requirements.txt`** contains:
- `pytest>=8.0.0` — installed
- `fastapi>=0.104.0` — installed
- `httpx` — NOT listed (required for `AsyncClient`)
- `pytest-asyncio` — NOT listed
- `anyio` — NOT listed (pulled in transitively by httpx/starlette but not explicit)

### Backend Routers Inventory

All mounted under `/api` prefix in `main.py`:

| Router | File | Key Endpoints |
|--------|------|---------------|
| system | `system.py` | `GET /api/health` |
| auth | `main.py` inline | `POST /api/auth/login`, `GET /api/auth/me`, `POST /api/auth/logout` |
| upload | `main.py` inline | `POST /api/upload/video` (multipart) |
| videos | `videos.py` | `GET /api/videos`, `GET /api/videos/{id}`, `POST /api/videos/upload-base64`, `DELETE /api/videos/{id}` |
| analyses | `analyses.py` | `GET /api/analysis`, `POST /api/analysis`, `GET /api/analysis/{id}`, `PUT /api/analysis/{id}/status`, `POST /api/analysis/{id}/terminate`, `GET /api/analysis/{id}/eta` |
| worker | `worker.py` | `GET /api/worker/pending`, `POST /api/worker/analysis/{id}/status`, `POST /api/worker/analysis/{id}/complete`, `POST /api/worker/upload-video` |
| commentary | `commentary.py` | `GET /api/commentary/types`, `GET /api/commentary/{id}`, `POST /api/commentary/{id}` |
| events | `events.py` | (read) |
| tracks | `tracks.py` | (read) |
| stats | `stats.py` | (read) |
| test_support | `test_support.py` | Conditional — only when `LOCAL_DEV_MODE=true` |

**Async DB pattern:** All routes use `AsyncSession` via `get_db` dependency in `deps.py`. Tests must override this dependency to avoid needing a live Supabase connection.

### Frontend Tests

**No frontend unit tests exist.** Search confirmed zero `.test.ts`, `.test.tsx`, `.spec.ts`, `.spec.tsx` files under `frontend/src/`.

**No vitest setup exists** — `frontend/package.json` has no `vitest`, `@testing-library/react`, `@testing-library/user-event`, or `jsdom` in deps. No `vitest.config.ts` at project root.

**Test scripts in `package.json`:** None. Only `dev`, `build`, `preview`, `check`.

### E2E Tests (Playwright)

**Location:** `tests/e2e/`
**Files:**
- `upload.spec.ts` — upload page render, file rejection toast, API rejection
- `analysis.spec.ts` — analysis page loads for seeded completed analysis
- `dashboard.spec.ts` — dashboard shows at least one analysis after seed
- `commentary.spec.ts` — commentary endpoint behavior

**Playwright config:** `playwright.config.ts` exists at project root.

These require a running stack (frontend + backend + DB). They are NOT unit tests.

### Linting

**Backend:**
- No `ruff` in `backend/api/requirements.txt`
- No `pyproject.toml` in `backend/` directory
- No `setup.cfg` with `[tool:ruff]` section
- No `.ruff.toml`

**Frontend:**
- No `eslint.config.*` or `.eslintrc*` at `frontend/` root (only node_modules copies)
- No `.prettierrc*` at `frontend/` root
- `@eslint/...` and `eslint` are NOT in `frontend/package.json` devDependencies
- `prettier` is NOT in `frontend/package.json` devDependencies

### CI

**File:** `.github/workflows/ci.yml`
**Single job: `frontend`**
- Steps: checkout, pnpm setup, node 22, `pnpm install --frozen-lockfile`, `pnpm check` (tsc), `pnpm build`
- No backend job
- No lint step (frontend or backend)
- No test step (frontend or backend)
- `cache-dependency-path: frontend/pnpm-lock.yaml` — references `frontend/pnpm-lock.yaml` but lockfile is at project root as `pnpm-lock.yaml`

---

## 2. Gaps (Per Requirement)

### TEST-01: Backend pytest covers health, upload, analysis, worker, commentary

**Gaps:**
- `httpx` missing from `backend/api/requirements.txt` — needed for `httpx.AsyncClient` with FastAPI
- `pytest-asyncio` missing — needed for `async def` test functions
- `anyio[trio]` or `anyio` missing (explicit) — pytest-asyncio requires it
- No `conftest.py` fixtures for: `AsyncClient`, DB session override, test database setup
- Zero HTTP endpoint tests exist; all existing tests are pure unit tests of service/utility classes
- DB dependency override pattern needed: `app.dependency_overrides[get_db] = override_get_db`
- Worker endpoints require worker auth token check — tests must set `X-Worker-Token` header or mock `verify_worker_token`

### TEST-02: Frontend vitest covers key components and hooks

**Gaps:**
- `vitest` not installed
- `@testing-library/react` not installed
- `@testing-library/user-event` not installed
- `@vitest/coverage-v8` not installed
- `jsdom` not installed (or `happy-dom`)
- `vitest.config.ts` does not exist
- No test script in `package.json`
- No existing component test files

**Key targets for vitest (from REQUIREMENTS.md intent):**
- `frontend/src/hooks/useWebSocket.ts` — WebSocket hook
- `frontend/src/lib/api-local.ts` — API client functions
- Analysis sub-components under `frontend/src/pages/analysis/`

### TEST-03: CI includes backend lint + test job

**Gaps:**
- CI has only one job (`frontend`)
- No `backend` job with: Python setup, pip install, ruff check, pytest run
- `cache-dependency-path` in existing CI references `frontend/pnpm-lock.yaml` but lockfile lives at repo root as `pnpm-lock.yaml` — potential existing CI bug to fix

### QUAL-06: ruff configured for backend Python linting

**Gaps:**
- `ruff` not in `backend/api/requirements.txt`
- No ruff config anywhere (`pyproject.toml`, `ruff.toml`, `.ruff.toml`)

### QUAL-07: ESLint + Prettier configured for frontend TypeScript

**Gaps:**
- `eslint` not in `frontend/package.json`
- `prettier` not in `frontend/package.json`
- No eslint config file
- No prettier config file
- No `lint` or `format` script in `frontend/package.json`

---

## 3. Technical Decisions

### Backend Testing: httpx AsyncClient Pattern

FastAPI's recommended test pattern uses `httpx.AsyncClient` with `ASGITransport`. This avoids starting a real server. Combined with `pytest-asyncio` for async test functions.

**DB override pattern** (avoids live Supabase):
```python
# conftest.py
from httpx import AsyncClient, ASGITransport
import pytest_asyncio
from api.main import app
from api.deps import get_db

@pytest_asyncio.fixture
async def client():
    async def override_get_db():
        yield None  # or a real in-memory SQLite session
    app.dependency_overrides[get_db] = override_get_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()
```

For endpoints that only need minimal DB interaction (health, auth with LOCAL_DEV_MODE), the override returning `None` is sufficient. For CRUD endpoints, a SQLite in-memory async session is needed.

**pytest-asyncio mode:** Use `asyncio_mode = "auto"` in `pyproject.toml` or pytest.ini to avoid per-test `@pytest.mark.asyncio` decoration.

**Dependencies to add to `backend/api/requirements.txt`:**
```
httpx>=0.27.0
pytest-asyncio>=0.23.0
ruff>=0.4.0
```

**pytest config** — add `pyproject.toml` in `backend/` (or `pytest.ini`):
```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["api/tests"]
```

### Backend Linting: ruff

Ruff replaces flake8 + black + isort in a single tool. Config in `backend/pyproject.toml`:

```toml
[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP"]
ignore = ["E501"]
```

Run: `ruff check backend/api/` and `ruff format --check backend/api/`

### Frontend Testing: vitest

vitest is the standard choice for Vite projects. It reuses the Vite config and runs in-process.

**Dependencies to add to `frontend/package.json` devDependencies:**
```json
"vitest": "^2.0.0",
"@vitest/coverage-v8": "^2.0.0",
"@testing-library/react": "^16.0.0",
"@testing-library/user-event": "^14.0.0",
"jsdom": "^25.0.0"
```

**`vitest.config.ts`** at `frontend/`:
```typescript
import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: './src/test/setup.ts',
  },
})
```

**Test script in `package.json`:**
```json
"test": "vitest run",
"test:watch": "vitest"
```

### Frontend Linting: ESLint + Prettier

ESLint v9 uses flat config (`eslint.config.js`). Dependencies:
```json
"eslint": "^9.0.0",
"@eslint/js": "^9.0.0",
"typescript-eslint": "^8.0.0",
"eslint-plugin-react-hooks": "^5.0.0",
"eslint-plugin-react-refresh": "^0.4.0",
"prettier": "^3.0.0"
```

Scripts:
```json
"lint": "eslint src/",
"format": "prettier --write src/",
"format:check": "prettier --check src/"
```

### CI: Backend Job

Add a `backend` job parallel to `frontend`:

```yaml
backend:
  name: Backend checks
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with:
        python-version: "3.11"
        cache: pip
        cache-dependency-path: backend/api/requirements.txt
    - name: Install dependencies
      run: pip install -r requirements.txt
      working-directory: backend/api
    - name: Lint (ruff)
      run: ruff check api/ && ruff format --check api/
      working-directory: backend
    - name: Test (pytest)
      run: pytest
      working-directory: backend
      env:
        LOCAL_DEV_MODE: "true"
        JWT_SECRET: "ci-test-secret"
        DATABASE_URL: "sqlite+aiosqlite:///:memory:"
```

**Note:** The existing CI `cache-dependency-path: frontend/pnpm-lock.yaml` is incorrect — `pnpm-lock.yaml` lives at repo root. The frontend job needs fixing alongside adding the backend job.

---

## 4. Risk Factors

### Risk 1: DB Dependency in Endpoint Tests (HIGH impact)
All analysis, video, worker, and commentary endpoints use `AsyncSession` from Supabase. Tests must override `get_db` to avoid needing a live DB. Two approaches: (a) pure mock returning `MagicMock()` for simple endpoints, (b) SQLite in-memory via `aiosqlite` for CRUD tests. `aiosqlite` is not in current requirements.

**Mitigation:** Use `app.dependency_overrides` pattern. For TEST-01 scope (health, upload, analysis, worker, commentary), focus on smoke-level HTTP tests that verify status codes and response shapes rather than full DB integration.

### Risk 2: Worker Auth Token
Worker endpoints check `X-Worker-Token` header against `settings.WORKER_SECRET`. CI tests must set `LOCAL_DEV_MODE=true` (which bypasses auth in AutoLoginMiddleware) OR set `WORKER_SECRET` env var and pass it in test headers.

### Risk 3: Upload Endpoint Requires File + Multipart
`POST /api/upload/video` is a multipart endpoint. httpx AsyncClient supports multipart via `files=` parameter — this is standard but requires a real or mock file object in tests.

### Risk 4: ESLint v9 Flat Config
ESLint v9 uses `eslint.config.js` (flat config), which is different from v8's `.eslintrc`. The Vite project has no prior ESLint setup, so starting with v9 flat config is clean. However, some Radix/React ecosystem plugins may not fully support v9 yet — stick to `typescript-eslint` + `eslint-plugin-react-hooks` which both support v9.

### Risk 5: next-themes Still Present
`frontend/package.json` still lists `next-themes: ^0.4.6` as a dependency — QUAL-03 claims it was removed but it appears in the lockfile snapshot. Lint may surface unused imports from this. Verify before running ESLint for the first time.

### Risk 6: Existing Tests Use asyncio.run() not pytest-asyncio
`test_tactical.py` uses `asyncio.run()` directly. After adding `pytest-asyncio` with `asyncio_mode = "auto"`, these tests will still pass (asyncio.run inside sync test functions is valid). No changes needed to existing tests.

---

## 5. Implementation Notes

### Test File Placement
- Backend HTTP endpoint tests: `backend/api/tests/test_endpoints.py` (new file)
- Pipeline tests: `backend/pipeline/test_analytics.py` (already exists — no move needed)
- Frontend component tests: `frontend/src/test/` directory (new)

### Minimal Endpoint Test Pattern
```python
# backend/api/tests/test_endpoints.py
import pytest
from httpx import AsyncClient, ASGITransport

@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
```

### Worker Token Pattern
```python
# In conftest or test helper
WORKER_TOKEN = "test-worker-token"

@pytest_asyncio.fixture
async def worker_client(monkeypatch):
    monkeypatch.setenv("WORKER_SECRET", WORKER_TOKEN)
    # ... same AsyncClient setup
```

### Vitest Component Test Pattern
```typescript
// frontend/src/test/useWebSocket.test.ts
import { renderHook } from '@testing-library/react'
import { useWebSocket } from '../hooks/useWebSocket'
// mock WebSocket global
```

### ruff First-Run Strategy
Run `ruff check --fix backend/api/` for auto-fixable violations (import sorting, unused imports). Review remaining violations manually. Start with `select = ["E", "F", "I"]` (errors, pyflakes, imports) and expand after first clean pass.

### CI pnpm lockfile path fix
The existing CI step has `cache-dependency-path: frontend/pnpm-lock.yaml`. The actual lockfile is at `/pnpm-lock.yaml` (repo root). This should be corrected to `pnpm-lock.yaml` in the updated CI.

### pytest.ini_options location
Place in `backend/pyproject.toml` (new file) rather than a standalone `pytest.ini`, since ruff config also goes there. This keeps backend config consolidated.

```toml
# backend/pyproject.toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["api/tests", "pipeline"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP"]
```

---

## Sources

### Primary (HIGH confidence)
- Codebase inspection via Read/Bash — all findings above are from direct file reads
- `backend/api/requirements.txt` — confirmed missing httpx, pytest-asyncio, ruff
- `frontend/package.json` — confirmed missing vitest, eslint, prettier
- `.github/workflows/ci.yml` — confirmed single frontend job
- `backend/api/tests/` — confirmed existing test scope and conftest state
- `backend/api/routers/` + `main.py` — confirmed full endpoint inventory

### Secondary (MEDIUM confidence)
- FastAPI official docs pattern: `httpx.AsyncClient` + `ASGITransport` for testing
- pytest-asyncio docs: `asyncio_mode = "auto"` eliminates per-test markers
- ruff docs: `pyproject.toml` `[tool.ruff]` config section
- ESLint v9 flat config: `eslint.config.js` format

---

## Metadata

**Confidence breakdown:**
- Current state (what exists): HIGH — direct file reads
- Gap analysis: HIGH — derived from direct file reads
- Technical decisions (tool choices): MEDIUM — based on well-established patterns; specific versions should be verified against npm/pypi at implementation time
- Risk factors: MEDIUM — based on code inspection + known FastAPI/pytest patterns

**Research date:** 2026-03-28
**Valid until:** 2026-05-28 (stable tooling — ruff, vitest, ESLint change slowly)
