# Technology Stack: Migration & Hardening Additions

**Project:** Football Analytics Dashboard v0.2
**Researched:** 2026-03-28
**Scope:** New dependencies only -- existing validated stack (React 19, Vite 6, FastAPI, SQLAlchemy async, etc.) not re-researched.

## Recommended Stack Additions

### 1. Database Migration: MySQL/aiomysql -> Supabase PostgreSQL/asyncpg

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| asyncpg | >=0.31.0 | Async PostgreSQL driver | Only production-grade async PG driver for Python. Direct replacement for aiomysql in SQLAlchemy's `create_async_engine`. HIGH confidence. |
| psycopg2-binary | >=2.9.9 | Sync PG driver (Alembic only) | Alembic migrations run synchronously by default. Needed for `alembic upgrade head` commands. Avoids complex async env.py hacks. |

**Connection string change:**
```
# OLD (MySQL)
mysql+aiomysql://user:pass@localhost:3307/football

# NEW (Supabase direct connection -- port 5432)
postgresql+asyncpg://postgres.[project-ref]:[password]@aws-0-[region].pooler.supabase.com:5432/postgres
```

**CRITICAL Supabase gotcha:** Use the **direct connection** string (port 5432), NOT the pooler (port 6543). Supabase's Supavisor pooler uses transaction-mode PgBouncer which breaks asyncpg's prepared statement cache. If pooler is unavoidable, pass `statement_cache_size=0` to `connect_args` in `create_async_engine`. For this project's scale, direct connection is correct.

**What to remove:**
- `aiomysql>=0.2.0` from requirements.txt
- All `mysql://` / `mysql2://` URL rewriting logic in database.py

**Model compatibility:** Current models use generic SQLAlchemy types (`Integer`, `String`, `Text`, `Float`, `JSON`, `TIMESTAMP`, `Enum`). No MySQL-specific types found. Migration is a clean driver swap -- no model rewrites needed. `func.now()` works identically on PostgreSQL.

Source: [Supabase Connection Docs](https://supabase.com/docs/guides/database/connecting-to-postgres) | [asyncpg + Supabase pooler issue](https://medium.com/@patrickduch93/supabase-pooling-and-asyncpg-dont-mix-here-s-the-real-fix-44f700b05249)

### 2. Schema Migrations: Alembic

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| alembic | >=1.18.4 | Database schema migrations | The standard migration tool for SQLAlchemy. No real alternative. Supports autogenerate from models. HIGH confidence. |

**Setup approach:** Use synchronous Alembic with `psycopg2-binary` driver rather than async env.py. Reason: Alembic's async support requires non-trivial env.py modifications (`run_async` wrapper, `asyncio.run()` in `run_migrations_online`). For a project that runs migrations as a CLI step (not at runtime), sync is simpler and equally effective. The async app engine uses asyncpg; Alembic uses its own sync psycopg2 connection.

**Key config:**
- `alembic.ini`: `sqlalchemy.url` uses `postgresql://` (sync, psycopg2)
- `env.py`: import `Base.metadata` from `backend.api.database` for autogenerate
- Set naming conventions on `Base.metadata` for deterministic constraint names

**Initial migration:** `alembic revision --autogenerate -m "initial_schema"` to capture current models, then `alembic stamp head` on Supabase after manual schema creation (or `alembic upgrade head` on empty DB).

Source: [Alembic Docs](https://alembic.sqlalchemy.org/en/latest/tutorial.html) | [Async Alembic guide](https://dev.to/matib/alembic-with-async-sqlalchemy-1ga)

### 3. Backend Testing: pytest + httpx

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| pytest | >=8.0.0 | Test runner | Already in requirements.txt. Standard Python test framework. |
| pytest-asyncio | >=0.24.0 | Async test support | Required for `async def` test functions with FastAPI's async endpoints. Use `mode="auto"` in pytest.ini to avoid decorating every test. |
| httpx | >=0.28.0 | Async HTTP test client | FastAPI's recommended async test client. `AsyncClient(transport=ASGITransport(app=app))` replaces sync `TestClient` for async route testing. |
| pytest-cov | >=6.0.0 | Coverage reporting | Coverage metrics for CI. `--cov=backend/api --cov-report=term-missing`. |

**Testing pattern:**
```python
import pytest
from httpx import ASGITransport, AsyncClient
from backend.api.main import app

@pytest.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as ac:
        yield ac

async def test_health(client):
    resp = await client.get("/api/health")
    assert resp.status_code == 200
```

**Database in tests:** Use `dependency_overrides` to swap the real DB session with a test session pointing at a test database (or SQLite async for unit tests). Do NOT mock SQLAlchemy -- test against a real DB engine.

Source: [FastAPI Async Tests](https://fastapi.tiangolo.com/advanced/async-tests/) | [pytest-asyncio docs](https://pytest-asyncio.readthedocs.io/)

### 4. Frontend Testing: vitest + Testing Library

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| vitest | >=3.0.0 | Test runner | Shares Vite config, no separate bundler setup. 4x+ faster than Jest. Native ESM, TypeScript support. |
| @testing-library/react | >=16.0.0 | React component testing | Standard React testing library. Works with React 19. |
| @testing-library/jest-dom | >=6.6.0 | DOM assertion matchers | `toBeInTheDocument()`, `toHaveTextContent()`, etc. |
| @testing-library/user-event | >=14.5.0 | User interaction simulation | Realistic event firing (click, type, etc.) |
| jsdom | >=25.0.0 | Browser environment | DOM implementation for Node.js test environment. |

**Vitest config** (in `vite.config.ts` or `vitest.config.ts`):
```typescript
/// <reference types="vitest" />
import { defineConfig } from 'vite'

export default defineConfig({
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: './src/test/setup.ts',
    include: ['src/**/*.{test,spec}.{ts,tsx}'],
  },
})
```

**Setup file** (`src/test/setup.ts`):
```typescript
import '@testing-library/jest-dom/vitest'
import { cleanup } from '@testing-library/react'
import { afterEach } from 'vitest'

afterEach(() => cleanup())
```

**What NOT to add:** `vitest-browser-react` (browser mode) -- overkill for unit tests, Playwright already covers E2E. Stick with jsdom.

Source: [Vitest Component Testing](https://vitest.dev/guide/browser/component-testing) | [RTL + Vitest guide (2026)](https://oneuptime.com/blog/post/2026-01-15-unit-test-react-vitest-testing-library/view)

### 5. Backend Linting: ruff

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| ruff | >=0.15.0 | Python linter + formatter | Replaces flake8 + black + isort in one tool. 100x faster (Rust). Single config in pyproject.toml. Project already decided on ruff. |

**Config** (`pyproject.toml`):
```toml
[tool.ruff]
target-version = "py311"
line-length = 88

[tool.ruff.lint]
select = [
    "E",   # pycodestyle errors
    "F",   # pyflakes
    "I",   # isort
    "N",   # pep8-naming
    "UP",  # pyupgrade
    "B",   # bugbear
    "S",   # bandit (security)
    "A",   # builtins shadowing
]
ignore = [
    "S101",  # assert in tests is fine
]

[tool.ruff.lint.per-file-ignores]
"tests/**/*.py" = ["S101"]

[tool.ruff.format]
quote-style = "double"
```

**Commands:** `ruff check .` (lint), `ruff format .` (format), `ruff check --fix .` (autofix).

Source: [Ruff Configuration](https://docs.astral.sh/ruff/configuration/) | [Ruff Settings](https://docs.astral.sh/ruff/settings/)

### 6. Frontend Linting: ESLint 9 + Prettier

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| eslint | >=9.0.0 | JavaScript/TypeScript linter | ESLint 9 with flat config is the current standard. |
| @eslint/js | >=9.0.0 | ESLint core rules | Required for flat config `recommended` rules. |
| typescript-eslint | >=8.0.0 | TypeScript ESLint integration | Unified package replacing deprecated @typescript-eslint/eslint-plugin + parser. |
| eslint-plugin-react-hooks | >=5.0.0 | React hooks rules | Enforces Rules of Hooks. Essential for React projects. |
| eslint-plugin-react-refresh | >=0.4.0 | Vite HMR compatibility | Validates components are HMR-safe. |
| prettier | >=3.4.0 | Code formatter | Consistent formatting. |
| eslint-config-prettier | >=10.0.0 | Disable conflicting ESLint rules | Prevents ESLint formatting rules from conflicting with Prettier. |

**Flat config** (`eslint.config.js`):
```javascript
import js from '@eslint/js'
import tseslint from 'typescript-eslint'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import prettier from 'eslint-config-prettier'

export default tseslint.config(
  { ignores: ['dist', 'node_modules'] },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    plugins: { 'react-hooks': reactHooks, 'react-refresh': reactRefresh },
    rules: {
      ...reactHooks.configs.recommended.rules,
      'react-refresh/only-export-components': 'warn',
    },
  },
  prettier,
)
```

**What NOT to add:** `eslint-plugin-prettier` (runs Prettier as an ESLint rule) -- slower, noisier. Use `eslint-config-prettier` to disable conflicts, run Prettier separately.

Source: [ESLint flat config extends](https://eslint.org/blog/2025/03/flat-config-extends-define-config-global-ignores/) | [typescript-eslint](https://typescript-eslint.io/)

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| PG async driver | asyncpg | psycopg3 (async) | asyncpg is faster, more mature for async, better SQLAlchemy integration |
| Migrations | Alembic (sync) | Alembic async env.py | Unnecessary complexity -- migrations are CLI commands, not runtime |
| Backend test client | httpx AsyncClient | FastAPI TestClient | TestClient is sync-only, blocks event loop with async routes |
| Frontend test runner | vitest | jest | Vitest shares Vite config, faster, native ESM |
| Python linter | ruff | flake8 + black + isort | ruff does all three, 100x faster |
| JS linter config | ESLint 9 flat config | Legacy .eslintrc | Flat config is ESLint 9 default, .eslintrc is deprecated |

## Installation

### Backend (add to requirements.txt)
```
# Database (replace aiomysql)
asyncpg>=0.31.0
psycopg2-binary>=2.9.9

# Migrations
alembic>=1.18.4

# Testing
pytest>=8.0.0
pytest-asyncio>=0.24.0
httpx>=0.28.0
pytest-cov>=6.0.0

# Linting (dev tool, install separately or add to requirements-dev.txt)
ruff>=0.15.0
```

### Frontend (pnpm add -D)
```bash
# Testing
pnpm add -D vitest @testing-library/react @testing-library/jest-dom @testing-library/user-event jsdom

# Linting
pnpm add -D eslint @eslint/js typescript-eslint eslint-plugin-react-hooks eslint-plugin-react-refresh prettier eslint-config-prettier
```

### Remove
```
# From requirements.txt
aiomysql>=0.2.0  # DELETE

# From package.json (already identified as dead)
next-themes  # DELETE (Next.js lib, not used in Vite)
```

## Integration Notes

### database.py Changes
The entire MySQL URL rewriting block is replaced with a simple PostgreSQL URL passthrough:
```python
_url = settings.DATABASE_URL
# Supabase provides postgresql:// URLs; SQLAlchemy needs postgresql+asyncpg://
if _url.startswith("postgresql://"):
    _url = _url.replace("postgresql://", "postgresql+asyncpg://", 1)

engine = create_async_engine(
    _url,
    echo=False,
    pool_pre_ping=True,
    # If using Supabase pooler (port 6543), uncomment:
    # connect_args={"statement_cache_size": 0},
)
```

### Environment Variable Changes
```
# OLD
DATABASE_URL=mysql://user:pass@localhost:3307/football_analytics

# NEW
DATABASE_URL=postgresql://postgres.[ref]:[pass]@aws-0-[region].pooler.supabase.com:5432/postgres
```

### CI Script Additions
```yaml
# Backend
- run: ruff check backend/
- run: ruff format --check backend/
- run: pytest backend/api/tests/ --cov=backend/api -q

# Frontend
- run: pnpm run lint
- run: pnpm run test:ci
```

### package.json Script Additions
```json
{
  "scripts": {
    "lint": "eslint src/",
    "lint:fix": "eslint src/ --fix",
    "format": "prettier --write src/",
    "format:check": "prettier --check src/",
    "test": "vitest",
    "test:ci": "vitest run --reporter=verbose"
  }
}
```

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| asyncpg driver swap | HIGH | SQLAlchemy docs confirm drop-in replacement, models use generic types |
| Supabase connection | HIGH | Official docs + confirmed asyncpg gotcha with pooler |
| Alembic setup | HIGH | Standard tooling, well-documented |
| pytest + httpx | HIGH | FastAPI official recommendation |
| vitest + RTL | HIGH | Standard Vite ecosystem tooling |
| ruff | HIGH | Well-established, simple config |
| ESLint 9 flat config | MEDIUM | Flat config is stable but plugin ecosystem still catching up |

## Sources

- [SQLAlchemy Async Docs](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html)
- [asyncpg PyPI](https://pypi.org/project/asyncpg/) -- v0.31.0, Nov 2025
- [Alembic PyPI](https://pypi.org/project/alembic/) -- v1.18.4, Feb 2026
- [Supabase Connection Docs](https://supabase.com/docs/guides/database/connecting-to-postgres)
- [Supabase Pooler + asyncpg Fix](https://medium.com/@patrickduch93/supabase-pooling-and-asyncpg-dont-mix-here-s-the-real-fix-44f700b05249)
- [FastAPI Async Tests](https://fastapi.tiangolo.com/advanced/async-tests/)
- [Ruff Configuration](https://docs.astral.sh/ruff/configuration/)
- [ESLint Flat Config](https://eslint.org/blog/2025/03/flat-config-extends-define-config-global-ignores/)
