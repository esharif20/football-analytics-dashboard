# Domain Pitfalls: Migration & Hardening

**Domain:** MySQL-to-PostgreSQL migration, Alembic adoption, CDN removal, React component decomposition, async test fixtures -- all applied to an existing football analytics dashboard
**Researched:** 2026-03-28
**Confidence:** HIGH (all pitfalls verified against actual codebase patterns in models.py, database.py, worker.py, Analysis.tsx)

---

## Critical Pitfalls

Mistakes that cause data loss, broken deployments, or major rework.

### Pitfall 1: Supabase Pooler + asyncpg Prepared Statements

**What goes wrong:** Using Supabase's pooler connection (port 6543) with asyncpg causes `prepared statement does not exist` errors under load. asyncpg caches prepared statements by default, but Supabase Supavisor uses transaction-mode PgBouncer which does not support prepared statements.

**Why it happens:** Developers copy the pooler connection string from the Supabase dashboard (it's shown first) without realizing it goes through PgBouncer.

**Consequences:** Application crashes on random queries. Often works initially, then fails under concurrent load or after connection recycling.

**Prevention:** Use direct connection string (port 5432), OR set `statement_cache_size=0`:
```python
engine = create_async_engine(
    _url, echo=False, pool_pre_ping=True,
    connect_args={"statement_cache_size": 0}  # Required for PgBouncer
)
```

**Detection:** Errors containing "prepared statement" in logs. Test with concurrent requests, not just single queries.

---

### Pitfall 2: `server_default="0"` for Boolean Is MySQL-Specific

**What goes wrong:** `models.py:82` has `skipCache: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="0")`. MySQL accepts `"0"` as boolean false. PostgreSQL does NOT -- produces: `ProgrammingError: column "skipcache" is of type boolean but default expression is of type integer`.

**Why it happens:** MySQL treats 0/1 as boolean aliases. PostgreSQL is strict.

**Consequences:** Any INSERT that omits `skipCache` (analysis creation endpoint) fails. This is a critical path.

**Prevention:** Change to `server_default=text("false")`. Also fix `server_default="0"` on Integer `progress` field (line 77) to `server_default=text('0')`.

**Detection:** `grep -n 'server_default="0"' models.py` should return zero matches after fix.

---

### Pitfall 3: PostgreSQL Enum Types Are Independent Schema Objects

**What goes wrong:** MySQL stores ENUM inline. PostgreSQL creates separate `CREATE TYPE` objects. Running `create_all()` twice or Alembic on existing types produces `DuplicateObject: type "processingstatus" already exists`. Dropping a table does NOT drop its enum type.

**Why it happens:** PostgreSQL enum types have their own lifecycle, separate from tables.

**Consequences:** Migration failures, orphaned types, Alembic autogenerate not detecting enum value changes.

**Prevention:**
1. Explicitly name types: `Enum(ProcessingStatus, name="processingstatus")`
2. For future enum additions: manually write `ALTER TYPE ... ADD VALUE` (autogenerate misses this)
3. Alternative: `native_enum=False` (VARCHAR+CHECK constraint) avoids the problem entirely

**Detection:** `\dT` in psql shows enum types. Orphaned types visible after table drops.

---

### Pitfall 4: Connection String Format Change Breaks Silently

**What goes wrong:** `database.py:8-9` only handles `mysql://` and `mysql2://`. Supabase provides `postgresql://` URLs. Without updating the conversion logic, SQLAlchemy tries the sync psycopg2 driver in an async context, failing with `NoSuchModuleError`.

**Why it happens:** Nobody added PostgreSQL URL handling to replace the MySQL handling.

**Consequences:** App fails to start. Error may be confusing (`NoSuchModuleError` for psycopg2, not an obvious "wrong driver" message).

**Prevention:** Replace URL rewriting to handle `postgresql://` and `postgres://` -> `postgresql+asyncpg://`.

**Detection:** App startup failure. Test with actual Supabase URL.

---

### Pitfall 5: Alembic Autogenerate on MySQL-Incompatible Models

**What goes wrong:** Running `alembic revision --autogenerate` BEFORE fixing `server_default="0"` and `TIMESTAMP` types produces a migration with MySQL artifacts that fails on PostgreSQL.

**Why it happens:** Alembic reads current Base.metadata. If models still have MySQL assumptions, the migration bakes them in.

**Consequences:** Must delete and regenerate migration, possibly losing migration chain.

**Prevention:** Fix ALL model compatibility issues BEFORE running autogenerate. Build order: models.py fixes -> database.py swap -> alembic init -> autogenerate.

**Detection:** Review generated migration for raw `"0"` defaults, MySQL-specific types.

---

### Pitfall 6: Runtime ALTER TABLE Conflicts with Alembic

**What goes wrong:** `worker.py:39` and `analyses.py:99` use `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`. Once Alembic manages schema, runtime DDL creates drift Alembic doesn't know about.

**Why it happens:** `_ensure_analysis_columns()` was a workaround for not having migrations. The columns (`config`, `claimedBy`) are already in models.py, so Alembic will include them.

**Prevention:** Remove `_ensure_analysis_columns()` entirely after Alembic initial migration includes these columns.

**Detection:** `grep -r "_ensure_analysis_columns" backend/` should return zero matches.

---

### Pitfall 7: Auth Middleware Module-Level Import Breaks Tests

**What goes wrong:** `auth.py` imports `async_session` from `database.py` at module level. In tests, `dependency_overrides[get_db]` injects a test session for routes, but the middleware still uses the module-level `async_session`. If it points to a real but wrong database, tests fail with connection errors in the middleware, not the route.

**Why it happens:** FastAPI `dependency_overrides` only affects `Depends()` in route handlers, not middleware.

**Consequences:** Tests fail with `ConnectionRefusedError` in auth middleware even though `get_db` is properly overridden.

**Prevention:** Ensure `DATABASE_URL` is empty in test environment. This makes `engine = None` and `async_session = None` (database.py lines 13-15). Middleware sees `async_session is None` (auth.py line 48) and falls back to `FallbackUser` (auth.py line 78). Routes use overridden `get_db`. Both paths work independently.

**Detection:** Tests fail with database connection errors from middleware stack trace, not route handler.

---

### Pitfall 8: Alembic Stamp vs Upgrade on Existing Database

**What goes wrong:** Running `alembic upgrade head` on a database that already has tables fails with `relation "users" already exists`.

**Why it happens:** Someone ran `create_all()` or manually created tables before Alembic adoption.

**Prevention:**
- Fresh Supabase: `alembic upgrade head` works
- Existing tables: `alembic stamp head` marks revision as applied without DDL
- Remove any `Base.metadata.create_all()` from app startup

**Detection:** `ProgrammingError: relation "X" already exists`.

---

## Moderate Pitfalls

### Pitfall 9: Analysis.tsx Import Path After Decomposition

**What goes wrong:** Moving `Analysis.tsx` to `Analysis/index.tsx` causes 404 if the import path includes `.tsx` extension.

**Why it happens:** Vite resolves `./pages/Analysis` to `./pages/Analysis/index.tsx`, but `./pages/Analysis.tsx` (with extension) fails after the file moves.

**Prevention:** Verify the router import uses extensionless path. Check: `grep -r "Analysis.tsx" frontend/src/`.

### Pitfall 10: Analysis.tsx Shared Closures Become Prop Drilling

**What goes wrong:** 25+ inline components share closure over `analysis`, `stats`, `filterTeam`, `selectedPlayer`, `videoRef`. Naive extraction requires 10+ props per component.

**Why it happens:** Original developer defined inline components specifically to avoid prop drilling.

**Prevention:** Group components by data dependency, not visual layout:
- **Stats group** (depends on `stats` only): PossessionDonut, StatsComparisonBar, etc.
- **Pitch group** (depends on analytics + selectedPlayer): PitchRadar, HeatmapView, etc.
- **Events group** (depends on events + filterTeam): EventTimeline, PlayerStatsTable
- **AI group** (depends on analysisId only): AICommentarySection

Use the existing `TeamColorsCtx` pattern. Tab container components accept 2-3 props max, handle their own internal layout.

### Pitfall 11: Losing Existing test_tactical.py

**What goes wrong:** Rewriting conftest.py removes the sys.path additions that test_tactical.py depends on.

**Prevention:** Preserve the `sys.path.insert()` lines at the top of conftest.py when adding async fixtures.

### Pitfall 12: TIMESTAMP Behavior Difference

**What goes wrong:** MySQL TIMESTAMP auto-handles UTC. PostgreSQL TIMESTAMP (without timezone) stores naive datetimes.

**Prevention:** The codebase already uses `.replace(tzinfo=None)` pattern (worker.py lines 53, 76, 116). Keep this. Consider `DateTime(timezone=True)` in a future phase.

### Pitfall 13: `onupdate=func.now()` Is ORM-Only

**What goes wrong:** Direct SQL updates via Supabase dashboard do not trigger `updatedAt` auto-update.

**Prevention:** Document that `updatedAt` only auto-updates through ORM. All current writes go through the API, so this is not a regression.

### Pitfall 14: `event_metadata` Column Alias Must Be Preserved

**What goes wrong:** `models.py:111` uses `event_metadata = mapped_column("metadata", JSON)`. If someone removes the string argument, Alembic creates a new `event_metadata` column instead of using existing `metadata`.

**Prevention:** Keep the explicit column name string. Add code comment explaining the alias.

### Pitfall 15: Vitest and Playwright File Pattern Collision

**What goes wrong:** vitest picks up `*.spec.ts` Playwright files, or Playwright runs `*.test.tsx` vitest files.

**Prevention:** Configure vitest to `include: ['src/**/*.test.{ts,tsx}']`. Playwright uses `tests/e2e/*.spec.ts`. Separate directories + separate extensions = no collision.

### Pitfall 16: pytest-asyncio Mode Configuration

**What goes wrong:** Tests fail with "coroutine was never awaited" because pytest-asyncio defaults to "strict" mode.

**Prevention:** Set `asyncio_mode = "auto"` in `pyproject.toml` under `[tool.pytest.ini_options]`.

### Pitfall 17: Ruff Formatting Creates Massive Diff

**What goes wrong:** Running `ruff format` on unformatted code produces hundreds of changes mixed with logic changes.

**Prevention:** Apply formatting in a single dedicated commit. Add hash to `.git-blame-ignore-revs`.

---

## Minor Pitfalls

### Pitfall 18: JSONB vs JSON in Initial Migration

If changing `JSON` to `JSONB` in models.py for an existing populated database, Alembic generates ALTER COLUMN migrations requiring full table rewrite. For fresh Supabase (no data), this is fine -- initial migration uses JSONB directly. For data migration, use `JSON` first, then separate migration with `USING` cast.

### Pitfall 19: Alembic Autogenerate Misses Enum Value Changes

Adding values to `ProcessingStatus`, `PipelineMode`, or `UserRole` requires manual migration: `ALTER TYPE ... ADD VALUE`. Cannot run inside transaction on PG < 12.

### Pitfall 20: ESLint 9 Flat Config Plugin Compatibility

Some ESLint plugins lag behind flat config. Use `typescript-eslint` unified package (flat-config native) instead of older `@typescript-eslint/eslint-plugin` + parser pair.

### Pitfall 21: SQLite Cannot Test SKIP LOCKED

`worker/pending` endpoint uses `.with_for_update(skip_locked=True)` which SQLite does not support. Test this endpoint against PostgreSQL or mock the query. Other endpoints work fine on SQLite.

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| DB migration | #1 (pooler), #2 (Boolean), #4 (URL format) | Direct connection, fix defaults, update URL rewriting |
| Alembic setup | #5 (autogenerate order), #6 (runtime DDL), #8 (stamp vs upgrade) | Fix models first, remove ALTER TABLE, check for existing tables |
| Backend testing | #7 (auth middleware), #11 (existing tests), #16 (asyncio mode) | Empty DATABASE_URL, preserve sys.path, auto mode |
| Frontend decomposition | #9 (import path), #10 (prop drilling) | Extensionless imports, group by data dependency |
| Frontend testing | #15 (file patterns), #21 (SQLite limits) | Separate extensions, PG for SKIP LOCKED tests |

## "Looks Done But Isn't" Checklist

- [ ] `grep -n 'server_default="0"' models.py` returns zero matches
- [ ] `grep -r "_ensure_analysis_columns" backend/` returns zero matches
- [ ] `grep -r "mysql" backend/api/database.py` returns zero matches
- [ ] `grep -r "aiomysql" backend/` returns zero matches
- [ ] `alembic current` shows head revision
- [ ] `SELECT 1` succeeds via async engine against Supabase
- [ ] Full upload-process-view flow works on PostgreSQL
- [ ] Test suite passes twice consecutively (no leaked state)
- [ ] `grep -r "Analysis.tsx" frontend/src/` returns zero matches (after decomposition)
- [ ] Vitest and Playwright both pass in CI without interference

## Sources

- [Supabase Pooler + asyncpg](https://medium.com/@patrickduch93/supabase-pooling-and-asyncpg-dont-mix-here-s-the-real-fix-44f700b05249)
- [Supabase disabling prepared statements](https://supabase.com/docs/guides/troubleshooting/disabling-prepared-statements-qL8lEL)
- [SQLAlchemy PostgreSQL Dialect](https://docs.sqlalchemy.org/en/20/dialects/postgresql.html)
- [Alembic autogenerate limitations](https://alembic.sqlalchemy.org/en/latest/autogenerate.html)
- [FastAPI Async Tests](https://fastapi.tiangolo.com/advanced/async-tests/)
- [PostgreSQL Enum Type Documentation](https://www.postgresql.org/docs/current/datatype-enum.html)
- [asyncpg FAQ - PgBouncer](https://magicstack.github.io/asyncpg/current/faq.html)
