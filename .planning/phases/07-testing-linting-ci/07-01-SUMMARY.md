---
phase: "07-testing-linting-ci"
plan: "07-01"
subsystem: "backend"
tags: [ruff, pytest, linting, testing, ci]
dependency_graph:
  requires: []
  provides: [ruff-configured, pytest-smoke-tests]
  affects: [backend/api]
tech_stack:
  added: [ruff>=0.4.0, pytest-asyncio>=0.23.0, httpx>=0.27.0]
  patterns: [AsyncClient-ASGI-transport, dependency_overrides, AsyncMock-DB-fixture]
key_files:
  created:
    - backend/pyproject.toml
    - backend/api/tests/test_endpoints.py
  modified:
    - backend/api/requirements.txt
    - backend/api/tests/conftest.py
    - backend/api/config.py
    - backend/api/models.py
decisions:
  - Use AsyncMock for DB fixture instead of MagicMock to support await expressions
  - Convert str+enum.Enum enums to enum.StrEnum (UP042) in models.py
  - Move logging import to top of config.py to fix E402
metrics:
  duration: "330s"
  completed: "2026-03-28"
  tasks_completed: 2
  files_created: 2
  files_modified: 4
requirements_satisfied: [QUAL-06, TEST-01]
---

# Phase 07 Plan 01: Ruff Linting + Pytest Smoke Tests Summary

**One-liner:** Ruff linting configured with E/F/I/UP rules on backend/api/ (zero violations), plus 6 HTTP endpoint smoke tests via AsyncClient with AsyncMock DB override — all 39 tests green.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Add ruff, pytest-asyncio, httpx + create pyproject.toml | 3c66573 | requirements.txt, pyproject.toml, config.py, models.py |
| 2 | Add AsyncClient fixture and write endpoint smoke tests | dbc353f | conftest.py, test_endpoints.py |

## Verification Results

- `ruff check api/` — exits 0, zero violations
- `ruff format --check api/` — 26 files already formatted
- `pytest api/tests/ -v` — 39 passed (33 test_tactical.py + 6 test_endpoints.py)
- `backend/pipeline/` — untouched (git diff clean)
- `backend/pyproject.toml` — contains [tool.ruff] with select = ["E", "F", "I", "UP"]

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] MagicMock not awaitable — replaced with AsyncMock for DB methods**
- **Found during:** Task 2 — first test run
- **Issue:** `MagicMock()` for the DB override caused `TypeError: object MagicMock can't be used in 'await' expression` when routes called `await db.execute(...)`, `await db.commit()`, etc.
- **Fix:** Changed `override_get_db` to yield a `MagicMock` with `execute`, `commit`, `flush`, `refresh` replaced by `AsyncMock` instances.
- **Files modified:** backend/api/tests/conftest.py
- **Commit:** dbc353f

**2. [Rule 1 - Bug] E402 import at module level after settings instantiation in config.py**
- **Found during:** Task 1 — ruff check run
- **Issue:** `import logging as _logging` appeared after `settings = Settings()`, violating E402.
- **Fix:** Moved the logging import to the top of the file with other imports; removed duplicate.
- **Files modified:** backend/api/config.py
- **Commit:** 3c66573

**3. [Rule 1 - Bug] UP042 str+enum.Enum inheritance in models.py**
- **Found during:** Task 1 — ruff check run
- **Issue:** `UserRole`, `PipelineMode`, `ProcessingStatus` all inherited from `(str, enum.Enum)` which ruff UP042 flags as deprecated pattern.
- **Fix:** Changed all three to `enum.StrEnum` (Python 3.11+ built-in).
- **Files modified:** backend/api/models.py
- **Commit:** 3c66573

## Known Stubs

None — this plan adds tooling and tests only, no UI-rendering data paths.

## Self-Check: PASSED
