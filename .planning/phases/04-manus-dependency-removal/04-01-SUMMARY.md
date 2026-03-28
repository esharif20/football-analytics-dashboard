---
phase: 04-manus-dependency-removal
plan: 01
subsystem: api, pipeline
tags: [env-vars, cors, fastapi, worker, security]

# Dependency graph
requires: []
provides:
  - Env-var-driven model URL configuration with fail-fast validation
  - Configurable CORS origins via CORS_ORIGINS env var
affects: [05-supabase-migration, pipeline-deployment]

# Tech tracking
tech-stack:
  added: []
  patterns: [env-var-driven config with fail-fast validation, comma-separated env var parsing]

key-files:
  created: []
  modified:
    - backend/pipeline/worker.py
    - backend/api/config.py
    - backend/api/main.py

key-decisions:
  - "Default CORS_ORIGINS includes localhost:5173 and localhost:3000 for dev parity"
  - "Worker exits with sys.exit(1) on missing model URLs rather than logging and continuing"

patterns-established:
  - "Fail-fast env var validation: validate required env vars at startup, not at first use"
  - "Comma-separated env var lists with property accessor for parsed values"

requirements-completed: [MANUS-01, DB-05]

# Metrics
duration: 3min
completed: 2026-03-28
---

# Phase 04 Plan 01: Manus Dependency Removal - Worker and CORS Summary

**Env-var-driven model URLs replacing manuscdn.com CDN hardcodes, plus configurable CORS origins via Settings**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-28T11:22:19Z
- **Completed:** 2026-03-28T11:25:00Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Removed all manuscdn.com CDN URLs from pipeline worker, replaced with MODEL_URL_PLAYER/BALL/PITCH env vars
- Added validate_model_urls() fail-fast startup check that exits with clear error if any model URL is missing
- Made CORS origins configurable via CORS_ORIGINS env var with sensible localhost defaults

## Task Commits

Each task was committed atomically:

1. **Task 1: Replace hardcoded model URLs with env vars and add startup validation** - `1753540` (feat)
2. **Task 2: Make CORS origins configurable via env var** - `6236cd1` (feat)

## Files Created/Modified
- `backend/pipeline/worker.py` - Replaced hardcoded manuscdn.com model URLs with env-var-driven config; added validate_model_urls() fail-fast check
- `backend/api/config.py` - Added CORS_ORIGINS setting and cors_origins_list property to Settings class
- `backend/api/main.py` - Replaced hardcoded CORS origins list with settings.cors_origins_list

## Decisions Made
- Default CORS_ORIGINS value is "http://localhost:5173,http://localhost:3000" to match existing dev setup (dropped localhost:3001 which was unused)
- validate_model_urls() calls sys.exit(1) for fail-fast behavior rather than raising an exception, since it runs at module startup before any request handling

## Deviations from Plan

None - plan executed exactly as written.

## Known Stubs

None - all values are wired to environment variables with no placeholder data.

## Issues Encountered

None.

## User Setup Required

Workers now require three environment variables to be set before starting:
- `MODEL_URL_PLAYER` - Download URL for player_detection.pt model weights
- `MODEL_URL_BALL` - Download URL for ball_detection.pt model weights
- `MODEL_URL_PITCH` - Download URL for pitch_detection.pt model weights

Optionally, `CORS_ORIGINS` can be set as a comma-separated list of allowed origins (defaults to `http://localhost:5173,http://localhost:3000`).

## Next Phase Readiness
- Backend no longer depends on manuscdn.com for model downloads
- CORS is now environment-configurable for any deployment target
- Ready for remaining Manus dependency removal (frontend hero images, localStorage keys) in plan 04-02

---
*Phase: 04-manus-dependency-removal*
*Completed: 2026-03-28*
