---
phase: 03-pipeline-integration-hardening
plan: 01
subsystem: [pipeline, api, frontend]
tags: [fastapi, websocket, react, pipeline]

# Dependency graph
requires:
  - phase: 02-stabilize-dev-env
    provides: "dev env commands reviewed"
provides:
  - "Gap assessment for pipeline/API/frontend flow"
affects: [pipeline, api, frontend]

# Tech tracking
tech-stack:
  added: []
  patterns: []

key-files:
  created: []
  modified:
    - backend/api/routers/worker.py
    - backend/api/schemas.py
    - backend/api/main.py
    - backend/api/ws.py
    - frontend/src/pages/Upload.tsx
    - frontend/src/pages/Analysis.tsx
    - frontend/src/hooks/useWebSocket.ts

key-decisions: []

patterns-established:
  - "Pending — to be documented after analysis"

requirements-completed: []

# Metrics
duration: TBC
completed: TBC
---

# Phase 3 Plan 01: Pipeline flow check Summary

**TBC — will update once all tasks are complete**

## Performance

- **Duration:** TBC
- **Started:** 2026-03-27T12:41:08Z
- **Completed:** TBC
- **Tasks:** 0/3
- **Files modified:** 0

## Findings

### Worker endpoints review
- Worker auth is optional: `verify_worker_key` only checks `X-Worker-Key` when `WORKER_API_KEY` is set, so production requests are unauthenticated if the env var is unset.
- `/worker/pending` atomically claims a job but does not record which worker claimed it, does not set `startedAt`, and returns an empty `modelConfig`, so workers lack mode/config details and cannot be tied back for lease management.
- `/analysis/{id}/status` accepts arbitrary `status`, `progress`, and `currentStage` values with no validation or clamping; missing analyses return `{success: False}` instead of an HTTP error, and any caller with the worker key can override status.
- `/analysis/{id}/complete` treats the analytics payload as a blob (stored in `analyticsDataUrl` despite being JSON), does not validate required fields, drops existing statistics before verifying payload, and can commit large JSON without size guards.
- WebSocket endpoint `/ws/{analysis_id}` has no authentication and allows any client to subscribe to any analysis ID, so progress updates are publicly readable.
- `/worker/upload-video` accepts base64 video uploads without size/type limits and returns a URL without linking it to an analysis record, leaving ownership/cleanup undefined.

## Accomplishments
- Pending — will fill after remaining tasks.

## Files Created/Modified
- Pending — will fill after remaining tasks.

## Decisions Made
- None yet.

## Deviations from Plan

## Issues Encountered
- None so far.

## Next Phase Readiness
- Pending remaining tasks.
