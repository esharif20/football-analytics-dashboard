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
  created:
    - .planning/phases/03-pipeline-integration-hardening/03-01-SUMMARY.md
  modified:
    - .planning/STATE.md
    - .planning/ROADMAP.md

key-decisions:
  - "None — discovery-only review"

patterns-established:
  - "Discovery only; no new implementation patterns established"

requirements-completed: []

# Metrics
duration: 4m
completed: 2026-03-27
---

# Phase 3 Plan 01: Pipeline flow check Summary

**Documented worker/API/frontend pipeline gaps and prioritized fixes for hardening**

## Performance

- **Duration:** ~4m
- **Started:** 2026-03-27T12:41:08Z
- **Completed:** 2026-03-27T12:44:50Z
- **Tasks:** 3/3
- **Files modified:** 3

## Findings

### Worker endpoints review
- Worker auth is optional: `verify_worker_key` only checks `X-Worker-Key` when `WORKER_API_KEY` is set, so production requests are unauthenticated if the env var is unset.
- `/worker/pending` atomically claims a job but does not record which worker claimed it, does not set `startedAt`, and returns an empty `modelConfig`, so workers lack mode/config details and cannot be tied back for lease management.
- `/analysis/{id}/status` accepts arbitrary `status`, `progress`, and `currentStage` values with no validation or clamping; missing analyses return `{success: False}` instead of an HTTP error, and any caller with the worker key can override status.
- `/analysis/{id}/complete` treats the analytics payload as a blob (stored in `analyticsDataUrl` despite being JSON), does not validate required fields, drops existing statistics before verifying payload, and can commit large JSON without size guards.
- WebSocket endpoint `/ws/{analysis_id}` has no authentication and allows any client to subscribe to any analysis ID, so progress updates are publicly readable.
- `/worker/upload-video` accepts base64 video uploads without size/type limits and returns a URL without linking it to an analysis record, leaving ownership/cleanup undefined.

### Frontend upload and analysis flow
- Upload page only sends `title`, `description`, `file`, `mode`, and `fresh` to the API; UI toggles for camera angle, model selection, and cache usage beyond `fresh` are not persisted to the backend, so pipeline workers never receive those choices.
- Upload uses raw `XMLHttpRequest` to `/api/upload/video` and assumes a JSON body with `{id}`; errors return generic toasts and do not handle auth/session expiry specifically.
- After upload, `analysisApi.create` is called without deduplication or validation of allowed modes; failures surface as generic toasts and do not roll back uploaded files.
- Analysis page gates data fetches on `status === "completed"`, so partial data from failed runs is hidden; processing stage UI expects stage IDs from `PROCESSING_STAGES`, but workers emit `queued`/custom stages, leading to unknown-stage displays.
- Real-time updates rely on `/ws/{analysis_id}` messages and fallback ETA polling; WebSocket connections are unauthenticated on the server and can be subscribed by any client, while the UI assumes private progress.
- Visualizations still render with `demoTrackingData` and `demoEvents` when no real tracking/events exist, so the dashboard shows synthetic positions/events rather than pipeline output unless tracks/events APIs are populated.

## Accomplishments
- Cataloged worker endpoint gaps across auth, leasing, payload validation, and storage expectations.
- Traced upload → analysis → realtime render flow and logged mismatches between UI toggles, API payloads, and websocket exposure.
- Produced actionable next-step list to align pipeline outputs, tighten auth, and remove demo data fallbacks.

## Task Commits

1. **Task 1: Review worker endpoints** — `9959b2d` (chore)
2. **Task 2: Trace frontend analysis flow** — `adbb9d3` (chore)
3. **Task 3: Summarize integration gaps** — `cbdaa11` (chore)

**Plan metadata:** `<pending docs commit>`

## Files Created/Modified
- `.planning/phases/03-pipeline-integration-hardening/03-01-SUMMARY.md` — Gap assessment notes for worker endpoints and frontend flow
- `.planning/STATE.md` — Progress counters and session metadata updated for Phase 3 Plan 01 completion
- `.planning/ROADMAP.md` — Phase 3 plan marked complete in roadmap progress table

## Decisions Made
- None — discovery-only review.

## Deviations from Plan
None — plan executed as written.

## Issues Encountered
None.

## Next Phase Readiness
- Require worker auth by default and track leases per worker; reject unauthenticated websocket subscribers.
- Validate worker status/completion payloads (status enum, progress bounds, stage IDs) and store analytics/tracks in dedicated tables or object storage instead of text blobs.
- Wire frontend toggles (camera angle, model selection, cache usage) into API payloads; replace demo tracking/events with real API data and handle failed runs gracefully.

## Self-Check

- Summary file present: FOUND
- Commits present: 9959b2d, adbb9d3, cbdaa11
- Status: **PASSED**
