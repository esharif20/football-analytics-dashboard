---
phase: "08"
plan: "03"
subsystem: backend
tags: [tracks, worker, api, pagination, pipeline]
dependency_graph:
  requires: [08-01]
  provides: [worker-tracks-endpoint, paginated-tracks-get, pipeline-tracks-upload]
  affects: [backend/api/routers/worker.py, backend/api/routers/tracks.py, backend/api/schemas.py, backend/pipeline/worker.py]
tech_stack:
  added: []
  patterns: [batch-insert, paginated-query, worker-auth]
key_files:
  created: []
  modified:
    - backend/api/schemas.py
    - backend/api/routers/worker.py
    - backend/api/routers/tracks.py
    - backend/pipeline/worker.py
decisions:
  - possessionTeamId stored in teamFormations JSON field to avoid schema change
  - Failed track uploads log a warning but do not block analysis completion
  - limit capped at 500 per request in GET /tracks regardless of user-supplied limit
metrics:
  duration: "205s"
  completed: "2026-03-30"
  tasks: 2
  files: 4
---

# Phase 08 Plan 03: Tracks API Wiring Summary

**One-liner:** Worker-authenticated POST /worker/tracks endpoint with 100-frame batch inserts, paginated GET /tracks with frame range filtering, and pipeline worker reads _tracks.json to POST all frames after pipeline completion.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Add schemas + POST /worker/tracks + paginated GET /tracks | 43b75d0 | schemas.py, worker.py, tracks.py |
| 2 | Wire pipeline worker to read tracks JSON and POST batches | 8ccc0fa | pipeline/worker.py |

## What Was Built

### schemas.py
- `WorkerTrackFrame`: frameNumber, timestamp, ballPosition, playerPositions, possessionTeamId
- `WorkerTracksCreate`: wraps list[WorkerTrackFrame]

### backend/api/routers/worker.py
- `POST /worker/tracks/{analysis_id}`: requires X-Worker-Key auth, validates analysis exists, bulk-inserts Track rows, commits with rollback on error, returns `{"success": true, "inserted": N}`
- possessionTeamId stored in teamFormations JSON as `{"possessionTeamId": int}` — no schema change required

### backend/api/routers/tracks.py
- `GET /tracks/{analysis_id}` now accepts `offset`, `limit` (max 500), `frame_start`, `frame_end` query params
- Filtered query uses `Track.frameNumber >= frame_start` / `<= frame_end` with order_by + offset/limit

### backend/pipeline/worker.py
- `post_tracks_to_api(analysis_id, tracks_file_path)`: reads JSON, POSTs in BATCH_SIZE=100 chunks, returns False on first error
- Called in `run_pipeline()` after analytics load: reads `{video_name}_tracks.json` from pipeline output dir, logs warning and continues if file missing or upload fails

## Deviations from Plan

None - plan executed exactly as written.

## Verification

- `grep "WorkerTrackFrame\|WorkerTracksCreate" backend/api/schemas.py` — returns matches
- `grep "post_tracks\|/tracks/" backend/api/routers/worker.py` — returns matches
- `grep "frame_start\|frame_end\|offset\|limit" backend/api/routers/tracks.py` — returns matches
- `grep "post_tracks_to_api\|tracks_json_file\|BATCH_SIZE" backend/pipeline/worker.py` — returns matches
- `python3 -m pytest api/tests/ -x -q` — 40 passed

## Known Stubs

None — all data paths are fully wired.

## Self-Check: PASSED

- `backend/api/schemas.py` — FOUND (WorkerTrackFrame, WorkerTracksCreate classes)
- `backend/api/routers/worker.py` — FOUND (post_tracks endpoint)
- `backend/api/routers/tracks.py` — FOUND (paginated list_tracks)
- `backend/pipeline/worker.py` — FOUND (post_tracks_to_api + call site)
- Commit 43b75d0 — FOUND
- Commit 8ccc0fa — FOUND
