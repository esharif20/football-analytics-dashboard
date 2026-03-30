---
status: complete
phase: 08-database-redesign-time-series-tracks
source: [08-01-SUMMARY.md, 08-02-SUMMARY.md, 08-03-SUMMARY.md]
started: 2026-03-30T10:30:00Z
updated: 2026-03-30T10:30:00Z
---

## Current Test

[testing complete]

## Tests

### 1. Cold Start Smoke Test
expected: Kill any running server. Start fresh: `cd backend && uvicorn api.main:app --port 8000 --reload`. Server boots without errors. GET http://localhost:8000/health or http://localhost:8000/docs returns a live response with no FK-related startup errors in the console.
result: pass

### 2. Apply Supabase Migration
expected: Run `supabase migration up` (or paste the file contents into the Supabase SQL editor). Migration completes without errors. The 7 FK constraints, 8 indexes, and RLS policies are applied. Re-running the migration is safe (idempotent — no duplicate constraint errors).
result: pass

### 3. GET /api/tracks/{id} — Paginated Response
expected: With the API running, call `GET http://localhost:8000/api/tracks/{some_analysis_id}?limit=10&offset=0`. Response is a JSON array (empty array `[]` is fine if no pipeline has run yet). No 500 error.
result: pass

### 4. GET /api/tracks/{id} — Frame Range Filtering
expected: Call `GET http://localhost:8000/api/tracks/{id}?frame_start=0&frame_end=100&limit=100`. Response is a JSON array. No 500 error. If tracks exist, only frames 0–100 are returned.
result: pass

### 5. Pipeline Produces _tracks.json
expected: Run a full pipeline analysis (requires GPU worker). After completion, a `{video_name}_tracks.json` file appears in the pipeline output directory alongside the existing `{video_name}_analytics.json`.
result: skipped
reason: GPU worker not available

### 6. Tracks Table Populated After Pipeline Run
expected: After a successful pipeline run, call `GET /api/tracks/{analysis_id}`. Response contains track records (non-empty array). Ball and player positions are present in the returned data.
result: pass

## Summary

total: 6
passed: 5
issues: 0
pending: 0
skipped: 1
blocked: 0

## Gaps

[none yet]
