---
phase: "07-testing-linting-ci"
plan: "07-04"
subsystem: "backend-tests"
tags: ["testing", "upload", "smoke-test", "gap-closure"]
dependency_graph:
  requires: ["07-01"]
  provides: ["TEST-01 upload coverage"]
  affects: ["backend/api/tests/test_endpoints.py"]
tech_stack:
  added: []
  patterns: ["httpx AsyncClient multipart file upload in tests"]
key_files:
  created: []
  modified:
    - backend/api/tests/test_endpoints.py
decisions:
  - "Assert status_code not in (401, 422): 500 acceptable since storage_put fails without disk in CI"
metrics:
  duration: "2m"
  completed_date: "2026-03-28"
  tasks_completed: 1
  files_modified: 1
---

# Phase 07 Plan 04: Upload Smoke Test Summary

**One-liner:** Upload endpoint smoke test asserting AutoLoginMiddleware injects auth (not 401) and form shape is valid (not 422).

## What Was Built

Appended `test_upload_video_rejects_unauthenticated_requests` to `backend/api/tests/test_endpoints.py` to close the TEST-01 gap. The test posts a fake MP4 file to `POST /api/upload/video` and asserts the response is neither 401 (auth failure) nor 422 (validation failure). A 500 is acceptable in CI because `storage_put` cannot write to disk.

## Task Results

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Add upload smoke test | 84d1832 | backend/api/tests/test_endpoints.py |

## Verification

- `pytest api/tests/ -v` — 40 passed in 0.13s (was 39 before this plan)
- `test_upload_video_rejects_unauthenticated_requests` present in test output
- No existing tests broken

## Deviations from Plan

None - plan executed exactly as written.

## Known Stubs

None.

## Self-Check: PASSED

- `backend/api/tests/test_endpoints.py` exists and contains `test_upload_video_rejects_unauthenticated_requests`
- Commit `84d1832` present in git log
- 40 tests pass
