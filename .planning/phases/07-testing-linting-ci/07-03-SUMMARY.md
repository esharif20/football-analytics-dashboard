---
phase: "07-testing-linting-ci"
plan: "07-03"
subsystem: "ci"
tags: [github-actions, ci, ruff, pytest, pnpm, frontend, backend]
dependency_graph:
  requires: ["07-01", "07-02"]
  provides: [ci-backend-job, ci-frontend-fixed]
  affects: [.github/workflows/ci.yml]
tech_stack:
  added: [actions/setup-python@v5, actions/setup-node@v4 (fixed cache)]
  patterns: [parallel-CI-jobs, pip-cache-by-requirements.txt]
key_files:
  created: []
  modified:
    - .github/workflows/ci.yml
decisions:
  - "DATABASE_URL set to postgresql+asyncpg://skip:skip@skip/skip in CI because config.py validates URL format at import time — dependency_overrides prevent actual connection"
metrics:
  duration: "3m"
  completed: "2026-03-28"
  tasks_completed: 1
  files_created: 0
  files_modified: 1
requirements_satisfied: [TEST-03]
---

# Phase 07 Plan 03: CI Pipeline Fix + Backend Job Summary

**One-liner:** Fixed pnpm cache path bug in frontend CI job and added parallel backend job running ruff lint + pytest under LOCAL_DEV_MODE with SQLite-compatible DATABASE_URL override.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Fix frontend CI cache path bug and add backend CI job | 9d6ff64 | .github/workflows/ci.yml |

## Verification Results

- `python3 -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"` — YAML valid, no error
- `grep "cache-dependency-path" .github/workflows/ci.yml` — shows `pnpm-lock.yaml` (repo root, not `frontend/pnpm-lock.yaml`)
- `grep -c "backend:" .github/workflows/ci.yml` — 1 match (backend job present)
- `grep "pnpm lint\|pnpm test\|format:check" .github/workflows/ci.yml` — all three steps present in frontend job
- Both jobs triggered on push and PR to main

## Deviations from Plan

None — plan executed exactly as written.

## Known Stubs

None — CI configuration only; no UI-rendering data paths.

## Self-Check: PASSED

- `.github/workflows/ci.yml` exists and is valid YAML
- Commit `9d6ff64` present in git log
