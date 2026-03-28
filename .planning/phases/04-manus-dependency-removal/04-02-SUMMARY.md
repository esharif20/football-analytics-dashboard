---
phase: 04-manus-dependency-removal
plan: 02
subsystem: ui, infra
tags: [svg, localStorage, env-vars, manus-removal]

requires:
  - phase: 04-manus-dependency-removal
    provides: "Model URL env var migration (plan 01)"
provides:
  - "Local SVG hero images replacing manuscdn.com CDN"
  - "Renamed auth localStorage key (football-dashboard-user)"
  - "backend/.env.example with all required env vars"
  - "frontend/.env.example with VITE_ env vars"
affects: [05-supabase-migration, onboarding, developer-setup]

tech-stack:
  added: []
  patterns: [local-asset-hosting, env-var-documentation]

key-files:
  created:
    - frontend/public/images/stadium.svg
    - frontend/public/images/heatmap.svg
    - frontend/public/images/ai-sports.svg
    - backend/.env.example
    - frontend/.env.example
  modified:
    - frontend/src/pages/Home.tsx
    - frontend/src/_core/hooks/useAuth.ts

key-decisions:
  - "SVG placeholders chosen over raster images for size and scalability"
  - "localStorage key renamed to football-dashboard-user for project identity"

patterns-established:
  - "Local assets in frontend/public/images/ for static imagery"
  - ".env.example files as onboarding documentation for env vars"

requirements-completed: [MANUS-02, MANUS-03, MANUS-04]

duration: 3min
completed: 2026-03-28
---

# Phase 04 Plan 02: Frontend Manus Removal and Env Documentation Summary

**Local SVG hero images replacing manuscdn.com CDN, renamed auth localStorage key, and .env.example files for backend and frontend**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-28T11:22:32Z
- **Completed:** 2026-03-28T11:25:30Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments
- Replaced all 3 manuscdn.com hero image URLs with local SVG placeholders
- Renamed auth localStorage key from manus-runtime-user-info to football-dashboard-user
- Created backend/.env.example documenting all 18 env vars (DB, JWT, CORS, models, worker, LLM)
- Created frontend/.env.example documenting VITE_OAUTH_PORTAL_URL and VITE_APP_ID
- Zero remaining manus references in frontend/src/

## Task Commits

Each task was committed atomically:

1. **Task 1: Replace hero images and rename localStorage key** - `7532f2c` (feat)
2. **Task 2: Create .env.example files** - `588a12a` (chore)

## Files Created/Modified
- `frontend/public/images/stadium.svg` - Green gradient SVG placeholder for stadium hero
- `frontend/public/images/heatmap.svg` - Orange-red gradient SVG placeholder for heatmap hero
- `frontend/public/images/ai-sports.svg` - Blue-purple gradient SVG placeholder for AI sports hero
- `frontend/src/pages/Home.tsx` - Updated IMAGES const to reference local SVG paths
- `frontend/src/_core/hooks/useAuth.ts` - Renamed localStorage key to football-dashboard-user
- `backend/.env.example` - Full backend env var documentation
- `frontend/.env.example` - Frontend VITE_ env var documentation

## Decisions Made
- Used SVG placeholders (not raster images) for zero-dependency, scalable hero imagery
- Comment in Home.tsx cleaned to avoid any residual "manus" string matches

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Removed manus reference from code comment**
- **Found during:** Task 1 verification
- **Issue:** Comment "replaced manuscdn.com CDN references" still matched grep for "manus"
- **Fix:** Changed comment to "Local placeholder images for hero section"
- **Files modified:** frontend/src/pages/Home.tsx
- **Verification:** grep -rc "manus" frontend/src/ returns zero matches
- **Committed in:** 7532f2c (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Minor comment cleanup to ensure zero manus references. No scope creep.

## Issues Encountered
None

## Known Stubs
None - all images are functional SVGs, all env vars are documented with appropriate defaults or empty values.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All manuscdn.com and manus-runtime references eliminated from frontend
- Both .env.example files ready for developer onboarding
- Ready for Phase 05 (Supabase migration) which will update DATABASE_URL in .env.example

---
*Phase: 04-manus-dependency-removal*
*Completed: 2026-03-28*
