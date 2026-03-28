# Phase 4: Manus Dependency Removal - Context

**Gathered:** 2026-03-28
**Status:** Ready for planning

<domain>
## Phase Boundary

Remove all references to the Manus AI platform from the codebase. This includes: 3 hardcoded manuscdn.com model download URLs in the pipeline worker, 3 manuscdn.com hero image URLs in the frontend Home page, 1 "manus-runtime-user-info" localStorage key in the auth hook, and creating proper .env.example configuration templates for both frontend and backend. Also make CORS origins configurable via environment variable.

</domain>

<decisions>
## Implementation Decisions

### Model URL Configuration
- **D-01:** Replace hardcoded MODEL_URLS dict in `backend/pipeline/worker.py` (lines 42-46) with env vars: `MODEL_URL_PLAYER`, `MODEL_URL_BALL`, `MODEL_URL_PITCH`
- **D-02:** Fail-fast behavior: if any MODEL_URL_* env var is empty/missing, worker logs which vars are missing and exits immediately with a clear error message. No fallback to manuscdn.com or local files.
- **D-03:** Model URL validation happens at worker startup before entering the poll loop

### Hero Images
- **D-04:** Replace 3 manuscdn.com image URLs in `frontend/src/pages/Home.tsx` (lines 28-32) with local placeholder assets in `frontend/public/images/`. Use free-to-use football/sports themed placeholder images or simple gradient/SVG placeholders.

### Auth localStorage Key
- **D-05:** Rename localStorage key from `"manus-runtime-user-info"` to `"football-dashboard-user"` in `frontend/src/hooks/useAuth.ts` (line 43). No migration of old key needed -- dev mode only.

### Environment Configuration
- **D-06:** Create `backend/.env.example` with dev-friendly defaults: `CORS_ORIGINS=http://localhost:5173,http://localhost:3000`, `JWT_SECRET=dev-secret-change-in-production`, `DATABASE_URL` with placeholder, all MODEL_URL_* vars with empty values and comments
- **D-07:** Create `frontend/.env.example` with `VITE_OAUTH_PORTAL_URL=` and `VITE_APP_ID=` (both empty, comments explaining local dev mode)
- **D-08:** Make CORS origins configurable via `CORS_ORIGINS` env var in `backend/api/config.py` and `backend/api/main.py`. Default to `http://localhost:5173,http://localhost:3000` for local dev.

### Claude's Discretion
- Exact wording of error messages when model URLs are missing
- Placeholder image style (gradient, SVG illustration, or stock photo)
- Whether to add CORS_ORIGINS to the Settings class or read directly via os.getenv

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Backend - Worker
- `backend/pipeline/worker.py` -- Lines 42-46: MODEL_URLS dict to replace. Lines 55-80: download_models() function that uses them.

### Backend - Config & CORS
- `backend/api/config.py` -- Settings class where env vars are registered
- `backend/api/main.py` -- Lines 35-40: CORS middleware setup with hardcoded origins

### Frontend - Manus References
- `frontend/src/pages/Home.tsx` -- Lines 28-32: manuscdn.com image URLs
- `frontend/src/hooks/useAuth.ts` -- Line 43: localStorage key "manus-runtime-user-info"

### Research
- `.planning/research/STACK.md` -- Stack context for this milestone
- `.planning/research/PITFALLS.md` -- Pitfalls to avoid

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `backend/api/config.py` Settings class: existing pattern for env var registration with defaults
- `frontend/public/` directory: already exists for static assets (favicon.png is there)

### Established Patterns
- Backend env vars: read via `os.getenv()` with defaults in Settings dataclass
- Worker env vars: read directly via `os.getenv()` at module level (not through Settings)
- Frontend env vars: accessed via `import.meta.env.VITE_*` in `frontend/src/const.ts`

### Integration Points
- `docker-compose.yml`: MySQL password is hardcoded (but DB migration is Phase 5 scope)
- `env.example` at project root: already exists with basic vars -- backend/.env.example and frontend/.env.example supplement this

</code_context>

<specifics>
## Specific Ideas

- Worker should validate all 3 model URLs at startup in one check, not fail one-at-a-time
- .env.example files should have inline comments explaining each variable's purpose
- CORS_ORIGINS should be comma-separated string parsed into a list

</specifics>

<deferred>
## Deferred Ideas

None -- discussion stayed within phase scope

</deferred>

---

*Phase: 04-manus-dependency-removal*
*Context gathered: 2026-03-28*
