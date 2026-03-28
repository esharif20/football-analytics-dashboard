# Phase 4: Manus Dependency Removal - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md -- this log preserves the alternatives considered.

**Date:** 2026-03-28
**Phase:** 04-manus-dependency-removal
**Areas discussed:** Model URL fallback, CORS & env defaults

---

## Model URL Fallback

| Option | Description | Selected |
|--------|-------------|----------|
| Fail fast with clear error | Worker logs which env vars are missing and exits immediately. Forces explicit configuration. | Yes |
| Check local models/ dir first | Worker looks in backend/pipeline/models/ for .pt files first, then fails if not found AND env vars empty. | |
| Keep manuscdn as default fallback | Env vars override, but if empty, fall back to the current manuscdn.com URLs. | |

**User's choice:** Fail fast with clear error (Recommended)
**Notes:** No fallback behavior. Worker must have all 3 MODEL_URL_* env vars configured or it refuses to start.

---

## CORS & Env Defaults

| Option | Description | Selected |
|--------|-------------|----------|
| Dev-friendly defaults | CORS_ORIGINS defaults to localhost ports. JWT_SECRET has placeholder. Ready for cp and use. | Yes |
| Strict empty defaults | All security-sensitive vars empty. Forces explicit configuration even for local dev. | |
| Separate dev/prod examples | Both .env.example and .env.production.example. More files but clearer intent. | |

**User's choice:** Dev-friendly defaults (Recommended)
**Notes:** Single .env.example per directory with dev-ready defaults and comments.

---

## Hero Image Strategy

**Not discussed** -- user did not select this area. Claude's discretion: use local placeholder images.

## Claude's Discretion

- Placeholder image style for hero images (gradient, SVG, or stock photo)
- Exact error message wording for missing model URLs
- Whether CORS_ORIGINS goes into Settings class or direct os.getenv

## Deferred Ideas

None
