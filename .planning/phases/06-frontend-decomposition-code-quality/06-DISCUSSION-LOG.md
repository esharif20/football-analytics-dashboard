# Phase 6: Frontend Decomposition & Code Quality - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-03-28
**Phase:** 06-frontend-decomposition-code-quality
**Areas discussed:** Analysis.tsx decomposition, Dead code & deps, Security guardrails

---

## Analysis.tsx Decomposition Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| By visual section | Group by what user sees: VideoPlayer, StatsPanel, ChartsGrid, PitchVisualizations, AICommentary, EventTimeline, PlayerStats. ~7-8 files. | ✓ |
| By data domain | Group by data source: PossessionStats, TrackingViz, EventsPanel, etc. ~5-6 files. | |
| Flat — one per component | Extract all 27 components into individual files. ~27 files. | |

**User's choice:** By visual section (Recommended)
**Notes:** Mirrors the page layout, easiest to navigate.

---

### Helper Components

| Option | Description | Selected |
|--------|-------------|----------|
| Keep inline in parent | Small helpers (<30 lines) stay in parent section file | ✓ |
| Shared helpers file | Create pages/analysis/components/shared.tsx | |

**User's choice:** Keep inline in parent (Recommended)
**Notes:** Fewer files, simpler imports.

---

### Shared Context Location

| Option | Description | Selected |
|--------|-------------|----------|
| pages/analysis/context.ts | Co-located with analysis page | ✓ |
| Keep in index.tsx | Context stays in main file, props drilling | |

**User's choice:** pages/analysis/context.ts (Recommended)

---

## Dead Code & Dependency Cleanup

### next-themes

| Option | Description | Selected |
|--------|-------------|----------|
| Remove and patch sonner.tsx | Remove dep, patch toaster to use simpler dark mode check | |
| Keep it | It works, it's small, sonner.tsx needs it | ✓ |

**User's choice:** Keep it
**Notes:** User overrides QUAL-03 requirement. next-themes stays.

---

### Dead Code Scope

| Option | Description | Selected |
|--------|-------------|----------|
| Remove both | Delete base64 upload + fix duplicate contentType + remove createWebSocket | |
| Only fix duplicate | Just fix schemas.py duplicate, leave api-local.ts alone | ✓ |

**User's choice:** Only fix duplicate
**Notes:** Conservative approach — don't remove functions that might be used somewhere unexpected.

---

## Security Guardrails

### JWT_SECRET Validation

| Option | Description | Selected |
|--------|-------------|----------|
| Crash on startup | ValueError if JWT_SECRET='dev-secret' and LOCAL_DEV_MODE not true | ✓ |
| Log warning but start | Print loud warning but allow startup | |
| Crash only if explicit prod flag | Add ENVIRONMENT=production env var check | |

**User's choice:** Crash on startup (Recommended)

---

### LOCAL_DEV_MODE Default

| Option | Description | Selected |
|--------|-------------|----------|
| Default to false | Change default, devs must explicitly set true | |
| Keep default true | Keep current behavior | ✓ |

**User's choice:** Keep default true
**Notes:** QUAL-05 only requires the explicit flag check, not changing the default.

---

## Claude's Discretion

- Exact file boundaries for borderline components
- Import organization and barrel exports
- Component docstrings

## Deferred Ideas

None — discussion stayed within phase scope.
