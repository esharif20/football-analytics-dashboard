## Plan Summary

**Plan:** 11-01 | **Phase:** 11 — Visualization Fixes | **Completed:** 2026-04-02

## Tasks

| Task | Description | Status |
|------|-------------|--------|
| 1 | HeatmapView SVG z-index 2→4 (VIZ-01) | ✓ |
| 2 | PassNetworkView: remove arrowhead marker + markerEnd (VIZ-02) | ✓ |

## Key Changes

- `PitchVisualizations.tsx` line 200: `zIndex: 2` → `zIndex: 4` — SVG lifts above `pitch-container::after` vignette (z-3)
- `PitchVisualizations.tsx`: `<marker id="arrowhead">` block deleted; `markerEnd="url(#arrowhead)"` removed from edge paths

## Self-Check: PASSED
