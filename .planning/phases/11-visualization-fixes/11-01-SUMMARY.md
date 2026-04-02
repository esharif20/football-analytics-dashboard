---
phase: 11-visualization-fixes
plan: 01
subsystem: ui
tags: [react, svg, css, z-index, pitch-visualization]

# Dependency graph
requires:
  - phase: 08-database-redesign-time-series-tracks
    provides: real tracking data wired to pitch visualizations
provides:
  - HeatmapView SVG visible above vignette overlay (z-index 4)
  - PassNetworkView with no arrowheads and player-node class on nodes
affects: [11-visualization-fixes, analysis-page]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "SVG z-index promotion: lift data SVGs above pitch-container::after vignette (z-index 3) by setting inline style zIndex to 4"
    - "Pass network edges: encode frequency via strokeWidth only, no directional markers"

key-files:
  created: []
  modified:
    - frontend/src/pages/Analysis.tsx

key-decisions:
  - "Apply fixes to Analysis.tsx (monolith) rather than PitchVisualizations.tsx — decomposition has not yet occurred, fixes unblocked by applying to the actual source"
  - "SVG z-index 4 approach (D-01): lift SVG above vignette rather than reducing vignette z-index to avoid affecting all pitch viz tabs"

patterns-established:
  - "All pitch-container data SVGs should use zIndex >= 4 to render above the ::after vignette at z-index 3"

requirements-completed: [VIZ-01, VIZ-02]

# Metrics
duration: 8min
completed: 2026-04-02
---

# Phase 11 Plan 01: Visualization Fixes Summary

**HeatmapView SVG lifted to z-index 4 above the vignette overlay; PassNetworkView arrowhead marker and markerEnd attribute removed, node divs confirmed using player-node class with --player-color**

## Performance

- **Duration:** ~8 min
- **Started:** 2026-04-02T00:00:00Z
- **Completed:** 2026-04-02T00:08:00Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments

- Fixed heatmap cell visibility: SVG was hidden under `pitch-container::after` vignette (z-index 3); promoted SVG to z-index 4
- Fixed pass network visual noise: deleted `<marker id="arrowhead">` definition and removed `markerEnd="url(#arrowhead)"` from all edge line elements
- Confirmed pass network node divs already have `className="player-node"` and `--player-color` CSS variable (no change needed)

## Task Commits

1. **Task 1 + Task 2: Fix HeatmapView z-index (VIZ-01) and remove PassNetworkView arrowheads (VIZ-02)** - `34f75be` (fix)

## Files Created/Modified

- `frontend/src/pages/Analysis.tsx` - HeatmapView SVG zIndex 2→4; PassNetworkView arrowhead marker deleted, markerEnd attribute removed

## Decisions Made

- Applied fixes directly to `Analysis.tsx` (the actual source) rather than the non-existent `frontend/src/pages/analysis/components/PitchVisualizations.tsx`. The CONTEXT.md anticipated a decomposition that has not yet occurred. The plan's CONTEXT.md listed the wrong file path. Fixes still achieve identical visual outcomes.
- Confirmed SVG z-index promotion approach (D-01) is safer than reducing the vignette z-index (which would affect all pitch viz tabs simultaneously).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Applied fixes to Analysis.tsx instead of PitchVisualizations.tsx**
- **Found during:** Task 1 (pre-read of target file)
- **Issue:** Plan referenced `frontend/src/pages/analysis/components/PitchVisualizations.tsx` which does not exist. The `HeatmapView` and `PassNetworkView` components are still inline in the `Analysis.tsx` monolith. The decomposition (planned in Phase 06-02) was documented but never executed.
- **Fix:** Applied all three targeted changes (z-index fix, marker deletion, markerEnd removal) directly to `Analysis.tsx` at the correct line locations
- **Files modified:** `frontend/src/pages/Analysis.tsx`
- **Verification:** grep confirms zIndex: 4, arrowhead count 0, markerEnd count 0; build exits 0
- **Committed in:** 34f75be

---

**Total deviations:** 1 auto-fixed (1 blocking — wrong target file path)
**Impact on plan:** Fix applied to correct location. Visual outcomes are identical to what the plan intended. No scope creep.

## Issues Encountered

None beyond the target file path deviation above.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- VIZ-01 and VIZ-02 fixed. Plan 11-02 (ball trajectory gradient) is independent and can proceed.
- When the Analysis.tsx decomposition eventually runs, the VIZ-01/VIZ-02 fixes will need to be preserved in the extracted component file.

---
*Phase: 11-visualization-fixes*
*Completed: 2026-04-02*
