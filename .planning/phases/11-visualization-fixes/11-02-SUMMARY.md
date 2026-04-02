---
phase: 11-visualization-fixes
plan: 02
subsystem: ui
tags: [react, svg, ball-trajectory, animation, gradient]

# Dependency graph
requires:
  - phase: 11-visualization-fixes
    provides: VIZ-01 heatmap z-index fix and VIZ-02 pass network arrowhead removal
provides:
  - BallTrajectoryDiagram with 5-segment white→amber color gradient trail (SEGMENT_STYLE)
  - Ball head pulse animation with r values=1.5;2.8;1.5 and stroke-opacity animate
affects: [11-visualization-fixes, analysis-page]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "SEGMENT_STYLE constant: 5 entries (tail→head) each with opacity/strokeWidth/color; color encodes the gradient, opacity is secondary reinforcement"
    - "Segment path split: divide trajectoryPoints into N equal slices; render each as independent <path> with its own stroke color"

key-files:
  created: []
  modified:
    - frontend/src/pages/Analysis.tsx

key-decisions:
  - "Applied to Analysis.tsx (BallTrajectoryDiagram inline function) rather than non-existent PipelineInfo.tsx — same deviation pattern as Plan 11-01"
  - "Removed trajGrad linearGradient defs (no longer needed) while keeping trajGlow and trajLineGlow filters"
  - "Ball head pulse upgraded from values=1.5;2.2;1.5/2s to values=1.5;2.8;1.5/1.5s with added stroke-opacity animate per plan spec"

patterns-established:
  - "Trail gradient via segment colors: use discrete path segments with explicit RGBA color per segment rather than SVG linearGradient on a single path — more control over per-segment opacity and thickness"

requirements-completed: [VIZ-03]

# Metrics
duration: 6min
completed: 2026-04-02
---

# Phase 11 Plan 02: Ball Trajectory Gradient Summary

**SEGMENT_STYLE 5-entry white→amber gradient trail replaces single amber stroke in BallTrajectoryDiagram; tail segments use near-white rgba(255,255,255,...) and head uses full amber rgba(251,191,36,1)**

## Performance

- **Duration:** ~6 min
- **Started:** 2026-04-02T00:00:00Z
- **Completed:** 2026-04-02T00:06:00Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments

- Added `SEGMENT_STYLE` constant with 5 entries (index 0 = tail/white, index 4 = head/amber), each with `opacity`, `strokeWidth`, and `color` fields
- Replaced single `stroke="url(#trajGrad)"` path with 5 discrete segment paths using `stroke={seg.color}`
- Upgraded ball head pulse animation: `r` values updated to `1.5;2.8;1.5` at 1.5s cycle; second `<animate>` for `stroke-opacity` values `0.9;0.3;0.9` added
- Removed `trajGrad` linearGradient (no longer needed); kept `trajGlow` and `trajLineGlow` filters

## Task Commits

1. **Task 1: Add SEGMENT_STYLE and white→amber gradient trail (VIZ-03)** - `a6bd5b3` (feat)

## Files Created/Modified

- `frontend/src/pages/Analysis.tsx` - SEGMENT_STYLE constant added, BallTrajectoryDiagram refactored to use 5 segment paths with per-segment color, ball head pulse upgraded

## Decisions Made

- Applied changes to `Analysis.tsx` (where BallTrajectoryDiagram actually lives) rather than the non-existent `PipelineInfo.tsx` — same deviation pattern as Plan 11-01 which found the same issue
- Removed the `trajGrad` linearGradient `<defs>` entry since the segmented approach replaces it; kept the `trajGlow` filter for the head segment and `trajLineGlow` for pitch lines

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Applied changes to Analysis.tsx instead of PipelineInfo.tsx**
- **Found during:** Task 1 (pre-read of target file)
- **Issue:** Plan referenced `frontend/src/pages/analysis/components/PipelineInfo.tsx` which does not exist. The `BallTrajectoryDiagram` component is still inline in the `Analysis.tsx` monolith — same finding as Plan 11-01.
- **Fix:** Applied SEGMENT_STYLE constant addition and path segmentation directly to `Analysis.tsx` at the `BallTrajectoryDiagram` function (lines 892–980)
- **Files modified:** `frontend/src/pages/Analysis.tsx`
- **Verification:** All acceptance criteria pass (rgba(255,255,255 count >= 2, seg.color in stroke, zero hardcoded amber path stroke, both animate elements present); build exits 0
- **Committed in:** a6bd5b3

**2. [Rule 1 - Bug] Ball head pulse values updated to match plan spec**
- **Found during:** Task 1 (review of existing code)
- **Issue:** Existing end marker used `values="1.5;2.2;1.5" dur="2s"` with no stroke-opacity animate. Plan spec requires `values="1.5;2.8;1.5" dur="1.5s"` plus `stroke-opacity` animate.
- **Fix:** Updated r values and duration; added second `<animate attributeName="stroke-opacity" values="0.9;0.3;0.9" ...>` element
- **Files modified:** `frontend/src/pages/Analysis.tsx`
- **Committed in:** a6bd5b3

---

**Total deviations:** 2 auto-fixed (1 blocking — wrong target file; 1 bug — pulse spec mismatch)
**Impact on plan:** Both fixes achieve the identical visual outcome the plan specified. No scope creep.

## Issues Encountered

None beyond the target file path deviation above.

## Known Stubs

None — all 5 SEGMENT_STYLE entries have real color values; the ball head renders with real trajectory data when available and falls back to randomized demo data otherwise (pre-existing behavior, not introduced here).

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- VIZ-03 fixed. Ball trajectory trail now fades from near-white at the tail to full amber at the head.
- When the Analysis.tsx decomposition eventually runs, the VIZ-03 SEGMENT_STYLE fix will need to be preserved in the extracted component file (same note as VIZ-01/02 from Plan 11-01).

---
*Phase: 11-visualization-fixes*
*Completed: 2026-04-02*
