# Phase 11: Visualization Fixes - Context

**Gathered:** 2026-04-02
**Status:** Ready for planning

<domain>
## Phase Boundary

Fix three broken SVG visualization components on the analysis page: heatmap, pass network, and ball trajectory. All fixes are frontend-only — the backend already serves correct data. No visual redesigns, no new capabilities, no backend changes.

</domain>

<decisions>
## Implementation Decisions

### Heatmap (VIZ-01)
- **D-01:** Root cause is z-index stacking — `pitch-container::after` vignette overlay sits at z-index 3, SVG at z-index 2. Cells are rendered but hidden. Fix by lifting the SVG above the vignette or restructuring the stacking context.
- **D-02:** After fixing visibility, keep team-colored cells (current approach — cell color = team color, opacity encodes intensity). No change to color scheme.

### Pass Network (VIZ-02)
- **D-03:** Remove arrowheads entirely — no `<marker id="arrowhead">` or `markerEnd` on paths.
- **D-04:** Encode pass frequency via edge thickness only (current `strokeWidth` scaling by weight). Opacity stays constant.
- **D-05:** Player nodes use the same `player-node` CSS class as the Radar tab — consistent appearance across all pitch viz tabs.

### Ball Trajectory (VIZ-03)
- **D-06:** Directional gradient: tail segment color starts near-white/pale (low opacity white), head segment is bright amber. Implement as multi-segment opacity + color shift from white→amber along the trail.
- **D-07:** Keep the pulse animation on the ball head marker (current `<animate>` elements stay).
- **D-08:** The "spaghetti" bug likely comes from all trajectory points rendering simultaneously without the trail window. Fix by ensuring the trail window + fade segments logic works correctly for the full point set.

### Claude's Discretion
- Exact white→amber segment color values (white at ~15% opacity for tail, ramping to full amber at head)
- Whether to address the z-index issue in CSS (`.index.css`) or inline in the SVG element
- Exact edge stroke-width range for pass network (current 0.15–1.2 is fine, or adjust)
- Whether to simplify the vignette `::after` to avoid future z-index conflicts across all pitch viz components

</decisions>

<specifics>
## Specific Ideas

- Pass network: "cleaner, less noisy" — user explicitly preferred removing arrowheads for visual simplicity
- Ball trajectory: "subtle" gradient — white-to-amber, not a dramatic multi-color shift

</specifics>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Visualization components
- `frontend/src/pages/analysis/components/PitchVisualizations.tsx` — `HeatmapView`, `PassNetworkView`, `VoronoiView`, `ModeSpecificTabs` implementations
- `frontend/src/pages/analysis/components/PipelineInfo.tsx` — `BallTrajectoryDiagram` implementation (lines ~282–480)

### Styling
- `frontend/src/index.css` — `pitch-container` CSS (lines ~384–440), `pitch-container::after` vignette (z-index: 3), `heatmap-gradient` class, `player-node` class

### Requirements
- `.planning/REQUIREMENTS.md` — VIZ-01, VIZ-02, VIZ-03 definitions and acceptance criteria
- `.planning/ROADMAP.md` — Phase 11 success criteria (3 items)

### No external specs — requirements fully captured above

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `player-node` CSS class — shared marker style used in Radar tab; pass network nodes should reuse this
- `pitch-container` CSS class — shared pitch background/aspect-ratio used by all viz components
- `useTeamColors()` hook — provides `TEAM1_HEX`, `TEAM2_HEX` for consistent team colors

### Established Patterns
- Inline SVG (not canvas) for all pitch visualizations
- `useMemo` for expensive path/cell computations
- `hasRealData` guard pattern — all components check for valid data before rendering real paths
- `viewBox="0 0 105 68"` — pitch coordinates (meters), consistent across all components
- Coordinate scaling: `(x / PITCH_WIDTH) * 100` for CSS positioning of HTML overlay elements

### Integration Points
- `ModeSpecificTabs` in `PitchVisualizations.tsx` — renders all three tab views; no changes needed there
- `BallTrajectoryDiagram` is rendered in `PipelineInfo.tsx` — receives `ballTrajectoryPoints`, `currentFrame`, `minFrame`, `maxFrame` props
- `HeatmapView` and `PassNetworkView` receive pre-computed analytics from the analysis page index

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 11-visualization-fixes*
*Context gathered: 2026-04-02*
