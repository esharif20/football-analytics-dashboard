# Phase 6: Frontend Decomposition & Code Quality - Context

**Gathered:** 2026-03-28
**Status:** Ready for planning

<domain>
## Phase Boundary

Decompose the Analysis.tsx monolith (2358 lines, 27 inline components) into maintainable sub-components, remove dead code, and add security guardrails for production readiness. No visual changes — the page must render identically.

</domain>

<decisions>
## Implementation Decisions

### D-01: Analysis.tsx Decomposition — By Visual Section
Split Analysis.tsx into ~7-8 files grouped by what the user sees on the page:
- `pages/analysis/index.tsx` — page shell, routing, top-level state, layout grid (~400 lines max)
- `pages/analysis/components/VideoPlayer.tsx` — video playback section
- `pages/analysis/components/StatsPanel.tsx` — possession stats, QuickStat cards, StatRow, StatusBadge
- `pages/analysis/components/ChartsGrid.tsx` — PossessionDonut, TeamPerformanceRadar, StatsComparisonBar, TeamShapeChart, DefensiveLineChart, PressingIntensityChart
- `pages/analysis/components/PitchVisualizations.tsx` — PitchRadar, PlayerNode, HeatmapView, PassNetworkView, VoronoiView, ModeSpecificTabs
- `pages/analysis/components/AICommentary.tsx` — AICommentarySection, EmptyCommentaryState
- `pages/analysis/components/PlayerStats.tsx` — PlayerStatsTable
- `pages/analysis/components/EventTimeline.tsx` — EventTimeline
- `pages/analysis/components/PipelineInfo.tsx` — PipelinePerformanceCard, ComingSoonCard, ProcessingStatus, BallTrajectoryDiagram, PlayerInteractionGraph

### D-02: Small Helpers Stay Inline
Tiny helper components (<30 lines) like ChartTooltip, QuickStat, StatusBadge, StatRow stay in their parent section file. Only extract if reused across multiple section files.

### D-03: Shared Context File
TeamColorsCtx context and shared TypeScript types live in `pages/analysis/context.ts`, co-located with the analysis page. All section components import from this shared file.

### D-04: AnimatedSection
The AnimatedSection wrapper component (IntersectionObserver scroll-in animation) should go in the shared context.ts or a small shared utils file since it's used across multiple sections.

### D-05: Keep next-themes (User Override of QUAL-03)
User decided to keep `next-themes` in package.json. It's only used by `sonner.tsx` (shadcn/ui toaster) and works fine. QUAL-03 requirement is overridden by user decision.

### D-06: Dead Code — Fix Duplicate Only
Only fix the duplicate `contentType` field in `backend/api/schemas.py` `WorkerUploadVideo` class (lines 203 & 205). Do NOT remove `videosApi.upload()` or `createWebSocket()` from `api-local.ts` — leave api-local.ts untouched.

### D-07: JWT_SECRET Production Guard — Crash on Startup
Add validation in `backend/api/config.py`: if `JWT_SECRET == "dev-secret"` AND `LOCAL_DEV_MODE` is not explicitly `true`, raise `ValueError` with a clear error message. Server refuses to start with insecure JWT in non-dev mode.

### D-08: LOCAL_DEV_MODE Default Stays True
Keep the current default of `LOCAL_DEV_MODE=true`. The QUAL-05 fix is ensuring AutoLoginMiddleware checks the explicit flag value — not changing the default. Developers must set `LOCAL_DEV_MODE=false` in production.

### D-09: AutoLogin Middleware Guard
AutoLoginMiddleware in `backend/api/auth.py` must check `settings.LOCAL_DEV_MODE` is explicitly `True` before activating. Log a warning on startup when auto-login is active.

### Claude's Discretion
- Exact file boundaries for borderline components (e.g., whether BallTrajectoryDiagram goes with PitchVisualizations or gets its own section)
- Import organization and barrel exports
- Whether to add a brief component docstring at the top of each new section file

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Frontend
- `frontend/src/pages/Analysis.tsx` — The 2358-line monolith being decomposed
- `frontend/src/lib/api-local.ts` — API client (no changes planned, but referenced for data flow understanding)
- `frontend/src/hooks/useWebSocket.ts` — WebSocket hook used by Analysis page
- `frontend/src/components/ui/` — shadcn/ui component library (53 components available)
- `frontend/package.json` — Dependency list (next-themes stays)

### Backend
- `backend/api/config.py` — Settings class with JWT_SECRET and LOCAL_DEV_MODE
- `backend/api/auth.py` — AutoLoginMiddleware implementation
- `backend/api/schemas.py` — Pydantic schemas (duplicate contentType fix needed)

### Planning
- `.planning/REQUIREMENTS.md` — QUAL-01 through QUAL-05 definitions
- `.planning/PROJECT.md` — Milestone context and constraints

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **shadcn/ui library** (53 components): Card, Tabs, Badge, Button, ScrollArea, Table, Separator — all used extensively in Analysis.tsx
- **Recharts**: Used for all chart components (PieChart, RadarChart, BarChart, AreaChart, LineChart, ScatterChart)
- **@tanstack/react-query**: useQuery/useMutation for all data fetching
- **useWebSocket hook**: Custom hook for real-time progress updates
- **ErrorBoundary component**: Existing error boundary at `frontend/src/components/ErrorBoundary.tsx`

### Established Patterns
- React Query for server state (no Redux/Zustand)
- shadcn/ui + Tailwind CSS for styling
- Custom context (TeamColorsCtx) for cross-component state
- Inline SVG for pitch visualizations (not canvas)

### Integration Points
- `pages/analysis/index.tsx` will be the new route target (React Router)
- Data fetching stays in index.tsx, passed as props to section components
- WebSocket connection stays in index.tsx, progress state passed down

</code_context>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches for the decomposition.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 06-frontend-decomposition-code-quality*
*Context gathered: 2026-03-28*
