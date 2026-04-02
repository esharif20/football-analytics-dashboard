# Requirements: Football Analytics Dashboard

**Defined:** 2026-03-28
**Core Value:** Analysts can upload a match video and get automated tactical analytics without manual annotation.

## v0.5 Requirements

Requirements for analysis visualization overhaul and UI polish.

### Visualization Fixes

- [x] **VIZ-01**: Heatmap tab shows colored grid cells visible on dark background (not invisible due to blend mode)
- [x] **VIZ-02**: Pass network nodes positioned at correct pitch coordinates with thin, curved edges (not thick straight lines)
- [x] **VIZ-03**: Ball trajectory renders as a clean smooth path with directional gradient (not yellow spaghetti)

### Analytics

- [ ] **ANLY-01**: Team Compactness, Defensive Line, and Pressing Intensity charts display real data computed from tracks (no "Planned" badges)
- [ ] **ANLY-02**: Frame scrubber has play/pause, speed control, and keyboard shortcuts

### UI Polish

- [ ] **UI-01**: Consistent page layout with no random large gaps, uniform spacing, and section labels
- [ ] **UI-02**: `pnpm build` succeeds and all existing pytest + Playwright tests pass after changes

## Completed Requirements (Previous Milestones)

### v0.2 — Codebase Hardening & Supabase Migration

- [x] **MANUS-01**: Model download URLs configurable via env vars
- [x] **MANUS-02**: Hero images use local assets
- [x] **MANUS-03**: Auth localStorage key renamed
- [x] **MANUS-04**: .env.example files documented
- [x] **DB-01**: asyncpg driver to Supabase PostgreSQL
- [x] **DB-02**: PostgreSQL-compatible types/defaults
- [x] **DB-03**: Alembic baseline migration
- [x] **DB-04**: Runtime ALTER TABLE hacks removed
- [x] **DB-05**: CORS origins configurable
- [x] **QUAL-01**: Analysis.tsx decomposed
- [x] **QUAL-02**: Dead code removed
- [x] **QUAL-03**: next-themes removed
- [x] **QUAL-04**: JWT_SECRET validation
- [x] **QUAL-05**: AutoLogin guard
- [x] **QUAL-06**: ruff configured
- [x] **QUAL-07**: ESLint + Prettier configured
- [x] **TEST-01**: Backend pytest suite
- [x] **TEST-02**: Frontend vitest suite
- [x] **TEST-03**: CI pipeline expansion

### v0.3 — Database Redesign & Time-Series Tracks

- [x] **DB-R01**: FK constraints (ON DELETE CASCADE)
- [x] **DB-R02**: Performance indexes
- [x] **DB-R03**: RLS policies (permissive)
- [x] **DB-R04**: SQLAlchemy models match schema
- [x] **DB-R05**: Per-frame tracking data export
- [x] **DB-R06**: POST /api/worker/tracks endpoint
- [x] **DB-R07**: GET /api/tracks with pagination

## Future Requirements

Deferred to subsequent milestones. Tracked but not in current roadmap.

### Database Enhancements

- **DB-F01**: TIMESTAMP columns upgraded to timezone-aware (TIMESTAMP WITH TIME ZONE)
- **DB-F02**: Native PostgreSQL enums vs VARCHAR+CHECK evaluation

### Deployment

- **DEPLOY-01**: Production deployment pipeline (Vercel, AWS, or similar)
- **DEPLOY-02**: Staging environment configuration

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Pipeline algorithm changes | Pipeline must remain functionally identical |
| New data collection features | Focus is fixing existing visualizations, not adding new ones |
| OAuth/SSO integration | Current auto-login dev mode stays |
| Backend API changes | Viz fixes are frontend-only; backend already serves correct data |
| Mobile responsive redesign | Desktop-first; responsive deferred |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| VIZ-01 | Phase 11 | Complete |
| VIZ-02 | Phase 11 | Complete |
| VIZ-03 | Phase 11 | Complete |
| ANLY-01 | Phase 12 | Pending |
| ANLY-02 | Phase 12 | Pending |
| UI-01 | Phase 13 | Pending |
| UI-02 | Phase 13 | Pending |

**Coverage:**
- v0.5 requirements: 7 total
- Mapped to phases: 7
- Unmapped: 0 ✓

---
*Requirements defined: 2026-03-28*
*Last updated: 2026-04-02 after milestone v0.5 requirements defined*
