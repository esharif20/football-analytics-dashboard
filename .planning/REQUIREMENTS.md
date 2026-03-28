# Requirements: Football Analytics Dashboard

**Defined:** 2026-03-28
**Core Value:** Analysts can upload a match video and get automated tactical analytics without manual annotation.

## v0.2 Requirements

Requirements for codebase hardening and Supabase migration. Each maps to roadmap phases.

### Manus Removal

- [x] **MANUS-01**: Model download URLs are configurable via env vars (MODEL_URL_PLAYER/BALL/PITCH), not hardcoded to manuscdn.com
- [x] **MANUS-02**: Home page hero images use local assets in frontend/public/images/ instead of manuscdn.com CDN
- [x] **MANUS-03**: Auth localStorage key renamed from "manus-runtime-user-info" to "football-dashboard-user"
- [x] **MANUS-04**: Backend and frontend .env.example files exist with all env vars documented

### Database Migration

- [x] **DB-01**: Backend uses asyncpg driver connecting to Supabase PostgreSQL instead of aiomysql/MySQL
- [x] **DB-02**: SQLAlchemy models use PostgreSQL-compatible server_defaults and types
- [x] **DB-03**: Alembic is initialized with a baseline migration covering all 7 tables
- [x] **DB-04**: Runtime ALTER TABLE hacks in worker router are removed
- [x] **DB-05**: CORS origins are configurable via CORS_ORIGINS env var

### Code Quality

- [x] **QUAL-01**: Analysis.tsx is decomposed into sub-components under pages/analysis/ (main file under 400 lines)
- [x] **QUAL-02**: Dead base64 upload function removed from api-local.ts; duplicate schema field fixed
- [x] **QUAL-03**: Unused next-themes dependency removed from frontend
- [x] **QUAL-04**: JWT_SECRET validation refuses startup in production with default "dev-secret"
- [x] **QUAL-05**: AutoLogin middleware only activates when LOCAL_DEV_MODE=true explicitly
- [ ] **QUAL-06**: ruff configured for backend Python linting
- [ ] **QUAL-07**: ESLint + Prettier configured for frontend TypeScript linting

### Testing

- [ ] **TEST-01**: Backend pytest suite covers health, upload, analysis, worker, and commentary endpoints
- [ ] **TEST-02**: Frontend vitest suite covers key components and hooks
- [ ] **TEST-03**: CI pipeline includes backend lint + test job alongside frontend

## Future Requirements

Deferred to subsequent milestones. Tracked but not in current roadmap.

### Database Enhancements

- **DB-F01**: ForeignKey declarations added to all models for referential integrity
- **DB-F02**: TIMESTAMP columns upgraded to timezone-aware (TIMESTAMP WITH TIME ZONE)
- **DB-F03**: Native PostgreSQL enums vs VARCHAR+CHECK evaluation

### Deployment

- **DEPLOY-01**: Production deployment pipeline (Vercel, AWS, or similar)
- **DEPLOY-02**: Staging environment configuration

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| New user-facing features | This milestone is purely structural/quality |
| Pipeline algorithm changes | Pipeline must remain functionally identical |
| OAuth/SSO integration | Current auto-login dev mode stays; auth hardening is security-only |
| Mobile responsive redesign | UI changes limited to Analysis.tsx decomposition (same visual output) |
| Database sharding/replication | Supabase handles this; out of scope for application code |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| MANUS-01 | Phase 4 | Complete |
| MANUS-02 | Phase 4 | Complete |
| MANUS-03 | Phase 4 | Complete |
| MANUS-04 | Phase 4 | Complete |
| DB-05 | Phase 4 | Complete |
| DB-01 | Phase 5 | Complete |
| DB-02 | Phase 5 | Complete |
| DB-03 | Phase 5 | Complete |
| DB-04 | Phase 5 | Complete |
| QUAL-01 | Phase 6 | Complete |
| QUAL-02 | Phase 6 | Complete |
| QUAL-03 | Phase 6 | Complete |
| QUAL-04 | Phase 6 | Complete |
| QUAL-05 | Phase 6 | Complete |
| QUAL-06 | Phase 7 | Pending |
| QUAL-07 | Phase 7 | Pending |
| TEST-01 | Phase 7 | Pending |
| TEST-02 | Phase 7 | Pending |
| TEST-03 | Phase 7 | Pending |

**Coverage:**
- v0.2 requirements: 19 total
- Mapped to phases: 19
- Unmapped: 0

---
*Requirements defined: 2026-03-28*
*Last updated: 2026-03-28 after initial definition*
