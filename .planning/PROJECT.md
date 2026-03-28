# Football Analytics Dashboard

## What This Is

An end-to-end football analytics platform where analysts upload match footage, run CV pipelines for detection/tracking/team classification, and review tactical insights in a responsive dashboard. Built with React 19 + Vite 6 frontend, FastAPI backend, and a YOLOv8/ByteTrack Python CV pipeline.

## Core Value

Analysts can upload a match video and get automated tactical analytics (possession, player tracking, events, radar view) without manual annotation.

## Current Milestone: v0.2 Codebase Hardening & Supabase Migration

**Goal:** Remove all Manus platform dependencies, migrate from Docker MySQL to Supabase PostgreSQL, and harden codebase with proper testing, linting, security, and SWE best practices -- while keeping all existing functionality intact.

**Target features:**
- Remove all manuscdn.com CDN references (model URLs, hero images, localStorage keys)
- Migrate database from Docker MySQL to Supabase PostgreSQL (asyncpg)
- Add Alembic migration framework for schema management
- Decompose Analysis.tsx monolith (~2300 lines) into sub-components
- Remove dead code and unused dependencies (next-themes, dead base64 upload)
- Harden security (JWT validation in production, CORS env config, env var enforcement)
- Add backend unit tests (pytest + httpx) and frontend unit tests (vitest)
- Add linting (ruff for backend, ESLint + Prettier for frontend)
- Expand CI pipeline to cover backend lint + tests

## Requirements

### Validated

- Upload match footage and trigger CV pipeline analysis
- Real-time progress updates via WebSocket during processing
- View annotated video with player/ball detection overlays
- View possession stats, player tracking, events, radar view
- AI tactical commentary generation (Gemini/OpenAI)
- Worker polling architecture for GPU pipeline processing

### Active

- [x] Remove all Manus platform dependencies (Validated in Phase 4)
- [x] Migrate to Supabase PostgreSQL (Validated in Phase 5)
- [x] Add Alembic migration framework (Validated in Phase 5)
- [ ] Decompose Analysis.tsx into sub-components
- [ ] Remove dead code and unused dependencies
- [-] Security hardening (JWT, CORS, env vars) — CORS config done in Phase 4, JWT/env hardening in Phase 6
- [ ] Backend unit tests
- [ ] Frontend unit tests
- [ ] Linting setup (ruff + ESLint/Prettier)
- [ ] CI expansion for backend

### Out of Scope

- New user-facing features -- this milestone is purely structural/quality
- Changing the CV pipeline algorithms or models -- pipeline must remain functionally identical
- Deployment pipeline or hosting changes -- focus is codebase quality
- OAuth/SSO integration -- current auto-login dev mode stays

## Context

- Project was originally scaffolded on the Manus AI platform, leaving CDN URLs (manuscdn.com) for ML models and hero images, and a "manus-runtime" localStorage key
- Database is MySQL 8.0 in Docker container (port 3307), no migration tool
- No unit tests exist (frontend or backend); only 4 Playwright E2E spec files
- No linting configured; CI only checks frontend TypeScript + build
- Analysis.tsx is a 2300-line monolith with 25+ inline component definitions
- Unused dependencies: next-themes (Next.js lib in Vite project), potentially unused Radix packages
- Dead code: base64 upload path in api-local.ts, duplicate schema fields

## Constraints

- **Pipeline integrity**: backend/pipeline/src/ must remain functionally identical -- no algorithm or model changes
- **Folder structure**: frontend/ and backend/ stay as separate directories
- **Model hosting**: ML model download URLs become env vars; user provides hosting
- **Supabase**: User confirmed migration from Docker MySQL to Supabase PostgreSQL

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Supabase over Docker MySQL | Eliminate Docker DB dependency, free hosted tier, built-in features for future | Done — Phase 5 |
| Env vars for model URLs | Decouple from manuscdn.com CDN, let user host models anywhere | Done — Phase 4 |
| Alembic for migrations | Standard Python migration tool, replaces fragile runtime ALTER TABLE | Done — Phase 5 |
| ruff over flake8/black | Single tool for linting + formatting, faster, modern Python standard | -- Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd:transition`):
1. Requirements invalidated? -> Move to Out of Scope with reason
2. Requirements validated? -> Move to Validated with phase reference
3. New requirements emerged? -> Add to Active
4. Decisions to log? -> Add to Key Decisions
5. "What This Is" still accurate? -> Update if drifted

**After each milestone** (via `/gsd:complete-milestone`):
1. Full review of all sections
2. Core Value check -- still the right priority?
3. Audit Out of Scope -- reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-03-28 after Phase 5 (Supabase Migration & Alembic) complete*
