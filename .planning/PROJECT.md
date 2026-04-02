# Football Analytics Dashboard

## What This Is

An end-to-end football analytics platform where analysts upload match footage, run CV pipelines for detection/tracking/team classification, and review tactical insights in a responsive dashboard. Built with React 19 + Vite 6 frontend, FastAPI backend, and a YOLOv8/ByteTrack Python CV pipeline.

## Core Value

Analysts can upload a match video and get automated tactical analytics (possession, player tracking, events, radar view) without manual annotation.

## Current Milestone: v0.5 Analysis Viz Overhaul & UI Polish

**Goal:** Fix all broken/ugly visualizations, replace placeholder data with real computed metrics, and polish the analysis page UI.

**Target features:**
- Fix heatmap visibility (colored grid cells on dark background, not invisible blend mode)
- Fix pass network (correct pitch positions, thin curved edges)
- Fix ball trajectory (clean smooth path with directional gradient)
- Replace placeholder "Planned" badges with real computed metrics (Team Compactness, Defensive Line, Pressing Intensity)
- Frame scrubber enhancements (play/pause, speed control, keyboard shortcuts)
- Layout polish (consistent spacing, section labels, no large gaps)

## Requirements

### Validated

- Upload match footage and trigger CV pipeline analysis
- Real-time progress updates via WebSocket during processing
- View annotated video with player/ball detection overlays
- View possession stats, player tracking, events, radar view
- AI tactical commentary generation (Gemini/OpenAI)
- Worker polling architecture for GPU pipeline processing
- Manus platform dependencies removed (v0.2, Phase 4)
- Supabase PostgreSQL migration with Alembic (v0.2, Phase 5)
- Frontend decomposition, dead code removal, security hardening (v0.2, Phase 6)
- Backend/frontend testing, linting, CI (v0.2, Phase 7)
- DB redesign with FK constraints, indexes, RLS, time-series tracks (v0.3, Phase 8)
- Real tracking data wired to pitch visualizations (Phase 9)

### Active

- [ ] Fix heatmap visibility on dark backgrounds
- [ ] Fix pass network node positions and edge styling
- [ ] Fix ball trajectory rendering (smooth path, directional gradient)
- [ ] Compute real metrics for Team Compactness, Defensive Line, Pressing Intensity from tracks data
- [ ] Frame scrubber play/pause, speed control, keyboard shortcuts
- [ ] Layout polish (consistent spacing, section labels)

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
*Last updated: 2026-04-02 after Milestone v0.5 started*
