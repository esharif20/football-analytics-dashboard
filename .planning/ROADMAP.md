---
milestone: v0.5
status: active
version: 2.0
---

# Roadmap

## Milestone v0.1 Objectives (Complete)

- Planning artifacts created and codebase mapped
- Development environments reproducible (frontend + backend)
- CV pipeline <-> API <-> frontend flow validated and gaps documented

## Milestone v0.2 Codebase Hardening & Supabase Migration (Complete)

Remove all Manus platform dependencies, migrate from Docker MySQL to Supabase PostgreSQL, harden codebase with testing, linting, security, and SWE best practices.

## Milestone v0.3 Database Redesign & Time-Series Tracks (Complete)

FK constraints, indexes, RLS, per-frame tracking data export and API.

## Milestone v0.4 Interactive Analysis with Real Tracking Data (Complete)

Wired real tracking data to pitch visualizations, added frame scrubber.

## Milestone v0.5 Analysis Viz Overhaul & UI Polish

Fix all broken/ugly visualizations, replace placeholder data with real computed metrics, and polish the analysis page UI.

## Phase Breakdown

### Previous Milestones (Complete)

| Phase | Name | Goal | Status |
|-------|------|------|--------|
| 4 | Manus Dependency Removal | Zero Manus platform references | complete |
| 5 | Supabase Migration & Alembic | PostgreSQL with managed migrations | complete |
| 6 | Frontend Decomposition & Code Quality | Maintainable patterns, security guardrails | complete |
| 7 | Testing, Linting & CI | Automated tests, linting, CI | complete |
| 8 | Database Redesign & Time-Series Tracks | FK constraints, indexes, RLS, tracks | complete |
| 9 | Wire Tracks to Pitch Visualizations | Real data in viz components | complete |

### Milestone v0.5

| Phase | Name | Goal | Requirements | Status |
|-------|------|------|--------------|--------|
| 11 | Visualization Fixes | 2/2 | Complete   | 2026-04-02 |
| 12 | Computed Metrics & Frame Scrubber | Real tactical metrics from tracks + playback controls | ANLY-01, ANLY-02 | pending |
| 13 | Layout Polish & Verification | Consistent layout, all tests pass | UI-01, UI-02 | pending |
| 14 | Multi-Model Evaluation Comparison | 2/3 | In Progress|  |

## Phase Details

### Phase 11: Visualization Fixes

**Goal:** All three pitch viz components render correctly
**Requirements:** VIZ-01, VIZ-02, VIZ-03
**Plans:** 2/2 plans complete

Plans:
- [x] 11-01-PLAN.md — Fix HeatmapView z-index and PassNetworkView arrowheads/node class (VIZ-01, VIZ-02)
- [x] 11-02-PLAN.md — Fix BallTrajectoryDiagram white→amber gradient trail (VIZ-03)

**Success Criteria:**
1. Heatmap displays colored grid cells visible against dark background (no blend-mode hiding)
2. Pass network nodes at correct pitch coordinates with thin curved edges
3. Ball trajectory renders as smooth path with directional gradient (not yellow spaghetti)

### Phase 12: Computed Metrics & Frame Scrubber

**Goal:** Real tactical metrics from tracks + playback controls
**Requirements:** ANLY-01, ANLY-02

**Success Criteria:**
1. Team Compactness chart shows real values from player position spread (no "Planned" badge)
2. Defensive Line chart shows computed avg y-coordinate of deepest 4 outfield players (no "Planned" badge)
3. Pressing Intensity chart shows real velocity/distance-to-ball metrics (no "Planned" badge)
4. Frame scrubber play/pause toggle auto-advances frames at selected speed
5. Speed control (0.5x/1x/2x) and keyboard shortcuts (Space, arrow keys)

### Phase 13: Layout Polish & Verification

**Goal:** Consistent layout, all tests pass
**Requirements:** UI-01, UI-02

**Success Criteria:**
1. No unexplained large gaps; uniform spacing between sections
2. Every viz section has a clear label/heading
3. `pnpm build` completes with zero errors
4. All pytest tests pass
5. All Playwright E2E tests pass

### Phase 14: Multi-Model Evaluation Comparison

**Goal:** Add HuggingFace + OpenAI providers to eval framework, run all no-annotation evaluations across multiple models
**Requirements:** EVAL-01 (HF provider), EVAL-02 (multi-model grounding), EVAL-03 (multi-model VLM)
**Depends on:** Pipeline output data from RunPod

Plans:
- [ ] TBD (run /gsd:plan-phase 14 to break down)

**Success Criteria:**
1. `HuggingFaceProvider` in `llm_providers.py` using `huggingface_hub` InferenceClient (Mistral-7B-Instruct)
2. `OpenAIVisionProvider` in `vlm_comparison.py` for GPT-4o-mini vision
3. `llm_grounding.py` accepts `--provider huggingface` and runs successfully
4. `vlm_comparison.py` accepts `--provider all` and runs Gemini + OpenAI vision matrix
5. `tracking_quality.py` runs and produces metrics + LaTeX tables
6. `llm_grounding.py` produces grounding comparison across 3 models x 3 formats x 4 analysis types
7. `vlm_comparison.py` produces 2 models x 3 conditions comparison matrix
