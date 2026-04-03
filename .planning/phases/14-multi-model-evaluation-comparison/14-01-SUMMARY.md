---
phase: 14-multi-model-evaluation-comparison
plan: "01"
subsystem: backend/evaluation
tags: [llm, evaluation, huggingface, multi-provider, grounding]
dependency_graph:
  requires: []
  provides: [HuggingFaceProvider, multi-provider-grounding-eval]
  affects: [backend/evaluation/llm_grounding.py, backend/api/services/llm_providers.py]
tech_stack:
  added: [huggingface-hub>=1.0.0]
  patterns: [LLMProvider ABC, lazy client initialization, TDD London School]
key_files:
  created:
    - backend/evaluation/tests/__init__.py
    - backend/evaluation/tests/conftest.py
    - backend/evaluation/tests/test_hf_provider.py
  modified:
    - backend/api/services/llm_providers.py
    - backend/api/requirements.txt
    - backend/evaluation/llm_grounding.py
decisions:
  - "HuggingFaceProvider follows the same lazy _get_client() pattern as OpenAIProvider for consistency"
  - "llm_grounding.py compacted to 492 lines via prose/verify_claim refactoring — no behavioral change"
  - "Section comment dividers removed to meet 500-line constraint without reducing functionality"
metrics:
  duration: 379s
  completed: "2026-04-03"
  tasks: 2
  files: 6
requirements: [EVAL-01, EVAL-02]
---

# Phase 14 Plan 01: HuggingFace Provider + Multi-Provider CLI Summary

**One-liner:** HuggingFaceProvider (Mistral-7B-Instruct-v0.3 via AsyncInferenceClient) registered in get_provider(), llm_grounding.py CLI extended with --provider huggingface and --provider all for cross-provider evaluation runs.

## Tasks Completed

| # | Task | Commit | Files |
|---|------|--------|-------|
| 1 (RED) | Add failing tests for HuggingFaceProvider | deb3fcc | tests/__init__.py, conftest.py, test_hf_provider.py |
| 1 (GREEN) | HuggingFaceProvider + dependency + registry | 3c2747c | llm_providers.py, requirements.txt |
| 2 | Wire multi-provider --provider all | fee20c2 | llm_grounding.py |

## What Was Built

**Task 1 — HuggingFaceProvider (TDD)**

- `HuggingFaceProvider` class in `backend/api/services/llm_providers.py` implementing `LLMProvider` ABC
  - Default model: `mistralai/Mistral-7B-Instruct-v0.3`
  - API key from constructor or `HUGGINGFACE_API_KEY` env var
  - Lazy `AsyncInferenceClient` initialization via `_get_client()`
  - `generate()` calls `chat.completions.create()` with temperature=0.7, max_tokens=4096
- `"huggingface": HuggingFaceProvider` registered in `get_provider()` provider dict
- Error message updated to include `HUGGINGFACE_API_KEY`
- `huggingface-hub>=1.0.0` added to `backend/api/requirements.txt`
- 9 unit tests passing: `is_available` (4), `generate` (2), `default_model` (2), `get_provider` (1)

**Task 2 — Multi-provider CLI**

- `--provider` choices updated: `["gemini", "openai", "huggingface", "all", "stub"]`
- `run_async()` refactored with `_run_provider()` helper encapsulating the format x analysis_type loop
- `providers_to_run = _REAL_PROVIDERS if config.provider == "all" else [config.provider]`
- Unavailable providers skipped with `"  Skipping {name} (no API key)"` message
- Per-provider artifacts: format comparison table, per-type table, format comparison plot, grounding summary plot, example claims JSON
- Cross-provider comparison table saved as `grounding_provider_comparison.tex` when multiple providers run
- File kept under 500 lines (492) via compaction of prose formatter, verify_claim, extract_claims, compute_grounding_score

## Verification Results

1. `pytest evaluation/tests/test_hf_provider.py -x -v` — 9 passed
2. `from api.services.llm_providers import get_provider, HuggingFaceProvider` — import OK
3. `from evaluation.llm_grounding import main` — import OK
4. `wc -l api/services/llm_providers.py` — 199 lines (under 200)
5. `wc -l evaluation/llm_grounding.py` — 492 lines (under 500)

## Decisions Made

- **HuggingFaceProvider lazy init:** Same `_get_client()` pattern as OpenAI/Gemini — avoids import at module load, consistent with existing providers.
- **File compaction approach:** Removed section comment dividers, compacted `verify_claim` with inner helper `_unverifiable()`, compacted formatters and extractors — no behavioral change, just style.
- **`_run_provider()` refactor:** The multi-provider loop cleanly replaces the old single-provider loop by extracting a per-provider async function. Reduces duplication and makes `run_async()` readable.

## Deviations from Plan

None — plan executed exactly as written. The file compaction was required by the 500-line constraint and was anticipated in the plan ("Keep the file under 500 lines").

## Known Stubs

None. Both `HuggingFaceProvider` and the multi-provider loop are fully wired. The actual LLM calls are gated by `is_available()` and require a real `HUGGINGFACE_API_KEY` at runtime.

## Self-Check: PASSED

- `backend/evaluation/tests/__init__.py` — FOUND
- `backend/evaluation/tests/conftest.py` — FOUND
- `backend/evaluation/tests/test_hf_provider.py` — FOUND
- `backend/api/services/llm_providers.py` — FOUND (contains HuggingFaceProvider, mistralai/Mistral-7B-Instruct-v0.3, AsyncInferenceClient, "huggingface")
- `backend/api/requirements.txt` — FOUND (contains huggingface-hub>=1.0.0)
- `backend/evaluation/llm_grounding.py` — FOUND (contains providers_to_run, Skipping, grounding_provider_comparison, choices with huggingface+all)
- Commit deb3fcc — FOUND
- Commit 3c2747c — FOUND
- Commit fee20c2 — FOUND
