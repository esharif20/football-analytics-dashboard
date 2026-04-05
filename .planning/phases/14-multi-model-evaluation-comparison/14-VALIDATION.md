---
phase: 14
slug: multi-model-evaluation-comparison
status: draft
nyquist_compliant: true
wave_0_complete: false
created: 2026-04-03
---

# Phase 14 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | `backend/evaluation/tests/` (created in Plan 01 Task 1) |
| **Quick run command** | `cd backend && python -m pytest evaluation/tests/ -x -q` |
| **Full suite command** | `cd backend && python -m pytest evaluation/tests/ -v` |
| **Estimated runtime** | ~15 seconds (mocked providers) |

---

## Sampling Rate

- **After every task commit:** Run `cd backend && python -m pytest evaluation/tests/ -x -q`
- **After every plan wave:** Run `cd backend && python -m pytest evaluation/tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 14-01-01 | 01 | 1 | EVAL-01 | unit | `cd backend && python -m pytest evaluation/tests/test_hf_provider.py -v` | Created in-task (TDD) | ⬜ pending |
| 14-01-02 | 01 | 1 | EVAL-02 | integration | `cd backend && python -c "from evaluation.llm_grounding import main; import argparse; p = argparse.ArgumentParser(); p.add_argument('--provider', choices=['gemini','openai','huggingface','all','stub']); p.parse_args(['--provider','all']); print('OK')"` | N/A (CLI check) | ⬜ pending |
| 14-02-01 | 02 | 1 | EVAL-03 | unit | `cd backend && python -m pytest evaluation/tests/test_openai_vision.py -v` | Created in-task (TDD) | ⬜ pending |
| 14-02-02 | 02 | 1 | EVAL-03 | integration | `cd backend && python -c "from evaluation.vlm_comparison import VISION_PROVIDERS, main; print(f'Providers: {list(VISION_PROVIDERS.keys())}')"` | N/A (CLI check) | ⬜ pending |
| 14-03-01 | 03 | 2 | EVAL-02 | execution | `test -f eval_output/tracking/tracking_metrics.json && test -f eval_output/tracking/tracking_summary.tex && echo OK` | N/A (output check) | ⬜ pending |
| 14-03-02 | 03 | 2 | EVAL-02 | execution | `test -f eval_output/grounding/grounding_provider_comparison.tex && echo OK` | N/A (output check) | ⬜ pending |
| 14-03-03 | 03 | 2 | EVAL-03 | execution | `test -f eval_output/vlm/vlm_provider_comparison.tex && echo OK` | N/A (output check) | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

Wave 0 test scaffolds are created inline by Plan 01 and Plan 02 via TDD tasks (tests written before implementation). No separate Wave 0 step needed.

- [ ] `backend/evaluation/tests/__init__.py` — created by Plan 01, Task 1
- [ ] `backend/evaluation/tests/conftest.py` — created by Plan 01, Task 1
- [ ] `backend/evaluation/tests/test_hf_provider.py` — created by Plan 01, Task 1 (TDD)
- [ ] `backend/evaluation/tests/test_openai_vision.py` — created by Plan 02, Task 1 (TDD)
- [ ] `pip install huggingface-hub>=1.0.0` — installed by Plan 01, Task 1

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| HF API returns valid completions | EVAL-01 | Requires live HF API key + quota | Run `llm_grounding.py --provider huggingface` with valid API key |
| OpenAI vision returns valid analysis | EVAL-03 | Requires live OpenAI API key | Run `vlm_comparison.py --provider openai` with valid API key |
| LaTeX tables render correctly | EVAL-02 | Visual verification of PDF output | Compile generated .tex files, inspect tables |
| Pipeline data files exist locally | EVAL-02, EVAL-03 | Requires RunPod output download | Confirm analytics.json, tracks.json, video.mp4 paths before Plan 03 |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references (handled inline via TDD)
- [x] No watch-mode flags
- [x] Feedback latency < 15s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
