---
focus: concerns
source: manual-draft
---

# Concerns

- No test suite; regressions undetected. Add minimal smoke tests.
- Secrets risk: `.env` present; ensure denylist prevents agent reads/commits.
- Worker depends on ngrok URL accuracy; tunnel drift can break processing.
- Model downloads (~400MB) on first run; document caching for GPU worker.
- DB connectivity: MySQL container must be running on :3307; migrations not present (tables auto-create).
- Observability limited: no centralized logging/metrics; pipeline failures could be silent.
- Optional external keys (OpenAI/Roboflow) must be validated—missing keys should not crash API/worker.
