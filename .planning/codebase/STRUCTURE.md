---
focus: structure
source: manual-draft
---

# Structure

- frontend/: React app (`src/pages`, `components/ui`, `hooks`, `lib`, `contexts`, `shared`, `const.ts`). Vite + Tailwind config at root.
- backend/api/: FastAPI app (`main.py`, `config.py`, `deps.py`, `routers/`, `services/`, `schemas.py`, `auth.py`). Tests under `api/tests/`.
- backend/pipeline/: `worker.py`, `requirements.txt`, and `src/` modules (analytics for events/kinematics/possession, trackers, radar utils, bbox utils).
- backend/api/routers/: feature routers for analyses, stats, commentary, worker coordination, plus test_support helpers.
- root configs: `docker-compose.yml` (MySQL), `Dockerfile.worker` (CV worker), `package.json`/`pnpm-lock.yaml`, `playwright.config.ts`.
- docs/ + README.md: contain diagrams (Mermaid) and usage instructions.
