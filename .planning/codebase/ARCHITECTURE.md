---
focus: architecture
source: manual-draft
---

# Architecture

- Vite-served React SPA (:5173) calls FastAPI REST endpoints and subscribes to a WebSocket for job progress.
- FastAPI backend (:8000) owns upload endpoints, analysis/commentary/stats routes, and worker coordination handlers.
- CV worker polls `/api/worker/pending`, downloads source media, runs detection/tracking/analytics, then POSTs metrics + annotated assets back.
- Persistence: FastAPI writes metadata/results to MySQL and files to `./uploads`; frontend queries aggregated data for dashboards/visuals.
- Control flow: Browser upload → API stores file + DB row → worker processes → API persists outputs → frontend visualizes (heatmaps, events, charts).
