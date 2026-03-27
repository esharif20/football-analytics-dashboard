---
focus: tech-stack
source: manual-draft
---

# Stack

## Frontend

- React 19 + Vite 6 with TypeScript strict mode.
- Tailwind CSS 4 with shadcn/ui components and `cn()` helper.
- Routing via `wouter`; data fetching with `@tanstack/react-query`.
- Visualization with Recharts and toast notifications via `sonner`.

## Backend API

- FastAPI (Python 3.11+) served by Uvicorn, exposing REST and WebSocket endpoints.
- SQLAlchemy async + aiomysql for MySQL 8.0 access; Pydantic schemas for IO validation.
- Handles uploads, analysis records, commentary/stats routes, and worker coordination endpoints.

## CV Pipeline / Worker

- PyTorch-based computer vision stages using YOLOv8 for detection and ByteTrack for tracking.
- Analytics modules for kinematics, possession, events, and radar visualizations.
- OpenCV/Supervision utilities for annotation; worker polls API for pending jobs and posts results.

## Infrastructure & Tooling

- Docker Compose hosts MySQL on `localhost:3307`; local storage under `./uploads`.
- ngrok tunnel used to expose API to the remote worker; optional GPU host (e.g., RunPod).
- Package managers: pnpm (frontend) and pip (backend/pipeline). No automated tests configured yet.
