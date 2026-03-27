title: Football Analytics Dashboard
version: v0.1
status: active
milestone: v0.1
---

## Vision

Deliver an end-to-end football analytics platform where analysts can upload match footage, run CV pipelines for detection/tracking/team classification, and review tactical insights in a responsive dashboard.

## Scope

- **Frontend:** React 19 + Vite 6 + TypeScript + Tailwind 4 UI for uploads, playback, overlays, and analytics visuals.
- **Backend:** FastAPI + async SQLAlchemy + MySQL for API, auth, storage orchestration, and analytics delivery.
- **Pipeline:** YOLOv8 + ByteTrack + custom analytics (possession, kinematics, events, radar) running in a Python worker.

## Status

- Milestone: **v0.1** (planning artifacts in place; codebase map generated)
- Next focus: stabilize dev environment and harden pipeline integration.
