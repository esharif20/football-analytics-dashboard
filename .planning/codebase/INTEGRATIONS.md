---
focus: integrations
source: manual-draft
---

# Integrations

- **Database:** MySQL 8.0 via docker-compose (`localhost:3307`, user `root`, pass `football123`).
- **Tunnels:** ngrok exposes the FastAPI server so the remote worker can poll/POST; worker expects `DASHBOARD_URL`/API base URL.
- **External APIs:** optional `OPENAI_API_KEY` for tactical commentary; optional `ROBOFLOW_API_KEY` for pitch/camera utilities.
- **Storage:** local `./uploads` for videos + outputs; worker downloads via API and re-uploads annotated assets/results.
- **Auth/ownership:** `OWNER_OPEN_ID` identifies local owner for access checks (see env). `LOCAL_DEV_MODE` toggles relaxed flows.
- **GPU host:** RunPod or similar GPU VM runs the worker; first start downloads large model weights (YOLOv8, trackers).
