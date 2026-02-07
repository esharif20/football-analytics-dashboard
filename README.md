# Football Analytics Dashboard

Full-stack sports analytics platform — upload football match footage, get real-time player tracking, team classification, heatmaps, pass networks, and AI tactical commentary.

Built with React + FastAPI + MySQL + a Python CV pipeline running on a cloud GPU.

---

## How It Works

```
    LOCAL MACHINE                                 CLOUD GPU (RunPod)
   ──────────────                                ──────────────────

   ┌────────────────────────┐
   │    Docker MySQL 8.0    │
   │    localhost:3307       │
   └───────────┬────────────┘
               │
               ▼
   ┌────────────────────────┐        ngrok tunnel        ┌──────────────────────────┐
   │  FastAPI    :8000      │ ◄──────────────────────── │  worker.py               │
   │  Vite Dev   :5173      │  https://xxx.ngrok.dev     │                          │
   │                         │ ────────────────────────► │  Polls /api/worker/      │
   │  ┌───────┐ ┌─────────┐ │                            │  pending every 5s        │
   │  │React  │ │ FastAPI  │ │   1. GET  pending jobs     │                          │
   │  │Front- │ │ REST API │ │   2. Downloads video       │  ┌──────────────────┐   │
   │  │end    │ │ + WS     │ │   3. POST progress         │  │ CV Pipeline      │   │
   │  │       │ │         │ │   4. POST results          │  │                  │   │
   │  └───────┘ └─────────┘ │                            │  │ YOLOv8 detect    │   │
   │                         │                            │  │ ByteTrack track  │   │
   │  Uploads → ./uploads/   │                            │  │ SigLIP teams     │   │
   └────────────────────────┘                            │  │ Homography map   │   │
                                                          │  │ Analytics stats  │   │
         Browser                                          │  └──────────────────┘   │
     http://localhost:5173                                └──────────────────────────┘
```

**The flow:**

1. You upload a video in the browser
2. FastAPI saves it to `./uploads/` and writes a `pending` analysis row in MySQL
3. The worker (on RunPod) polls `/api/worker/pending` through the ngrok tunnel
4. Worker downloads the video, runs detection + tracking + team classification + analytics
5. Worker uploads annotated video and results back through the tunnel
6. Frontend displays the analysis (pitch visualizations, stats, heatmaps)

---

## Tech Stack

| | |
|---|---|
| **Frontend** | React 19, Vite 6, TypeScript, Tailwind CSS 4, Recharts, shadcn/ui, Wouter |
| **Backend** | FastAPI, SQLAlchemy (async), Pydantic, Uvicorn |
| **Database** | MySQL 8.0 via Docker (port 3307) |
| **Pipeline** | Python 3.11+, PyTorch, YOLOv8 (Ultralytics), ByteTrack, SigLIP, OpenCV |
| **Infra** | Docker, ngrok, RunPod |

---

## Prerequisites

- **Python 3.11+** — for the FastAPI backend
- **Node.js 18+** and **pnpm** — for the frontend
- **Docker Desktop** — for running MySQL
- **ngrok** — `brew install ngrok` + [free account](https://ngrok.com)
- **RunPod account** (or any cloud GPU) — for the CV pipeline worker

---

## Local Development

### Quick start

```bash
# 1. Clone
git clone https://github.com/esharif20/football-analytics-dashboard.git
cd football-analytics-dashboard

# 2. Copy environment config
cp env.example .env

# 3. Start MySQL (Docker)
docker compose up db -d

# 4. Install Python backend dependencies
cd backend && pip install -r api/requirements.txt && cd ..

# 5. Install frontend dependencies
cd frontend && pnpm install && cd ..

# 6. Start FastAPI backend (terminal 1)
cd backend && uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# 7. Start frontend dev server (terminal 2)
cd frontend && pnpm dev
```

Open **http://localhost:5173** — you're auto-logged in as "Local Developer" (no auth needed).

### What runs where

| Process | Port | What it does |
|---------|------|-------------|
| FastAPI (uvicorn) | 8000 | REST API, WebSocket, static file serving |
| Vite dev server | 5173 | React frontend with hot-reload, proxies `/api` + `/uploads` + `/ws` to :8000 |
| MySQL (Docker) | 3307 | Database |

### MySQL credentials

| | |
|---|---|
| Host | `localhost` |
| Port | `3307` |
| User | `root` |
| Password | `football123` |
| Database | `football_dashboard` |

Data persists in a Docker volume (`mysql_data`). To connect manually:

```bash
docker exec -it football-db mysql -u root -pfootball123 football_dashboard
```

### Start ngrok

In a separate terminal:

```bash
ngrok http 8000
```

Copy the HTTPS forwarding URL — the worker needs this.

> Free-tier ngrok URLs change on every restart. Update the worker's `DASHBOARD_URL` accordingly.

### Terminal summary

| Terminal | Command | What it does |
|----------|---------|-------------|
| 1 | `docker compose up db -d` | Starts MySQL (run once, stays up) |
| 2 | `cd backend && uvicorn api.main:app --port 8000 --reload` | FastAPI backend on :8000 |
| 3 | `cd frontend && pnpm dev` | React frontend on :5173 |
| 4 | `ngrok http 8000` | Public tunnel for the worker |

---

## GPU Worker (RunPod)

The CV pipeline requires a GPU. It runs as a background service on a cloud GPU instance.

### Setup on RunPod

```bash
# SSH into your RunPod instance, then:
git clone https://github.com/esharif20/football-analytics-dashboard.git
cd football-analytics-dashboard/backend/pipeline
pip install -r requirements.txt
pip install requests
```

Or one-liner:
```bash
curl -sSL https://raw.githubusercontent.com/esharif20/football-analytics-dashboard/main/scripts/setup-cloud-gpu.sh | bash
```

### Run the worker

```bash
cd /workspace/football-analytics-dashboard/backend/pipeline
export DASHBOARD_URL=https://<your-ngrok-url>.ngrok-free.dev
export ROBOFLOW_API_KEY=<your-key>    # optional
python worker.py
```

On first run it downloads ~400MB of ML models (`player_detection.pt`, `ball_detection.pt`, `pitch_detection.pt`) from CDN.

### Background mode

```bash
nohup python worker.py > worker.log 2>&1 &
tail -f worker.log      # watch
pkill -f worker.py      # stop
```

---

## Environment Variables

### `.env` (project root)

```bash
# Required for local dev
LOCAL_DEV_MODE=true
DATABASE_URL=mysql://root:football123@localhost:3307/football_dashboard
LOCAL_STORAGE_DIR=./uploads
OWNER_OPEN_ID=local-dev-user

# Optional
OPENAI_API_KEY=                  # AI commentary
ROBOFLOW_API_KEY=                # pitch detection
```

### Worker env vars (on RunPod)

```bash
DASHBOARD_URL=https://xxx.ngrok-free.dev   # your ngrok URL
ROBOFLOW_API_KEY=<key>                     # optional
POLL_INTERVAL=5                            # seconds between polls
```

---

## Database

**ORM**: SQLAlchemy (async, MySQL via aiomysql) — models in `backend/api/models.py`.

| Table | Purpose |
|-------|---------|
| `users` | User accounts (auto-created in local mode) |
| `videos` | Uploaded video metadata |
| `analyses` | Pipeline jobs — status, progress, output URLs |
| `events` | Detected match events (passes, shots, etc.) |
| `tracks` | Per-frame tracking data (player positions, ball, formations) |
| `statistics` | Aggregated stats (possession, pass accuracy, heatmaps) |
| `commentary` | AI-generated tactical analysis |

Tables are auto-created by SQLAlchemy on first startup if they don't exist.

### Reset database

```bash
docker compose down -v
docker compose up db -d
# Tables are auto-created on next FastAPI startup
```

---

## Project Structure

```
football-analytics-dashboard/
│
├── frontend/                         # React 19 + Vite + TypeScript
│   └── src/
│       ├── pages/                    # Home, Upload, Dashboard, Analysis
│       ├── components/               # shadcn/ui components
│       ├── lib/api-local.ts          # REST client (all API calls)
│       ├── hooks/useWebSocket.ts     # WebSocket for live progress
│       └── shared/                   # Shared types & constants
│
├── backend/
│   ├── api/                          # FastAPI backend
│   │   ├── main.py                   # App entry point, routers, middleware
│   │   ├── models.py                 # SQLAlchemy models (7 tables)
│   │   ├── schemas.py                # Pydantic request/response models
│   │   ├── database.py               # Async engine + session
│   │   ├── deps.py                   # Dependencies (get_db, get_current_user)
│   │   ├── auth.py                   # Auto-login middleware (dev mode)
│   │   ├── storage.py                # Local file storage + H.264 re-encoding
│   │   ├── ws.py                     # WebSocket for analysis progress
│   │   └── routers/
│   │       ├── videos.py             # Video CRUD + upload
│   │       ├── analyses.py           # Analysis CRUD + modes/stages/eta
│   │       ├── events.py             # Event queries
│   │       ├── tracks.py             # Track queries
│   │       ├── stats.py              # Statistics queries
│   │       ├── commentary.py         # Commentary queries + generation
│   │       ├── worker.py             # Worker endpoints (poll, status, complete)
│   │       └── system.py             # Health check
│   │
│   ├── pipeline/                     # Python CV pipeline
│   │   ├── worker.py                 # GPU worker (polls API)
│   │   ├── requirements.txt
│   │   └── src/
│   │       ├── main.py               # CLI entry point
│   │       ├── pipeline.py           # Orchestrator
│   │       ├── trackers/             # YOLOv8 + ByteTrack
│   │       ├── team_assigner/        # SigLIP + KMeans
│   │       ├── pitch/                # Homography transform
│   │       └── analytics/            # Stats computation
│   │
│   ├── uploads/                      # Local file storage (gitignored)
│   └── .env                          # Backend environment config
│
├── docker-compose.yml                # MySQL container
├── Dockerfile.worker                 # Worker Docker image
├── .env                              # Root environment config
└── env.example                       # Config template
```

---

## API Endpoints

### REST (frontend <-> backend) — `/api/*`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/user/me` | GET | Current user session |
| `/api/videos` | GET | List videos |
| `/api/videos/{id}` | GET/DELETE | Get or delete video |
| `/api/upload/video` | POST | Upload video (multipart) |
| `/api/analyses` | GET/POST | List or create analyses |
| `/api/analyses/{id}` | GET | Get analysis details |
| `/api/analyses/{id}/status` | PATCH | Update analysis status |
| `/api/analyses/modes` | GET | Available pipeline modes |
| `/api/analyses/stages` | GET | Processing stage definitions |
| `/api/events/{analysisId}` | GET | Events for an analysis |
| `/api/tracks/{analysisId}` | GET | Tracks for an analysis |
| `/api/statistics/{analysisId}` | GET | Statistics for an analysis |
| `/api/commentary/{analysisId}` | GET/POST | Commentary list or generate |

### Worker endpoints — `/api/worker/*`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/worker/pending` | GET | List pending analyses |
| `/api/worker/analysis/{id}/status` | POST | Update progress |
| `/api/worker/analysis/{id}/complete` | POST | Submit results |
| `/api/worker/upload-video` | POST | Upload processed video |

### WebSocket — `/ws/{analysis_id}`

Real-time progress updates during pipeline processing.

---

## Pipeline Modes

| Mode | Value | Output |
|------|-------|--------|
| Full analysis | `all` | Annotated video + radar + analytics JSON |
| Radar view | `radar` | 2D pitch visualization |
| Team classification | `team` | Video with team colors |
| Player tracking | `track` | Tracking data JSON |
| Player detection | `players` | Annotated video |
| Ball detection | `ball` | Ball trajectory data |
| Pitch detection | `pitch` | Homography matrix |

---

## Pipeline Architecture

The CV pipeline has three layers:

**Layer 1 — Perception** (runs on GPU)

| Stage | Model / Method | Detail |
|-------|---------------|--------|
| Object Detection | YOLOv8x fine-tuned | Player 99.4%, GK 94.2%, Referee 98.2%, Ball 92.5% mAP@50 |
| Multi-Object Tracking | ByteTrack | Two-stage association + TrackStabiliser (majority voting) |
| Team Classification | SigLIP + UMAP + KMeans | 768-dim embeddings, k=2 clustering |
| Coordinate Transform | Homography | Pitch keypoints -> real-world metres |
| Data Export | Structured JSON | Per-frame positions, teams, ball |

**Layer 2 — Analytics** (derived from tracking data)
Possession, territorial dominance, distance/speed per player, formation compactness, defensive line height, pressing intensity, passes, shots, pass accuracy.

**Layer 3 — VLM Reasoning** (planned)
Grounded tactical commentary using structured tracking data as context.

---

## API Keys

Both keys are **optional** — the pipeline works fully without them.

| Key | Used for | Get it at |
|-----|----------|-----------|
| `OPENAI_API_KEY` | AI tactical commentary | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| `ROBOFLOW_API_KEY` | Pitch detection fallback | [app.roboflow.com/settings/api](https://app.roboflow.com/settings/api) |

Set them in `.env` or pass as env vars to the worker.

---

## GPU Options

The worker needs a GPU. Rough costs for a BSc-budget setup:

| Platform | $/hr | Best for |
|----------|------|----------|
| **RunPod** (recommended) | ~$0.20 (RTX 3090) | Best value, Docker support |
| **Google Colab Pro** | ~$10/month flat | Easiest if you already have it |
| **Vast.ai** | ~$0.10-0.30 | Cheapest, community GPUs |

**Estimated processing times (30-second clip):**

| GPU | Total |
|-----|-------|
| RTX 3090 / A100 | ~1-2 min |
| Apple M1/M2 (MPS) | ~5-7 min |
| CPU only | ~10+ min |

For local testing on Apple Silicon the pipeline auto-detects MPS — no config needed.

---

## Troubleshooting

**MySQL won't connect** — make sure the container is running:
```bash
docker compose up db -d && docker ps
```

**Worker can't reach dashboard** — check that ngrok is running and `DASHBOARD_URL` matches the forwarding URL exactly. Free-tier URLs change on restart.

**Port 8000 in use** — `lsof -i :8000` then `kill <PID>`.

**Video won't play in browser** — the pipeline outputs mp4v codec which browsers can't play. The FastAPI server auto-re-encodes to H.264 on upload if `ffmpeg` is installed locally (`brew install ffmpeg`).

**Pipeline module errors on RunPod** — `pip install -r requirements.txt && pip install requests`

**Models fail to download** — if CDN URLs become unavailable, download the `.pt` files manually into `backend/pipeline/models/`. The worker skips downloading if files already exist.

**Full reset:**
```bash
docker compose down -v && rm -rf backend/uploads/*
docker compose up db -d
# Restart FastAPI — tables auto-create
```

---

## License

MIT
