# Football Analytics Dashboard

Full-stack sports analytics platform — upload football match footage, get real-time player tracking, team classification, heatmaps, pass networks, and AI tactical commentary.

Built with **React + FastAPI + MySQL** and a **Python CV pipeline** running on a cloud GPU.

---

## System Architecture

```mermaid
graph TD
    Browser["Browser<br/><i>React + Vite · localhost:5173</i>"]

    Browser -->|"upload video · view results"| API
    API -.->|"WebSocket · live progress"| Browser

    API["FastAPI Server<br/><i>REST API · localhost:8000</i>"]
    API --> DB["MySQL 8.0<br/><i>localhost:3307</i>"]
    API --> FS["File Storage<br/><i>./uploads/</i>"]

    API <-->|"ngrok tunnel"| Worker

    Worker["worker.py<br/><i>Polls /api/worker/pending</i>"]
    Worker --> Detect["YOLOv8 Detection"]
    Detect --> Track["ByteTrack Tracking"]
    Track --> Classify["SigLIP Team Classification"]
    Classify --> Map["Homography Mapping"]
    Map --> Analyse["Analytics Engine"]
    Analyse -->|"POST results + video"| API

    style Browser fill:#1f2937,stroke:#3b82f6,color:#f0f6fc
    style API fill:#1f2937,stroke:#10b981,color:#f0f6fc
    style DB fill:#1f2937,stroke:#8b5cf6,color:#f0f6fc
    style FS fill:#1f2937,stroke:#8b5cf6,color:#f0f6fc
    style Worker fill:#1f2937,stroke:#f59e0b,color:#f0f6fc
    style Detect fill:#1f2937,stroke:#f59e0b,color:#f0f6fc
    style Track fill:#1f2937,stroke:#f59e0b,color:#f0f6fc
    style Classify fill:#1f2937,stroke:#f59e0b,color:#f0f6fc
    style Map fill:#1f2937,stroke:#f59e0b,color:#f0f6fc
    style Analyse fill:#1f2937,stroke:#f59e0b,color:#f0f6fc
```

### The Flow

1. You upload a video in the browser
2. FastAPI saves it to `./uploads/` and writes a `pending` analysis row in MySQL
3. The worker (on RunPod) polls `/api/worker/pending` through the ngrok tunnel
4. Worker downloads the video, runs detection + tracking + team classification + analytics
5. Worker uploads annotated video and results back through the tunnel
6. Frontend displays the analysis (pitch visualizations, stats, heatmaps)

---

## Pipeline Flow

```mermaid
graph TD
    Input["Video File"] --> Frames["Frame Loading"]
    Frames --> Detection{"Detection Stage"}

    Detection --> Players["Player Detection<br/><i>YOLOv8x fine-tuned</i>"]
    Detection --> Ball["Ball Detection<br/><i>YOLOv8 + SAHI Slicer</i>"]
    Detection --> Pitch["Pitch Detection<br/><i>YOLOv8 Keypoints</i>"]

    Players --> Tracking["People Tracking<br/><i>ByteTrack</i>"]
    Tracking --> Stabiliser["Role Stabiliser<br/><i>Lock GK/Ref labels</i>"]
    Stabiliser --> Teams["Team Classification<br/><i>SigLIP + UMAP + KMeans</i>"]

    Ball --> BallTrack["Ball Tracking<br/><i>8-Stage Filter + Kalman</i>"]

    Pitch --> Homography["View Transformer<br/><i>Homography</i>"]
    Homography --> Radar["Radar View<br/><i>2D Pitch Diagram</i>"]

    Teams --> Annotation["Annotation"]
    BallTrack --> Annotation
    Annotation --> OutputVideo["Output Video"]

    Teams --> AnalyticsEngine["Analytics Engine"]
    BallTrack --> AnalyticsEngine
    Homography --> AnalyticsEngine

    AnalyticsEngine --> Stats["Match Statistics<br/><i>Possession, passes, shots,<br/>distance, speed, heatmaps</i>"]

    style Input fill:#1f2937,stroke:#10b981,color:#f0f6fc
    style Detection fill:#1f2937,stroke:#f59e0b,color:#f0f6fc
    style Players fill:#1f2937,stroke:#10b981,color:#f0f6fc
    style Ball fill:#1f2937,stroke:#10b981,color:#f0f6fc
    style Pitch fill:#1f2937,stroke:#10b981,color:#f0f6fc
    style Tracking fill:#1f2937,stroke:#10b981,color:#f0f6fc
    style Stabiliser fill:#1f2937,stroke:#10b981,color:#f0f6fc
    style Teams fill:#1f2937,stroke:#10b981,color:#f0f6fc
    style BallTrack fill:#1f2937,stroke:#10b981,color:#f0f6fc
    style Homography fill:#1f2937,stroke:#10b981,color:#f0f6fc
    style Radar fill:#1f2937,stroke:#10b981,color:#f0f6fc
    style Annotation fill:#1f2937,stroke:#10b981,color:#f0f6fc
    style OutputVideo fill:#1f2937,stroke:#10b981,color:#f0f6fc
    style AnalyticsEngine fill:#1f2937,stroke:#10b981,color:#f0f6fc
    style Stats fill:#1f2937,stroke:#10b981,color:#f0f6fc
    style Frames fill:#1f2937,stroke:#10b981,color:#f0f6fc
```

### Core Components

| Stage | Model / Method | Performance |
|-------|---------------|-------------|
| Object Detection | YOLOv8x fine-tuned | Player 99.4%, GK 94.2%, Referee 98.2%, Ball 92.5% mAP@50 |
| Multi-Object Tracking | ByteTrack | Two-stage association + TrackStabiliser (majority voting) |
| Team Classification | SigLIP + UMAP + KMeans | 768-dim embeddings, k=2 clustering |
| Coordinate Transform | Homography | Pitch keypoints to real-world metres |
| Ball Tracking | 8-stage filter + Kalman | Interpolation across occlusions |
| Analytics | Custom engine | Possession, passes, shots, distance, speed, heatmaps |

---

## Tech Stack

```mermaid
graph LR
    subgraph Frontend
        React["React 19"]
        Vite2["Vite 6"]
        TS["TypeScript"]
        TW["Tailwind CSS 4"]
        RQ["React Query"]
        Recharts["Recharts"]
        Shadcn["shadcn/ui"]
    end

    subgraph Backend
        FA["FastAPI"]
        SA["SQLAlchemy<br/><i>async</i>"]
        Pydantic["Pydantic"]
        Uvicorn["Uvicorn"]
    end

    subgraph Pipeline2["Pipeline"]
        PyTorch["PyTorch"]
        YOLO2["Ultralytics<br/>YOLOv8"]
        CV2["OpenCV"]
        SV["Supervision"]
    end

    subgraph Infra
        MySQL2["MySQL 8.0"]
        Docker["Docker"]
        Ngrok["ngrok"]
        RunPod["RunPod"]
    end

    style Frontend fill:#161b22,stroke:#3b82f6,color:#f0f6fc
    style Backend fill:#161b22,stroke:#10b981,color:#f0f6fc
    style Pipeline2 fill:#161b22,stroke:#f59e0b,color:#f0f6fc
    style Infra fill:#161b22,stroke:#8b5cf6,color:#f0f6fc
```

---

## Prerequisites

- **Python 3.11+** — for the FastAPI backend
- **Node.js 18+** and **pnpm** — for the frontend
- **Docker Desktop** — for running MySQL
- **ngrok** — `brew install ngrok` + [free account](https://ngrok.com)
- **RunPod account** (or any cloud GPU) — for the CV pipeline worker

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/esharif20/football-analytics-dashboard.git
cd football-analytics-dashboard

# 2. Copy environment config
cp env.example .env

# 3. Start MySQL
docker compose up db -d

# 4. Install Python backend deps
cd backend && pip install -r api/requirements.txt && cd ..

# 5. Install frontend deps
cd frontend && pnpm install && cd ..

# 6. Start FastAPI (terminal 1)
cd backend && uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# 7. Start frontend (terminal 2)
cd frontend && pnpm dev
```

Open **http://localhost:5173** — auto-logged in, no auth needed.

### Terminal Layout

| Terminal | Command | What it does |
|----------|---------|-------------|
| 1 | `docker compose up db -d` | MySQL (run once, stays up) |
| 2 | `cd backend && uvicorn api.main:app --port 8000 --reload` | FastAPI on :8000 |
| 3 | `cd frontend && pnpm dev` | React frontend on :5173 |
| 4 | `ngrok http 8000` | Public tunnel for the GPU worker |

---

## GPU Worker (RunPod)

The CV pipeline requires a GPU. It runs as a polling worker on a cloud GPU instance.

### Setup

```bash
# SSH into RunPod, then:
git clone https://github.com/esharif20/football-analytics-dashboard.git
cd football-analytics-dashboard/backend/pipeline
pip install -r requirements.txt
pip install requests
```

### Run

```bash
export DASHBOARD_URL=https://<your-ngrok-url>.ngrok-free.dev
python worker.py
```

First run downloads ~400MB of ML models from CDN.

### Background mode

```bash
nohup python worker.py > worker.log 2>&1 &
tail -f worker.log      # watch
pkill -f worker.py      # stop
```

---

## Project Structure

```
football-analytics-dashboard/
├── frontend/                          React 19 + Vite + TypeScript
│   └── src/
│       ├── pages/                     Home, Upload, Dashboard, Analysis
│       ├── components/                shadcn/ui components
│       ├── lib/api-local.ts           REST client (all API calls)
│       ├── hooks/useWebSocket.ts      WebSocket for live progress
│       └── shared/                    Shared types & constants
│
├── backend/
│   ├── api/                           FastAPI backend
│   │   ├── main.py                    App entry, routers, middleware
│   │   ├── models.py                  SQLAlchemy models (7 tables)
│   │   ├── schemas.py                 Pydantic request/response models
│   │   ├── database.py                Async engine + session
│   │   ├── storage.py                 File storage + H.264 re-encoding
│   │   ├── ws.py                      WebSocket for analysis progress
│   │   └── routers/                   API route handlers
│   │
│   ├── pipeline/                      Python CV pipeline
│   │   ├── worker.py                  GPU worker (polls API)
│   │   ├── requirements.txt
│   │   └── src/                       Detection, tracking, analytics
│   │
│   └── uploads/                       Local file storage
│
├── docker-compose.yml                 MySQL container
├── Dockerfile.worker                  Worker Docker image
└── .github/workflows/ci.yml          CI pipeline
```

---

## API Reference

### REST Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/auth/me` | GET | Current user session |
| `/api/videos` | GET | List videos |
| `/api/videos/{id}` | GET / DELETE | Get or delete video |
| `/api/upload/video` | POST | Upload video (multipart) |
| `/api/analyses` | GET / POST | List or create analyses |
| `/api/analyses/{id}` | GET | Get analysis details |
| `/api/analyses/modes` | GET | Available pipeline modes |
| `/api/analyses/stages` | GET | Processing stage definitions |
| `/api/events/{id}` | GET | Events for an analysis |
| `/api/tracks/{id}` | GET | Tracks for an analysis |
| `/api/statistics/{id}` | GET | Statistics for an analysis |
| `/api/commentary/{id}` | GET / POST | Commentary list or generate |

### Worker Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/worker/pending` | GET | Poll for pending analyses |
| `/api/worker/analysis/{id}/status` | POST | Push progress updates |
| `/api/worker/analysis/{id}/complete` | POST | Submit final results |
| `/api/worker/upload-video` | POST | Upload processed video |

### WebSocket

```
ws://localhost:8000/ws/{analysis_id}
```

Real-time progress updates during pipeline processing.

---

## Pipeline Modes

| Mode | Value | Output |
|------|-------|--------|
| Full Analysis | `all` | Annotated video + radar + analytics |
| Radar View | `radar` | 2D pitch visualization |
| Team Classification | `team` | Video with team colors |
| Player Tracking | `track` | Tracking data JSON |
| Player Detection | `players` | Annotated video |
| Ball Detection | `ball` | Ball trajectory data |
| Pitch Detection | `pitch` | Homography matrix |

---

## Database

**ORM**: SQLAlchemy async (MySQL via aiomysql)

| Table | Purpose |
|-------|---------|
| `users` | User accounts (auto-created in local dev) |
| `videos` | Uploaded video metadata |
| `analyses` | Pipeline jobs — status, progress, output URLs |
| `events` | Detected match events (passes, shots, etc.) |
| `tracks` | Per-frame tracking data (positions, ball, formations) |
| `statistics` | Aggregated stats (possession, heatmaps, pass networks) |
| `commentary` | AI-generated tactical analysis |

Tables auto-create on first FastAPI startup.

| | |
|---|---|
| Host | `localhost:3307` |
| User | `root` |
| Password | `football123` |
| Database | `football_dashboard` |

---

## Environment Variables

### `.env` (project root)

```bash
LOCAL_DEV_MODE=true
DATABASE_URL=mysql://root:football123@localhost:3307/football_dashboard
LOCAL_STORAGE_DIR=./uploads
OWNER_OPEN_ID=local-dev-user

# Optional
OPENAI_API_KEY=          # AI commentary
ROBOFLOW_API_KEY=        # pitch detection
```

### Worker (on RunPod)

```bash
DASHBOARD_URL=https://xxx.ngrok-free.dev
ROBOFLOW_API_KEY=<key>   # optional
POLL_INTERVAL=5           # seconds
```

---

## GPU Options

| Platform | $/hr | Best for |
|----------|------|----------|
| **RunPod** | ~$0.20 (RTX 3090) | Best value, Docker support |
| **Google Colab Pro** | ~$10/mo flat | Easiest setup |
| **Vast.ai** | ~$0.10-0.30 | Cheapest, community GPUs |

**Processing times (30s clip):**

| GPU | Time |
|-----|------|
| RTX 3090 / A100 | ~1-2 min |
| Apple M1/M2 (MPS) | ~5-7 min |
| CPU only | ~10+ min |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| MySQL won't connect | `docker compose up db -d && docker ps` |
| Worker can't reach dashboard | Check ngrok is running, `DASHBOARD_URL` matches exactly |
| Video won't play in browser | Install `ffmpeg` locally — `brew install ffmpeg` |
| Port 8000 in use | `lsof -i :8000` then `kill <PID>` |
| Pipeline module errors | `pip install -r requirements.txt && pip install requests` |
| Models fail to download | Download `.pt` files manually into `backend/pipeline/models/` |

### Full Reset

```bash
docker compose down -v && rm -rf backend/uploads/*
docker compose up db -d
# Restart FastAPI — tables auto-create
```

---

## License

MIT
