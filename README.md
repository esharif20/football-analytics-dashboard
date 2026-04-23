# Football Analytics Dashboard

![CI](https://github.com/esharif20/football-analytics-dashboard/actions/workflows/ci.yml/badge.svg)

Full-stack sports analytics platform — upload football match footage, get real-time player tracking, team classification, heatmaps, pass networks, and AI tactical commentary.

Built with **React 19 + FastAPI + PostgreSQL** and a **Python CV pipeline** running on a cloud GPU.

---

## For Assessors — Where to Start

This repository accompanies a CS310 dissertation. If you are reviewing the code without running it, the fastest tour is:

| You want to read… | Look at |
|---|---|
| The thesis itself | submitted as `project.pdf` on Tabula (separate from this repo) |
| Dissertation chapter sources & findings | [`dissertation/`](dissertation/) — markdown chapters, evaluation findings, figures referenced in the PDF |
| The CV pipeline (detection → tracking → analytics) | [`backend/pipeline/src/`](backend/pipeline/src/) — entrypoint `worker.py`, modules in `analytics/`, `trackers/`, `team_classification/` |
| Computer-vision evaluation (probing, ablations, grounding) | [`backend/evaluation/`](backend/evaluation/) — 40+ standalone scripts, one per dissertation experiment (e.g. `linear_probing.py`, `prompt_stability.py`, `vlm_comparison.py`, `tactical_validation.py`) |
| Web application (FastAPI) | [`backend/api/`](backend/api/) — `main.py`, routers in `routers/`, ORM models in `models.py` |
| Web application (React) | [`frontend/src/`](frontend/src/) — pages in `pages/`, REST client `lib/api-local.ts` |
| Architecture overview | [`ARCHITECTURE.md`](ARCHITECTURE.md) — module-level system design |
| Continuous integration | [`.github/workflows/ci.yml`](.github/workflows/ci.yml) — lint, format, type-check, unit tests, build |

### Verifying the build without running the full stack

The fastest signal is the **CI status badge above** — every commit on `main` runs lint, type-check, unit tests, and a production build for both backend and frontend. To reproduce locally without provisioning Postgres or a GPU:

```bash
# Frontend (React 19 + Vite 6)
cd frontend && pnpm install && pnpm lint && pnpm check && pnpm test && pnpm build

# Backend (FastAPI + ruff + pytest)
cd backend && pip install -r api/requirements.txt
ruff check api/ && ruff format --check api/
LOCAL_DEV_MODE=true JWT_SECRET=test \
  DATABASE_URL=postgresql+asyncpg://skip:skip@skip/skip \
  pytest                          # 40 unit tests, no DB or GPU required
```

Running the full application end-to-end (upload a video, see analytics) requires PostgreSQL and a GPU — see the [Quick Start](#quick-start) and [GPU Worker](#gpu-worker-runpod) sections below.

---

## System Architecture

```mermaid
graph TD
    Browser["Browser<br/><i>React 19 + Vite 6 · localhost:5173</i>"]

    Browser -->|"upload video · view results"| API
    API -.->|"WebSocket · live progress"| Browser

    API["FastAPI Server<br/><i>REST API · localhost:8000</i>"]
    API --> DB["PostgreSQL<br/><i>localhost:5432</i>"]
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

1. Upload a video in the browser
2. FastAPI saves it to `./uploads/` and writes a `pending` analysis row in PostgreSQL
3. The worker (on RunPod) polls `/api/worker/pending` through the ngrok tunnel
4. Worker downloads the video, runs detection + tracking + team classification + analytics
5. Worker uploads annotated video and results back through the tunnel
6. Frontend displays the analysis (pitch visualizations, stats, heatmaps, AI commentary)

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
        TS["TypeScript 5.9"]
        TW["Tailwind CSS 4"]
        RQ["React Query v5"]
        Recharts["Recharts"]
        Shadcn["shadcn/ui"]
        FM["Framer Motion"]
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
        PG["PostgreSQL"]
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
- **Node.js 22+** and **pnpm** — for the frontend
- **PostgreSQL** — any running instance on port 5432 (or update `DATABASE_URL`)
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
# Edit .env: set DATABASE_URL to your PostgreSQL instance

# 3. Install backend deps
cd backend && pip install -r api/requirements.txt && cd ..

# 4. Install frontend deps
cd frontend && pnpm install && cd ..

# 5. Start FastAPI (terminal 1)
cd backend && uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# 6. Start frontend (terminal 2)
cd frontend && pnpm dev
```

Open **http://localhost:5173** — auto-logged in locally, no auth required.

### Terminal Layout

| Terminal | Command | What it does |
|----------|---------|-------------|
| 1 | `cd backend && uvicorn api.main:app --port 8000 --reload` | FastAPI on :8000 |
| 2 | `cd frontend && pnpm dev` | React frontend on :5173 |
| 3 | `ngrok http 8000` | Public tunnel for the GPU worker |

---

## Testing

### Frontend

```bash
cd frontend
pnpm lint          # ESLint
pnpm format:check  # Prettier
pnpm check         # TypeScript (tsc --noEmit)
pnpm test          # Vitest unit tests
pnpm build         # Production build
```

### Backend

```bash
cd backend
ruff check api/                    # Lint
ruff format --check api/           # Format check
LOCAL_DEV_MODE=true \
  JWT_SECRET=test \
  DATABASE_URL=postgresql+asyncpg://skip:skip@skip/skip \
  pytest                           # 40 unit tests (no DB required)
```

CI runs all of the above on every push to `main` via `.github/workflows/ci.yml`.

### End-to-End

```bash
pnpm dlx playwright test           # Playwright e2e suite under tests/e2e/
```

---

## GPU Worker (RunPod)

The CV pipeline requires a GPU. It runs as a polling worker on a cloud GPU instance.

### Setup

```bash
# SSH into RunPod, then:
git clone https://github.com/esharif20/football-analytics-dashboard.git
cd football-analytics-dashboard/backend/pipeline
pip install -r requirements.txt
pip install umap-learn sentencepiece protobuf
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

### Docker image

```bash
docker build -f docker/Dockerfile.worker -t football-worker .
```

---

## Project Structure

```
football-analytics-dashboard/
├── frontend/                          React 19 + Vite 6 + TypeScript 5.9
│   └── src/
│       ├── pages/                     Home, Upload, Dashboard, Analysis
│       ├── components/                shadcn/ui + custom components
│       ├── lib/api-local.ts           REST client (all API calls)
│       ├── hooks/useWebSocket.ts      WebSocket for live progress
│       └── test/                      Vitest unit tests
│
├── backend/
│   ├── api/                           FastAPI backend
│   │   ├── main.py                    App entry, routers, middleware
│   │   ├── models.py                  SQLAlchemy models (7 tables)
│   │   ├── schemas.py                 Pydantic request/response models
│   │   ├── database.py                Async engine + session
│   │   ├── storage.py                 File storage + H.264 re-encoding
│   │   ├── ws.py                      WebSocket for analysis progress
│   │   ├── services/                  LLM providers (Gemini, OpenAI, Anthropic, Groq, HuggingFace)
│   │   ├── routers/                   API route handlers
│   │   └── tests/                     pytest unit tests (40 tests)
│   │
│   ├── pipeline/                      Python CV pipeline
│   │   ├── worker.py                  GPU worker (polls API for jobs)
│   │   ├── requirements.txt
│   │   └── src/
│   │       ├── trackers/              ByteTrack + role stabiliser
│   │       ├── team_classification/   SigLIP + UMAP + KMeans
│   │       ├── analytics/             Possession, passes, tactical metrics
│   │       └── utils/                 Video I/O, drawing, homography
│   │
│   ├── evaluation/                    40+ standalone dissertation experiments:
│   │                                  linear/dense probing, ablations,
│   │                                  prompt stability, VLM comparison,
│   │                                  tactical validation, db grounding
│   │
│   └── event_detector/                Match-event detection module
│
├── dissertation/                      Dissertation source content
│   ├── chapter_reasoning_evaluation.md
│   ├── findings/                      Per-experiment writeups (DB findings,
│   │                                  linear probing, visual time-series,
│   │                                  examples appendix, etc.)
│   ├── prompts_and_outputs.md         Prompts used + raw model outputs
│   └── figures/                       PDF + PNG figures referenced in chapters
│
├── notebooks/                         Jupyter notebooks
│   ├── Full Pipeline.ipynb            End-to-end pipeline walkthrough
│   └── EfficientNet BCE.ipynb         Event-detector training experiment
│
├── tests/                             Playwright end-to-end tests
│   └── e2e/                           7 spec files covering upload → analysis
│
├── docker/Dockerfile.worker           Worker container image
├── scripts/                           Utility scripts (sync-to-pod, e2e helpers)
├── supabase/                          Supabase config + SQL migrations
├── docs/                              Architecture diagrams (.mmd, .d2, .png)
└── .github/workflows/ci.yml           CI pipeline (lint, test, build)
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
| `/api/commentary/{id}` | GET / POST | Commentary list or generate (streaming) |
| `/api/chat/{id}` | POST | AI chat about an analysis |

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

Real-time progress updates during pipeline processing. Messages:
- `{"type": "progress", "stage": "...", "percent": 42}`
- `{"type": "complete", "annotatedVideoUrl": "..."}`
- `{"type": "error", "message": "..."}`

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

**ORM**: SQLAlchemy async (PostgreSQL via asyncpg)

| Table | Purpose |
|-------|---------|
| `users` | User accounts (auto-created in local dev) |
| `videos` | Uploaded video metadata |
| `analyses` | Pipeline jobs — status, progress, output URLs |
| `events` | Detected match events (passes, shots, etc.) |
| `tracks` | Per-frame tracking data (positions, ball, formations) |
| `statistics` | Aggregated stats (possession, heatmaps, pass networks) |
| `commentary` | AI-generated tactical analysis |

Tables auto-create on first FastAPI startup (`CREATE TABLE IF NOT EXISTS`).

---

## Environment Variables

### `.env` (project root)

```bash
LOCAL_DEV_MODE=true
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/football_dashboard
LOCAL_STORAGE_DIR=./uploads
OWNER_OPEN_ID=local-dev-user

# AI Commentary — at least one required for tactical analysis
LLM_PROVIDER=gemini          # gemini, openai, claude, groq, huggingface, or stub
GEMINI_API_KEY=              # https://aistudio.google.com/app/apikey
OPENAI_API_KEY=              # https://platform.openai.com/api-keys
ANTHROPIC_API_KEY=           # https://console.anthropic.com/settings/keys
GROQ_API_KEY=                # https://console.groq.com/keys
HUGGINGFACE_API_KEY=         # https://huggingface.co/settings/tokens
LLM_MODEL=                   # optional model override

# Optional
ROBOFLOW_API_KEY=            # pitch detection (ultralytics backend used by default)
```

### Worker (on RunPod)

```bash
DASHBOARD_URL=https://xxx.ngrok-free.dev
PIPELINE_SUBPROCESS=1        # non-TTY safe output
POLL_INTERVAL=5              # seconds
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
| DB won't connect | Verify `DATABASE_URL` in `.env` and that PostgreSQL is running |
| Worker can't reach dashboard | Check ngrok is running; `DASHBOARD_URL` must match exactly |
| Video won't play in browser | Install `ffmpeg` locally — `brew install ffmpeg` |
| Port 8000 in use | `lsof -i :8000` then `kill <PID>` |
| Pipeline module errors | `pip install -r requirements.txt && pip install umap-learn sentencepiece protobuf` |
| Models fail to download | Place `.pt` files manually in `backend/pipeline/models/` |
| Local worker stealing jobs | Kill it: `ps aux \| grep worker.py` → `kill <PID>` |

---

## License

MIT
