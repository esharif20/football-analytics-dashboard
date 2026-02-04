# Football Analysis Dashboard

A full-stack application for analyzing football match footage using computer vision and AI. Upload tactical wide-shot videos and get real-time player tracking, team classification, heatmaps, pass networks, and AI-generated tactical commentary.

---

## System Architecture

![System Architecture](https://files.manuscdn.com/user_upload_by_module/session_file/310519663334363677/QiweQNeftGgWJTaS.png)

### Architecture Overview

The system consists of three main layers that work together to process football footage:

**Frontend (React + Vite)** handles the user interface, video uploads, and real-time visualizations. It communicates with the backend via tRPC for type-safe API calls and WebSocket for live progress updates during video processing.

**Backend (Node.js + tRPC)** serves as the API gateway, managing authentication, database operations, and file storage. It orchestrates communication between the frontend and the CV pipeline, storing analysis results and serving them to the dashboard.

**CV Pipeline (Python + FastAPI)** performs the heavy lifting of computer vision tasks. It runs as a separate service with a background worker that processes videos through multiple stages: detection, tracking, team classification, and analytics generation.

### Data Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Upload    │────▶│   S3/Local  │────▶│   Worker    │
│   Video     │     │   Storage   │     │   Queue     │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                    ┌──────────────────────────┼──────────────────────────┐
                    ▼                          ▼                          ▼
             ┌─────────────┐           ┌─────────────┐           ┌─────────────┐
             │   Player    │           │    Ball     │           │   Pitch     │
             │  Detection  │           │  Detection  │           │  Keypoints  │
             │  (YOLOv8)   │           │   (SAHI)    │           │  (Custom)   │
             └──────┬──────┘           └──────┬──────┘           └──────┬──────┘
                    │                         │                         │
                    ▼                         ▼                         ▼
             ┌─────────────┐           ┌─────────────┐           ┌─────────────┐
             │  ByteTrack  │           │   Ball      │           │ Homography  │
             │ Persistence │           │Interpolation│           │  Transform  │
             └──────┬──────┘           └──────┬──────┘           └──────┬──────┘
                    │                         │                         │
                    └─────────────────────────┼─────────────────────────┘
                                              ▼
                                    ┌─────────────────┐
                                    │ Team Assignment │
                                    │ SigLIP + UMAP   │
                                    │    + KMeans     │
                                    └────────┬────────┘
                                             │
                    ┌────────────────────────┼────────────────────────┐
                    ▼                        ▼                        ▼
             ┌─────────────┐          ┌─────────────┐          ┌─────────────┐
             │  Annotated  │          │   Radar     │          │  Analytics  │
             │   Video     │          │    View     │          │    JSON     │
             └─────────────┘          └─────────────┘          └─────────────┘
```

### Component Details

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Frontend** | React 19, Vite, TailwindCSS, shadcn/ui | Interactive dashboard with real-time updates |
| **API Server** | Node.js, Express, tRPC | Type-safe API, authentication, file management |
| **Database** | MySQL/SQLite, Drizzle ORM | Store videos, analyses, user data |
| **Storage** | S3/Local filesystem | Video files, output artifacts |
| **Pipeline API** | FastAPI, Python 3.11 | REST endpoints for pipeline control |
| **Worker** | Background process | Async video processing |
| **Detection** | YOLOv8, Ultralytics | Player, ball, referee detection |
| **Tracking** | ByteTrack, Supervision | Multi-object tracking with ID persistence |
| **Team Classification** | SigLIP, UMAP, KMeans | Jersey color-based team assignment |
| **Pitch Mapping** | Custom keypoint model | Homography for 2D pitch projection |

---

## Project Structure

```
football-dashboard/
├── frontend/                    # React Dashboard
│   ├── src/
│   │   ├── pages/               # Home, Upload, Dashboard, Analysis
│   │   ├── components/          # UI components (shadcn/ui)
│   │   ├── hooks/               # Custom React hooks (useWebSocket)
│   │   └── lib/                 # Utilities, tRPC client
│   ├── public/                  # Static assets
│   └── package.json
│
├── backend/
│   ├── server/                  # Node.js API
│   │   ├── routers.ts           # tRPC endpoints
│   │   ├── db.ts                # Database queries
│   │   ├── storage.ts           # File storage helpers
│   │   └── _core/               # Auth, context, middleware
│   │
│   ├── pipeline/                # Python CV Pipeline
│   │   ├── api/                 # FastAPI server
│   │   │   └── server.py        # REST API endpoints
│   │   ├── src/
│   │   │   ├── trackers/        # Detection & tracking
│   │   │   ├── team_assigner/   # Team classification
│   │   │   ├── pitch/           # Homography & pitch mapping
│   │   │   └── analytics/       # Statistics computation
│   │   ├── models/              # ML model files (.pt)
│   │   ├── main.py              # CLI entry point
│   │   └── requirements.txt
│   │
│   ├── drizzle/                 # Database schema
│   └── shared/                  # Shared TypeScript types
│
├── docs/                        # Documentation
│   ├── SETUP_GUIDE.md
│   ├── GPU_SETUP.md
│   └── API_KEYS.md
│
├── docker/                      # Docker configuration
│   ├── Dockerfile
│   ├── Dockerfile.worker
│   └── docker-compose.yml
│
├── Makefile                     # Simple commands
├── README.md
└── package.json                 # Root orchestrator
```

---

## Quick Start

### Prerequisites

- **Node.js 18+** - `brew install node`
- **pnpm** - `npm install -g pnpm`
- **Python 3.10+** - `brew install python@3.11`

### One-Command Setup

```bash
make setup    # Install all dependencies
make run      # Start the application
```

### Manual Setup

```bash
# 1. Install frontend dependencies
cd frontend && pnpm install

# 2. Install backend dependencies
cd ../backend && pnpm install

# 3. Set up Python pipeline
cd pipeline
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 4. Add your models
cp /path/to/player_detection.pt models/

# 5. Start the app
cd ../..
make run
```

Open http://localhost:3000 in your browser.

---

## Pipeline Modes

| Mode | Description | Output | Speed |
|------|-------------|--------|-------|
| `all` | Complete analysis pipeline | Annotated video, radar, analytics | Slowest |
| `radar` | 2D pitch visualization only | Radar video, tracking data | Fast |
| `team` | Team classification | Annotated video with team colors | Medium |
| `track` | Object tracking only | Tracking data JSON | Fast |
| `players` | Player detection only | Annotated video | Fastest |
| `ball` | Ball detection + interpolation | Ball trajectory data | Fast |
| `pitch` | Pitch keypoint detection | Homography matrix | Fast |

### Running the Pipeline

```bash
# Via Makefile
make process VIDEO=/path/to/match.mp4

# Via Python directly
cd backend/pipeline
source venv/bin/activate
python main.py --source-video-path /path/to/video.mp4 --mode all
```

---

## GPU Acceleration

### Apple Silicon (M1/M2/M3/M4)

The pipeline automatically uses **MPS (Metal Performance Shaders)**:

```python
# Automatic detection
if torch.backends.mps.is_available():
    device = "mps"  # Uses Apple GPU
```

### NVIDIA GPU (Linux/Windows)

```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Cloud GPU (RunPod/Colab)

See `docs/GPU_SETUP.md` for detailed cloud setup instructions.

---

## API Reference

### tRPC Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `video.upload` | mutation | Upload video file |
| `video.list` | query | List user's videos |
| `analysis.start` | mutation | Start pipeline processing |
| `analysis.status` | query | Get processing status |
| `analysis.results` | query | Get analysis results |

### FastAPI Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/process` | POST | Start video processing |
| `/status/{job_id}` | GET | Get job status |
| `/health` | GET | Health check |

---

## WebSocket Events

Real-time progress updates via WebSocket:

```typescript
// Subscribe to analysis updates
ws.send(JSON.stringify({
  type: 'subscribe',
  analysisId: 123
}));

// Receive progress updates
{
  type: 'progress',
  analysisId: 123,
  progress: 45,
  stage: 'tracking',
  eta: 120 // seconds remaining
}
```

---

## Development

### Dashboard Development

```bash
cd frontend
pnpm dev      # Start dev server with hot reload
pnpm test     # Run tests
pnpm check    # Type check
```

### Pipeline Development

```bash
cd backend/pipeline
source venv/bin/activate
python main.py --video test.mp4 --mode all --verbose
```

### Running Tests

```bash
make test     # Run all tests
```

---

## License

MIT License
