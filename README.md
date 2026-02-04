# Football Analysis Dashboard

A full-stack application for analyzing football match footage using computer vision and AI. Upload tactical wide-shot videos and get real-time player tracking, team classification, heatmaps, pass networks, and AI-generated tactical commentary.

---

## Quick Start (Docker)

**Easiest way - just need Docker installed:**

```bash
# Clone the repository
git clone https://github.com/esharif20/football-analytics-dashboard.git
cd football-analytics-dashboard

# Run (builds and starts everything)
docker compose up
```

Open **http://localhost:8000** in your browser. That's it.

**No authentication required** - just upload a video and start analyzing.

---

## Alternative: Manual Setup

If you don't have Docker, you can run locally:

```bash
make setup   # Install dependencies (one time)
make run     # Start the app
```

Requires: Python 3.9+, Node.js 18+, pnpm

---

## System Architecture

![System Architecture](https://files.manuscdn.com/user_upload_by_module/session_file/310519663334363677/qnWaZAlvezOpVSiR.png)

### Architecture Overview

The system uses a **FastAPI backend** for local development:

```
┌─────────────────────────────────────────────────────────────────┐
│                        Your Browser                              │
│                    http://localhost:8000                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Backend (Python)                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   REST API   │  │  WebSocket   │  │ Static Files │          │
│  │  /api/*      │  │  Real-time   │  │  (React UI)  │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                              │                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    CV Pipeline                            │  │
│  │  Detection → Tracking → Team Assignment → Analytics       │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     SQLite Database                              │
│              backend/data/football.db                            │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Upload    │────▶│   Local     │────▶│   Worker    │
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

---

## Project Structure

```
football-dashboard/
├── frontend/                    # React Dashboard
│   ├── src/
│   │   ├── pages/               # Home, Upload, Dashboard, Analysis
│   │   ├── components/          # UI components (shadcn/ui)
│   │   ├── hooks/               # Custom React hooks (useWebSocket)
│   │   └── lib/                 # Utilities, API client
│   ├── public/                  # Static assets
│   └── package.json
│
├── backend/
│   ├── api/                     # FastAPI Backend (Local Mode)
│   │   ├── main.py              # FastAPI entry point
│   │   ├── routers/             # API endpoints
│   │   │   ├── auth.py          # Authentication
│   │   │   ├── videos.py        # Video management
│   │   │   ├── analysis.py      # Analysis processing
│   │   │   ├── events.py        # Match events
│   │   │   ├── tracks.py        # Tracking data
│   │   │   └── statistics.py    # Match statistics
│   │   ├── services/            # Business logic
│   │   │   ├── database.py      # SQLite database
│   │   │   └── websocket_manager.py
│   │   └── requirements.txt
│   │
│   ├── pipeline/                # Python CV Pipeline
│   │   ├── src/
│   │   │   ├── trackers/        # Detection & tracking
│   │   │   ├── team_assigner/   # Team classification
│   │   │   ├── pitch/           # Homography & pitch mapping
│   │   │   └── analytics/       # Statistics computation
│   │   ├── models/              # ML model files (.pt)
│   │   ├── main.py              # CLI entry point
│   │   └── requirements.txt
│   │
│   ├── server/                  # Node.js API (Manus Mode)
│   ├── drizzle/                 # Database schema
│   └── shared/                  # Shared types
│
├── docs/                        # Documentation
├── docker/                      # Docker configuration
├── Makefile                     # Simple commands
├── run-local.sh                 # Local development script
└── README.md
```

---

## Installation

### Prerequisites

- **Python 3.10+** - `brew install python@3.11`
- **Node.js 18+** - `brew install node` (for building frontend)
- **pnpm** - `npm install -g pnpm`

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/esharif20/football-analytics-dashboard.git
cd football-analytics-dashboard

# 2. Setup (installs Python + Node dependencies, builds frontend)
make setup

# 3. Run the application
make run
```

Open http://localhost:8000 in your browser.

---

## Available Commands

```bash
make setup    # Install all dependencies
make run      # Start dashboard at http://localhost:8000
make check    # Check system requirements
make clean    # Remove all dependencies

# Process video directly via CLI
make process VIDEO=/path/to/video.mp4
```

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

### Check GPU Support

```bash
make check
```

---

## API Reference (FastAPI)

### REST Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|

| `/api/videos` | GET | List videos |
| `/api/videos` | POST | Upload video |
| `/api/videos/{id}` | GET | Get video |
| `/api/videos/{id}` | DELETE | Delete video |
| `/api/analysis` | GET | List analyses |
| `/api/analysis` | POST | Start analysis |
| `/api/analysis/{id}` | GET | Get analysis |
| `/api/analysis/{id}/status` | PUT | Update status |
| `/api/analysis/modes` | GET | Get pipeline modes |
| `/api/events/{analysis_id}` | GET | Get events |
| `/api/tracks/{analysis_id}` | GET | Get tracks |
| `/api/statistics/{analysis_id}` | GET | Get statistics |

### Interactive API Docs

Open http://localhost:8000/docs for Swagger UI documentation.

---

## WebSocket Events

Real-time progress updates via WebSocket:

```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8000/ws/123');

// Receive progress updates
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  // { type: 'progress', analysisId: 123, progress: 45, currentStage: 'tracking' }
};
```

---

## Development

### Frontend Development

```bash
cd frontend
pnpm dev      # Start dev server with hot reload
pnpm build    # Build for production
```

### Pipeline Development

```bash
cd backend/pipeline
source venv/bin/activate
python main.py --video test.mp4 --mode all --verbose
```

---

## Troubleshooting

### "Module not found" errors

```bash
# Reinstall dependencies
make clean
make setup-local
```

### GPU not detected

```bash
# Check GPU support
make check

# For Apple Silicon, ensure PyTorch is installed correctly
pip install torch torchvision
```

### Port already in use

```bash
# Kill existing process
lsof -i :8000
kill -9 <PID>
```

---

## License

MIT License
