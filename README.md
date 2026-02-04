# Football Analysis Dashboard

A full-stack application for analyzing football match footage using computer vision and AI. Upload tactical wide-shot videos and get real-time player tracking, team classification, heatmaps, pass networks, and AI-generated tactical commentary.

## Quick Start (Mac)

### Prerequisites

- **Node.js 18+** - `brew install node`
- **pnpm** - `npm install -g pnpm`
- **Python 3.10+** - `brew install python@3.11`
- **MySQL** (optional for local dev) - `brew install mysql`

### 1. Clone and Install

```bash
git clone <your-repo-url>
cd football-dashboard
pnpm install
```

### 2. Set Up Environment

**Option A: Local Development Mode (Recommended for Testing)**

Create a `.env` file in the project root:

```bash
# .env
LOCAL_DEV_MODE=true
DATABASE_URL="mysql://root:password@localhost:3306/football"
JWT_SECRET="any-random-string-for-local-dev"
```

This mode:
- Bypasses Manus OAuth (auto-logged in as "Local Developer")
- Uses local filesystem for file storage (`./uploads/`)
- Works without any external API keys

**Option B: Full Manus Mode (Production)**

```bash
# .env
DATABASE_URL="mysql://user:pass@host:3306/football"
JWT_SECRET="your-session-secret"
VITE_APP_ID="manus_app_id"
OAUTH_SERVER_URL="https://api.manus.im"
VITE_OAUTH_PORTAL_URL="https://manus.im/login"
BUILT_IN_FORGE_API_URL="https://..."
BUILT_IN_FORGE_API_KEY="your-forge-key"
```

### 3. Set Up Database

```bash
# Start MySQL (if using local)
brew services start mysql

# Create database
mysql -u root -e "CREATE DATABASE football;"

# Push schema
pnpm db:push
```

### 4. Run the Dashboard

```bash
pnpm dev
```

Open http://localhost:3000 in your browser.

---

## CV Pipeline Setup (For Video Processing)

The Python pipeline processes videos and generates tracking data.

### 1. Set Up Python Environment

```bash
cd backend
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

### 2. Download Models

Download your custom-trained models and place them in `backend/models/`:

```
backend/models/
├── player_detection.pt
├── ball_detection.pt
└── pitch_detection.pt
```

Or use the setup script which downloads them automatically.

### 3. Run the Pipeline

```bash
# Full analysis
python main.py \
  --source-video-path input_videos/your_video.mp4 \
  --target-video-path output_videos/result.mp4 \
  --mode all

# Just radar view (faster)
python main.py \
  --source-video-path input_videos/your_video.mp4 \
  --target-video-path output_videos/result.mp4 \
  --mode radar
```

### Pipeline Modes

| Mode | Description | Speed |
|------|-------------|-------|
| `all` | Full analysis with annotated + radar video | Slowest |
| `radar` | Radar view only | Fast |
| `team` | Team classification | Medium |
| `track` | Object tracking only | Fast |
| `players` | Player detection only | Fastest |
| `ball` | Ball detection + interpolation | Fast |
| `pitch` | Pitch keypoint detection | Fast |

---

## Project Structure

```
football-dashboard/
│
├── client/                  # React Frontend (Vite)
│   └── src/
│       ├── pages/           # Home, Upload, Dashboard, Analysis
│       ├── components/      # UI components (shadcn/ui)
│       └── lib/             # Utilities, tRPC client
│
├── server/                  # Express + tRPC API
│   ├── routers.ts           # API endpoints
│   ├── db.ts                # Database queries
│   ├── storage.ts           # File storage (local/cloud)
│   └── _core/               # Auth, context, middleware
│       └── localMode.ts     # Local dev mode config
│
├── backend/                 # Python CV Pipeline
│   ├── main.py              # CLI entry point
│   ├── config.py            # Pipeline configuration
│   ├── setup.sh             # One-command setup
│   │
│   ├── pipeline/            # Pipeline modes
│   │   ├── all.py           # Full analysis
│   │   ├── radar.py         # Radar view
│   │   ├── team.py          # Team classification
│   │   └── ...
│   │
│   ├── trackers/            # Detection & tracking
│   │   ├── tracker.py       # Main tracker (YOLO + ByteTrack)
│   │   └── ball_tracker.py  # Ball-specific tracking
│   │
│   ├── team_assigner/       # Team classification
│   │   └── team_assigner.py # SigLIP + UMAP + KMeans
│   │
│   ├── pitch/               # Pitch detection
│   │   ├── view_transformer.py  # Homography
│   │   └── homography_smoother.py
│   │
│   ├── analytics/           # Statistics computation
│   │   ├── possession.py
│   │   └── kinematics.py
│   │
│   └── utils/               # Shared utilities
│       ├── cache.py         # Stub caching
│       └── device.py        # GPU/MPS detection
│
├── drizzle/                 # Database schema
│   └── schema.ts
│
└── shared/                  # Shared TypeScript types
    └── types.ts
```

---

## Mac Compatibility

### Apple Silicon (M1/M2/M3/M4)

The pipeline automatically uses **MPS (Metal Performance Shaders)** for GPU acceleration:

```python
# Automatic detection in backend/utils/device.py
if torch.backends.mps.is_available():
    device = "mps"  # Uses Apple GPU
```

### Intel Mac

Falls back to CPU processing (slower but works).

### Troubleshooting

**"MPS not available"**
- Ensure macOS 12.3+ and PyTorch 1.12+
- Run: `pip install torch torchvision --upgrade`

**Models not downloading**
- Download manually from Google Drive links in `backend/setup.sh`
- Place in `backend/models/`

**Slow processing**
- Use `--mode radar` for faster results
- Reduce video resolution before processing

---

## API Keys (Optional)

| Key | Purpose | Required? |
|-----|---------|-----------|
| Roboflow | Pitch detection fallback | No - custom model works |
| Gemini | AI tactical commentary | No - basic stats still work |

The pipeline works fully offline with your custom models.

---

## Development Workflow

### Dashboard Development

```bash
# Start dev server with hot reload
pnpm dev

# Run tests
pnpm test

# Type check
pnpm check
```

### Pipeline Development

```bash
cd backend
source venv/bin/activate

# Run with debug output
python main.py --video test.mp4 --mode all --verbose

# Test specific module
python -c "from trackers import Tracker; print('OK')"
```

---

## Connecting Pipeline to Dashboard

Currently, the pipeline runs independently. To see results in the dashboard:

1. **Manual**: Run pipeline, then import JSON results via dashboard API
2. **Automated**: Set up a worker service (see `GPU_SETUP.md`)

The dashboard polls for analysis status and displays results when available.

---

## License

MIT License
