# Football Analysis Dashboard

A full-stack application for analyzing football match footage using computer vision and AI. Upload tactical wide-shot videos (DFL Bundesliga style) and get real-time player tracking, team classification, heatmaps, pass networks, and AI-generated tactical commentary.

## Project Structure

```
football-dashboard/
│
├── backend/                 # 🐍 Python CV Pipeline (runs separately)
│   ├── main.py              # CLI entry point
│   ├── setup.sh             # One-command setup (Mac/Linux/GPU)
│   ├── config.py            # Pipeline configuration
│   │
│   ├── pipeline/            # Pipeline modes (all, radar, team, etc.)
│   ├── trackers/            # YOLO detection + ByteTrack
│   ├── team_assigner/       # SigLIP + UMAP + KMeans
│   ├── pitch/               # Pitch detection & homography
│   ├── analytics/           # Possession, kinematics, events
│   └── utils/               # Shared utilities
│
├── client/                  # ⚛️ React Frontend
│   └── src/
│       ├── pages/           # Home, Upload, Dashboard, Analysis
│       └── components/      # UI components
│
├── server/                  # 🚀 Express + tRPC API
│   ├── routers.ts           # API endpoints
│   └── db.ts                # Database queries
│
├── drizzle/                 # 🗄️ Database Schema
│   └── schema.ts            # Videos, analyses, events, tracks
│
└── shared/                  # 📦 Shared TypeScript types
    └── types.ts             # Pipeline modes, API types
```

## Quick Start

### Option 1: Dashboard Only (View/Upload Interface)

```bash
# Install Node.js dependencies
pnpm install

# Push database schema
pnpm db:push

# Start development server
pnpm dev
```

Dashboard runs at `http://localhost:3000`

### Option 2: Full Pipeline (CV Processing)

```bash
# Navigate to backend
cd backend

# Run setup (creates venv, installs deps, downloads models)
chmod +x setup.sh
./setup.sh

# Activate environment
source venv/bin/activate

# Process a video
python main.py --video /path/to/video.mp4 --mode all
```

## API Keys Required

| Key | Required | Purpose | How to Get |
|-----|----------|---------|------------|
| **Roboflow API Key** | Optional | Fallback for pitch detection if custom model fails | [roboflow.com](https://roboflow.com) - Free tier available |
| **Gemini API Key** | Optional | AI tactical commentary generation | [ai.google.dev](https://ai.google.dev) - Free tier available |

**Note:** The pipeline works without any API keys using the custom-trained models. API keys are only needed for fallback/enhanced features.

### Setting API Keys

Create a `.env` file in the `backend/` directory:

```bash
# backend/.env
ROBOFLOW_API_KEY=your_roboflow_key_here  # Optional - for pitch detection fallback
```

For the dashboard AI commentary, add your Gemini key in the web interface settings.

## Pipeline Modes

| Mode | Description | Output |
|------|-------------|--------|
| `all` | Full analysis | Annotated video, radar, tracks JSON, analytics JSON |
| `radar` | Radar view only | Radar video, tracks JSON |
| `team` | Team classification | Tracks with team IDs |
| `track` | Object tracking | Tracks JSON |
| `players` | Player detection | Player bounding boxes |
| `ball` | Ball detection | Ball positions with interpolation |
| `pitch` | Pitch detection | Keypoints, homography matrix |

## Mac Compatibility

The pipeline is fully compatible with Mac (both Apple Silicon and Intel):

- **Apple Silicon (M1/M2/M3)**: Uses MPS (Metal Performance Shaders) for GPU acceleration
- **Intel Mac**: CPU-only processing (slower but functional)

The setup script automatically detects your hardware and installs the appropriate PyTorch version.

## Output Files

After processing, outputs are saved to `backend/output_videos/<video_name>/`:

```
output_videos/<video_name>/
├── <video_name>_annotated.mp4   # Video with bounding boxes & overlays
├── <video_name>_radar.mp4       # 2D pitch radar view
├── <video_name>_tracks.json     # Raw tracking data
└── <video_name>_analytics.json  # Computed statistics
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| **Frontend** | React 19, TypeScript, Tailwind CSS 4 |
| **API** | Express 4, tRPC 11, Drizzle ORM |
| **Database** | MySQL/TiDB |
| **CV Pipeline** | Python 3.10+, PyTorch, Ultralytics YOLO |
| **Tracking** | ByteTrack, supervision |
| **Team Classification** | SigLIP, UMAP, KMeans |

## Camera Support

| Camera Type | Status |
|-------------|--------|
| Tactical Wide Shot (DFL Bundesliga style) | ✅ Supported |
| Broadcast Camera Angle | 🔜 Coming Soon |

## Troubleshooting

### "MPS not available" on Mac
Ensure you have macOS 12.3+ and PyTorch 1.12+. The setup script handles this automatically.

### Models not downloading
If gdown fails, manually download from Google Drive:
- [player_detection.pt](https://drive.google.com/uc?id=17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q)
- [ball_detection.pt](https://drive.google.com/uc?id=1isw4wx-MK9h9LMr36VvIWlJD6ppUvw7V)
- [pitch_detection.pt](https://drive.google.com/uc?id=1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf)

Place them in `backend/models/`.

### Slow processing
- Use GPU if available (CUDA or MPS)
- Reduce video resolution before processing
- Use `--mode radar` for faster processing (skips annotated video)

## License

MIT License
