# Football Analysis Dashboard

A full-stack application for analyzing football match footage using computer vision and AI. Upload tactical wide-shot videos and get real-time player tracking, team classification, heatmaps, pass networks, and AI-generated tactical commentary.

## Project Structure

```
football-dashboard/
│
├── backend/             # Python CV Pipeline (runs separately)
│   ├── main.py          # CLI entry point
│   ├── analytics/       # Possession, kinematics, ball path
│   ├── pipeline/        # Pipeline modes (all, radar, team, etc.)
│   ├── pitch/           # Pitch detection & homography
│   ├── team_assigner/   # SigLIP + UMAP + KMeans
│   ├── trackers/        # YOLO + ByteTrack
│   └── utils/           # Shared utilities
│
├── client/              # React Frontend
│   └── src/
│       ├── pages/       # Home, Upload, Dashboard, Analysis
│       └── components/  # UI components
│
├── server/              # Express + tRPC Backend
│   ├── routers.ts       # API endpoints
│   └── db.ts            # Database queries
│
├── drizzle/             # Database Schema
│   └── schema.ts        # Videos, analyses, events, tracks
│
└── shared/              # Shared types
    └── types.ts         # Pipeline modes, API types
```

## Quick Start

### 1. Dashboard (Frontend + API)

```bash
# Install dependencies
pnpm install

# Push database schema
pnpm db:push

# Start development server
pnpm dev
```

Dashboard runs at `http://localhost:3000`

### 2. CV Pipeline (Backend)

```bash
cd backend

# Setup Python environment
chmod +x setup.sh
./setup.sh
source venv/bin/activate

# Process a video
python main.py --video /path/to/video.mp4 --mode all
```

For GPU acceleration, use RunPod or Google Colab (see `backend/README.md`).

## Features

### Dashboard
- Video upload with drag-and-drop
- Pipeline mode selection (all, radar, team, track, players, ball, pitch)
- Real-time processing status
- Video player with event timeline
- 2D pitch radar with player positions
- Voronoi diagram overlay
- Heatmaps (player movement, ball possession)
- Pass network visualization
- Statistics dashboard (possession, distance, speed)
- AI tactical commentary

### Pipeline
- YOLOv8 player/ball/goalkeeper detection
- ByteTrack object tracking with ID persistence
- SigLIP + UMAP + KMeans team classification
- Pitch keypoint detection (custom model or Roboflow API)
- Homography transformation for pitch coordinates
- Ball interpolation for missing frames
- Analytics computation (possession, kinematics)

## Camera Support

| Camera Type | Status |
|-------------|--------|
| Tactical Wide Shot (DFL Bundesliga style) | ✅ Supported |
| Broadcast Camera Angle | 🔜 Coming Soon |

## Tech Stack

### Frontend
- React 19 + TypeScript
- Tailwind CSS 4
- tRPC for type-safe API calls
- Recharts for visualizations

### Backend (Dashboard)
- Express 4
- tRPC 11
- Drizzle ORM + MySQL

### Backend (Pipeline)
- Python 3.10+
- PyTorch + CUDA/MPS
- Ultralytics YOLO
- supervision
- transformers (SigLIP)

## License

MIT License
