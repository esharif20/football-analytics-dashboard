# Football Analysis Pipeline (Backend)

Computer vision pipeline for football match analysis from tactical wide-shot footage (e.g., DFL Bundesliga clips). Processes video to extract player tracking, team classification, pitch coordinates, and analytics.

## Directory Structure

```
backend/
├── main.py              # CLI entry point
├── config.py            # Pipeline configuration
├── run.sh               # Shell wrapper script
├── setup.sh             # Environment setup script
├── requirements.txt     # Python dependencies
├── Dockerfile           # RunPod/Docker deployment
│
├── analytics/           # Analytics computation
│   ├── __init__.py      # AnalyticsEngine main interface
│   ├── types.py         # Data types (PossessionStats, KinematicStats, etc.)
│   ├── possession.py    # Possession calculation
│   ├── kinematics.py    # Player/ball movement stats
│   └── ball_path.py     # Ball trajectory tracking
│
├── cli/                 # Command-line interface
│   ├── __init__.py
│   ├── args.py          # Argument parsing
│   └── parsing.py       # Config parsing utilities
│
├── pipeline/            # Pipeline modes
│   ├── __init__.py      # Pipeline dispatcher
│   ├── base.py          # Base pipeline class
│   ├── all.py           # Full analysis pipeline
│   ├── radar.py         # Radar-only pipeline
│   ├── team.py          # Team classification pipeline
│   ├── tracking.py      # Tracking-only pipeline
│   ├── players.py       # Player detection pipeline
│   ├── ball.py          # Ball detection pipeline
│   └── pitch.py         # Pitch detection pipeline
│
├── pitch/               # Pitch detection & homography
│   ├── __init__.py
│   ├── config.py        # Pitch configuration
│   ├── view_transformer.py    # Homography transformation
│   ├── homography_smoother.py # Temporal smoothing
│   └── annotators.py    # Pitch visualization
│
├── team_assigner/       # Team classification
│   ├── __init__.py
│   └── team_assigner.py # SigLIP + UMAP + KMeans clustering
│
├── trackers/            # Detection & tracking
│   ├── __init__.py
│   ├── tracker.py       # Main tracker (YOLO + ByteTrack)
│   ├── ball_tracker.py  # Ball-specific tracking
│   ├── ball_config.py   # Ball detection config
│   ├── detection.py     # Detection utilities
│   ├── people.py        # Person detection
│   ├── annotator.py     # Visualization
│   ├── track_stabiliser.py  # Track ID stabilization
│   └── ball/            # Ball detection sub-module
│       ├── __init__.py
│       └── filter.py    # Ball filtering
│
└── utils/               # Shared utilities
    ├── __init__.py
    ├── video_utils.py   # Video I/O
    ├── bbox_utils.py    # Bounding box utilities
    ├── cache.py         # Stub caching
    ├── device.py        # Device detection
    ├── drawing.py       # Drawing utilities
    ├── errors.py        # Custom exceptions
    ├── metrics.py       # Evaluation metrics
    ├── pitch_detector.py # Pitch keypoint detection
    ├── logging_config.py # Logging setup
    └── validation.py    # Input validation
```

## Quick Start

### Local Setup (Mac/Linux)

```bash
cd backend
chmod +x setup.sh
./setup.sh
source venv/bin/activate
python main.py --video /path/to/video.mp4 --mode all
```

### CLI Usage

```bash
# Full analysis (all features)
python main.py --video input.mp4 --mode all

# Radar view only
python main.py --video input.mp4 --mode radar

# Team classification only
python main.py --video input.mp4 --mode team

# Player tracking only
python main.py --video input.mp4 --mode track

# Use Roboflow API for pitch detection (fallback)
python main.py --video input.mp4 --mode all --use-roboflow

# Custom output directory
python main.py --video input.mp4 --mode all --output-dir ./results
```

### Pipeline Modes

| Mode | Description | Output |
|------|-------------|--------|
| `all` | Full analysis | Annotated video, radar, tracks JSON, analytics JSON |
| `radar` | Radar view only | Radar video, tracks JSON |
| `team` | Team classification | Tracks with team IDs |
| `track` | Object tracking | Tracks JSON |
| `players` | Player detection | Player bounding boxes |
| `ball` | Ball detection | Ball positions with interpolation |
| `pitch` | Pitch detection | Keypoints, homography matrix |

## GPU Deployment

### RunPod

```bash
docker build -t football-pipeline .
docker run --gpus all -v /path/to/videos:/data football-pipeline \
    python main.py --video /data/match.mp4 --mode all
```

### Google Colab

Open `Football_Analysis_Pipeline.ipynb` in Colab for GPU-accelerated processing.

## Output Files

```
output_videos/<video_name>/
├── <video_name>_annotated.mp4   # Video with bounding boxes & overlays
├── <video_name>_radar.mp4       # 2D pitch radar view
├── <video_name>_tracks.json     # Raw tracking data
└── <video_name>_analytics.json  # Computed statistics
```

## Output Format

### Tracks JSON
```json
{
  "players": [
    {
      "1": {"bbox": [x1, y1, x2, y2], "confidence": 0.95, "team_id": 0},
      "2": {"bbox": [x1, y1, x2, y2], "confidence": 0.92, "team_id": 1}
    }
  ],
  "goalkeepers": [...],
  "referees": [...],
  "ball": [{"1": {"bbox": [x1, y1, x2, y2], "confidence": 0.88}}]
}
```

### Analytics JSON
```json
{
  "possession": {
    "team_1_percentage": 52.3,
    "team_2_percentage": 47.7,
    "possession_changes": 15
  },
  "player_stats": {
    "1": {"total_distance_m": 1523, "avg_speed_m_per_sec": 4.2, "max_speed_m_per_sec": 8.5}
  },
  "ball_stats": {
    "total_distance_m": 823, "avg_speed_m_per_sec": 12.5, "detection_rate": 0.85
  }
}
```

## Custom Models

Place custom-trained models in a `models/` directory:

- `ball_detection.pt` - Ball detection model
- `pitch_detection.pt` - Pitch keypoint detection model

If not found, falls back to:
- Main YOLOv8x model for ball detection
- Roboflow API for pitch detection (requires API key)

## Performance

| Stage | GPU (RTX 3090) | CPU |
|-------|----------------|-----|
| Detection | ~45s | ~180s |
| Tracking | ~5s | ~5s |
| Team Classification | ~30s | ~60s |
| Pitch Detection | ~20s | ~40s |
| Analytics | ~2s | ~2s |
| Rendering | ~60s | ~120s |
| **Total (30s video)** | **~2.5 min** | **~7 min** |

## Requirements

- Python 3.10+
- PyTorch with CUDA/MPS support (optional)
- Ultralytics YOLO
- supervision
- transformers (SigLIP)
- OpenCV, NumPy

See `requirements.txt` for full list.
