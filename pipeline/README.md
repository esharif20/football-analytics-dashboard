# Football Analysis Pipeline

A computer vision pipeline for analyzing football match footage. This service processes uploaded videos through YOLOv8 detection, ByteTrack tracking, SigLIP team classification, and pitch homography to generate tracking data, analytics, and visualizations.

## Features

- **Object Detection**: YOLOv8x for players, goalkeepers, referees, and ball
- **Object Tracking**: ByteTrack for persistent ID tracking across frames
- **Team Classification**: SigLIP embeddings + UMAP + KMeans clustering
- **Pitch Detection**: Custom model or Roboflow API fallback
- **Homography**: Frame-to-pitch coordinate transformation
- **Analytics**: Possession, distance, speed, and event detection
- **Output Videos**: Annotated video with overlays and radar view

## Quick Start

### Local Setup (Mac/Linux)

```bash
cd pipeline
chmod +x setup.sh
./setup.sh
source venv/bin/activate
python main.py
```

### Google Colab

1. Open `Football_Analysis_Pipeline.ipynb` in Google Colab
2. Enable GPU runtime (Runtime → Change runtime type → GPU)
3. Run all cells to process your video

### RunPod / Docker

```bash
docker build -t football-pipeline .
docker run -p 8001:8001 --gpus all football-pipeline
```

## API Endpoints

### Health Check
```
GET /health
```

### Start Processing
```
POST /process
{
  "video_url": "https://example.com/video.mp4",
  "analysis_id": "unique-id",
  "mode": "all",
  "use_custom_models": true,
  "callback_url": "https://your-dashboard.com/api/callback"
}
```

### Get Status
```
GET /status/{analysis_id}
```

### Get Result
```
GET /result/{analysis_id}
```

## Pipeline Modes

| Mode | Description |
|------|-------------|
| `all` | Full pipeline with all features |
| `radar` | Detection + tracking + team + pitch + radar overlay |
| `team` | Detection + tracking + team classification |
| `track` | Detection + tracking only |
| `players` | Player detection only |
| `ball` | Ball detection only |
| `pitch` | Pitch keypoint detection only |

## Custom Models

Place your custom-trained models in the `models/` directory:

- `ball_detection.pt` - Ball detection model
- `pitch_detection.pt` - Pitch keypoint detection model

If custom models are not found, the pipeline will:
- Use the main YOLOv8x model for ball detection
- Fall back to Roboflow API for pitch detection (requires API key)

## Environment Variables

```bash
# Pipeline Configuration
PIPELINE_PORT=8001
DEVICE=auto  # auto, cuda, mps, cpu

# Model Paths
DET_MODEL_PATH=models/yolov8x.pt
BALL_MODEL_PATH=models/ball_detection.pt
PITCH_MODEL_PATH=models/pitch_detection.pt

# Roboflow API (fallback for pitch detection)
ROBOFLOW_API_KEY=your_api_key

# Dashboard Callback URL
CALLBACK_URL=https://your-dashboard.com/api/callback
```

## Output Format

### Tracks JSON Structure
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
  "ball": [
    {"1": {"bbox": [x1, y1, x2, y2], "confidence": 0.88}}
  ]
}
```

### Analytics JSON Structure
```json
{
  "possession": {
    "team_1_percentage": 52.3,
    "team_2_percentage": 47.7,
    "team_1_frames": 523,
    "team_2_frames": 477
  },
  "player_stats": {
    "1": {
      "track_id": 1,
      "team_id": 0,
      "total_distance_px": 15234,
      "avg_speed_px": 12.5,
      "max_speed_px": 45.2
    }
  },
  "ball_stats": {
    "total_distance_px": 8234,
    "avg_speed_px": 25.3,
    "detection_rate": 0.85
  }
}
```

## Integration with Dashboard

The pipeline service communicates with the dashboard via:

1. **Callback URL**: Progress updates sent to `callback_url` during processing
2. **Result Endpoint**: Dashboard polls `/result/{analysis_id}` for final data
3. **WebSocket** (optional): Real-time progress streaming

## Performance

| Stage | Time (30s video @ 25fps) |
|-------|-------------------------|
| Detection | ~45s (GPU) / ~180s (CPU) |
| Tracking | ~5s |
| Team Classification | ~30s |
| Pitch Detection | ~20s |
| Analytics | ~2s |
| Rendering | ~60s |
| **Total** | **~2.5 min (GPU)** |

## License

MIT License - See LICENSE file for details.
