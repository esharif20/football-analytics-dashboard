# Pipeline Capabilities (from CS350 Progress Report)

## Layer 1: Perception Pipeline (COMPLETE)
Five sequential stages:

1. **Object Detection** — YOLOv8x fine-tuned
   - Classes: Player (99.4% mAP@50), Goalkeeper (94.2%), Referee (98.2%), Ball
   - Dedicated ball detection model using InferenceSlicer (92.5% mAP@50)
   - YOLOv8x-pose for 32 pitch keypoints (0.97 mAP@50)

2. **Multi-Object Tracking** — ByteTrack
   - Two-stage association (high-conf IoU + low-conf recovery)
   - TrackStabiliser for role flickering (majority voting)
   - 25 fps temporal resolution

3. **Team Classification** — SigLIP + UMAP + KMeans
   - 768-dim embeddings from SigLIP vision transformer
   - UMAP dimensionality reduction
   - KMeans k=2 clustering

4. **Coordinate Transformation** — Homography
   - Pitch keypoint detection → homography matrix
   - Pixel coords → real-world pitch positions (metres)

5. **Data Export** — Structured JSON
   - Player positions, team assignments, ball location per frame
   - Caches intermediate results to disk (stubs)

## Layer 2: Analytics (derived from tracking data)
- Possession percentage
- Territorial dominance
- Distance covered per player/team
- Average speed per player/team
- Formation compactness
- Defensive line height
- Pressing intensity
- Shots, passes, pass accuracy

## Layer 3: VLM Reasoning (planned/in progress)
- Grounded tactical commentary using structured tracking data as context
- Natural language explanations of tactical patterns
- Grounding prevents hallucination

## Visualizations the pipeline produces
- Annotated video overlays (bounding boxes + track IDs)
- 2D tactical radar view (pitch with player positions)
- Player heatmaps
- Team shape graphs
- Pass networks (implied)

## What the dashboard should show
- Video player with annotated output
- 2D pitch radar with player positions (real tracking data)
- Heatmap of player activity zones
- Pass network visualization
- Match statistics (possession, passes, shots, distance, speed)
- AI tactical commentary (grounded in data)
- Detection model performance metrics
- Pipeline stage progress tracking
