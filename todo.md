# Football Analysis Dashboard - TODO

## Core Features
- [x] Video upload interface with drag-and-drop support
- [x] Pipeline mode selector (all/radar/team/track/players/ball/pitch)
- [x] Real-time processing status tracker with progress percentage
- [x] Video player with synchronized event timeline
- [x] Live 2D pitch radar visualization with player positions
- [x] Ball trajectory visualization on pitch
- [x] Voronoi diagram overlay on pitch
- [x] Interactive heatmap generator for player movement
- [x] Ball possession zone heatmap
- [x] Pass network visualization with directional arrows
- [x] Pass success rate display
- [x] Statistics dashboard (possession, distance, speed)
- [x] AI tactical commentary panel
- [x] Mode-specific visualization panels

## Backend
- [x] Database schema for videos, analyses, events, tracks
- [x] Video upload endpoint with S3 storage
- [x] Pipeline processing status tracking
- [x] Event detection from tracking data
- [x] AI commentary generation with LLM
- [x] Statistics calculation from tracks

## UI/UX
- [x] Dark theme professional sports analytics design
- [x] Responsive layout for dashboard
- [x] Loading states and progress indicators
- [x] Error handling and user feedback

## New Features
- [x] Model selection option (custom models vs Roboflow API)
- [x] Upload custom model files (ball_detection.pt, pitch_detection.pt)
- [x] Enhanced landing page with modern visuals
- [x] Animated hero section with football imagery
- [x] Feature showcase with visual cards

## Pipeline Integration
- [x] Create Python processing service (FastAPI worker)
- [x] Port YOLOv8 player/ball/goalkeeper detection
- [x] Port ByteTrack object tracking with ID persistence
- [x] Port SigLIP + UMAP + KMeans team classification
- [x] Port pitch keypoint detection (custom model + Roboflow fallback)
- [x] Port homography transformation for pitch coordinates
- [x] Port ball interpolation for missing frames
- [x] Implement event detection (passes, shots, challenges)
- [x] Connect pipeline output to database (via callback URL)
- [x] Real-time progress updates via callback/polling
- [x] Generate annotated video output
- [x] Generate radar video output
- [x] Store tracking JSON in database

## Deployment
- [x] Setup script for Mac/Linux/RunPod/Colab
- [x] Docker configuration for RunPod
- [x] Google Colab notebook with GPU support
- [x] Pipeline API documentation (README.md)

## Pipeline Restructure
- [x] Restructure pipeline to match original repo's modular architecture
- [x] Remove FastAPI overhead - keep as simple Python CLI/package
- [x] Separate modules: trackers/, team_assigner/, pitch/, analytics/, utils/
- [x] Match original CLI interface (python main.py --video --mode)
- [x] Add "Coming Soon" option for broadcast/normal camera angle pipeline
- [x] Camera type selection in Upload UI (Tactical vs Broadcast)

## Project Restructure
- [x] Remove monolithic processor.py - split into proper modules
- [x] Create proper Python packages with __init__.py exports
- [x] Match original repo's module structure exactly
- [x] Clear backend/frontend separation in project root (backend/ vs client/)
- [x] Clean up and neaten code following original repo patterns
- [x] Add proper imports and package structure
- [x] Root README explaining full project structure


## Cleanup & Verification
- [x] Clean up repo structure for clarity
- [x] Verify pipeline works end-to-end (CLI --help works)
- [x] Ensure Mac compatibility (MPS support in device.py)
- [x] Document required API keys clearly (API_KEYS.md)
- [x] Create clear setup instructions (README.md, setup.sh)
