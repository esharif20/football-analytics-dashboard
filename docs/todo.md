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


## New Features (User Request)
- [x] Add player detection model to backend
- [x] GPU access guide (pay-as-you-go for BSc dissertation)
- [x] Stub caching for mode-dependent processing
- [x] ETA display during video processing
- [x] Termination option during processing
- [x] Clean up repo following SWE best practices


## Player Detection Model Integration
- [x] Copy player_detection.pt to backend/models/
- [x] Update config.py with player model path option
- [x] Update base.py to use custom player model when selected
- [x] Add CLI arguments for model source selection
- [x] Add player model selection in Upload UI
- [x] Document player model option in README


## Codebase Refactor & Local Setup
- [x] Refactor codebase following SWE best practices
- [x] Clear separation of concerns (client/server/backend)
- [x] Create comprehensive Mac-friendly README
- [x] Add local dev mode to bypass Manus OAuth/Forge dependencies
- [x] Include player_detection.pt model in backend
- [x] Create one-command setup script for Mac (setup-mac.sh)
- [x] Add video analysis caching system (utils/cache.py)
- [ ] Create worker service to connect pipeline to dashboard (future)


## Fully Standalone Mode (No Manus)
- [x] Remove all Manus OAuth dependencies
- [x] Switch to SQLite (zero-config database)
- [x] Create local-only auth bypass
- [x] Use local filesystem for all storage
- [x] Fix pitch backend default to ultralytics (no Roboflow needed)
- [x] Create simple one-command Mac setup
- [x] Test fully offline operation


## Worker Service & Caching (No External Dependencies)
- [x] Worker service that polls dashboard API for new uploads
- [x] Automatically runs Python pipeline on new videos
- [x] Posts results back to dashboard API
- [x] Pre-processed demo data for Test6.mp4
- [x] Video hash caching with SHA256 + model config
- [x] Skip re-processing for identical videos


## Real-time WebSocket Updates
- [x] Add WebSocket server to standalone backend
- [x] Create WebSocket event types for progress updates
- [x] Update worker to broadcast progress via WebSocket
- [x] Update frontend Analysis page to use WebSocket
- [x] Replace polling with WebSocket subscription
- [x] Handle reconnection and error states


## Local Setup & Deployment Improvements
- [x] Create detailed step-by-step setup guide (SETUP_GUIDE.md)
- [x] Add Makefile with simple commands (make setup, make run, make test)
- [x] Create Docker setup for easier deployment (Dockerfile, docker-compose.yml)
- [x] Document Mac M1/M2/M3 MPS GPU acceleration
- [x] Document RunPod/Colab Pro setup for faster processing


## Major Codebase Restructure
- [x] Restructure to cleaner architecture (pipeline/, frontend/, server/)
- [x] Move CV pipeline outside backend to dedicated pipeline/ folder
- [x] Create dedicated frontend/ folder for React dashboard
- [x] Add FastAPI for pipeline API (proper framework)
- [x] Add player_detection.pt model to pipeline/models/
- [x] Update all imports and configurations
- [x] Simplify folder structure for better readability


## Clean Root Structure
- [x] Restructure to minimal root with backend/ and frontend/ only
- [x] Move pipeline/ inside backend/
- [x] Move server/ inside backend/
- [x] Move docs (API_KEYS.md, GPU_SETUP.md, SETUP_GUIDE.md) to docs/
- [x] Remove redundant setup scripts (keep only Makefile)
- [x] Move Docker files to docker/ subdirectory
- [x] Update all imports and path references
- [x] Keep root clean: README, Makefile, package.json only


## Final Root Cleanup
- [x] Move package.json, pnpm-lock.yaml to frontend/
- [x] Move tsconfig.json to frontend/
- [x] Move vite.config.ts to frontend/
- [x] Move vitest.config.ts to backend/
- [x] Move drizzle.config.ts to backend/
- [x] Move .prettierrc, .prettierignore to appropriate location
- [x] Update Makefile to work from new locations
- [x] Root should only have: README.md, Makefile, .gitignore, package.json (orchestrator)


## Documentation
- [x] Add detailed system design diagram to README.md
