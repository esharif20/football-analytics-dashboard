# Football Analysis Dashboard - Complete Setup Guide

This guide walks you through setting up the Football Analysis Dashboard on your local machine. The application consists of three main components:

1. **Frontend** - React dashboard for uploading videos and viewing results
2. **Server** - Node.js/tRPC API for data management
3. **Pipeline** - Python/FastAPI computer vision pipeline

## Project Structure

```
football-dashboard/
├── frontend/           # React dashboard (Vite + Tailwind)
│   ├── src/
│   │   ├── pages/      # Page components
│   │   ├── components/ # Reusable UI components
│   │   └── hooks/      # Custom React hooks
│   └── public/         # Static assets
├── server/             # Node.js tRPC API
│   ├── routers.ts      # API endpoints
│   └── db.ts           # Database helpers
├── pipeline/           # Python CV pipeline (FastAPI)
│   ├── api/            # FastAPI server
│   ├── src/            # Pipeline modules
│   │   ├── trackers/   # Object detection & tracking
│   │   ├── team_assigner/ # Team classification
│   │   ├── pitch/      # Pitch detection & homography
│   │   └── analytics/  # Match analytics
│   └── models/         # ML model weights (.pt files)
├── drizzle/            # Database schema
└── shared/             # Shared types
```

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Quick Start](#quick-start)
3. [Detailed Installation](#detailed-installation)
4. [Running the Application](#running-the-application)
5. [GPU Acceleration](#gpu-acceleration)
6. [Cloud GPU Options](#cloud-gpu-options)
7. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Minimum Requirements

| Component | Requirement |
|-----------|-------------|
| Operating System | macOS 12+, Ubuntu 20.04+, or Windows 11 (WSL2) |
| RAM | 8 GB (16 GB recommended for video processing) |
| Storage | 5 GB free space |
| Node.js | v18.0.0 or higher |
| Python | 3.9, 3.10, or 3.11 |
| pnpm | v8.0.0 or higher |

### Recommended for Best Performance

For processing full match videos efficiently, we recommend Apple Silicon (M1/M2/M3) Macs or a machine with an NVIDIA GPU. The pipeline automatically detects and uses available hardware acceleration.

---

## Quick Start

If you want to get up and running quickly, use the one-command setup:

```bash
# Clone the repository (or download and extract)
git clone <your-repo-url> football-dashboard
cd football-dashboard

# Full setup (dashboard + pipeline)
make setup

# Start everything
make run
```

This starts:
- Dashboard at `http://localhost:3000`
- Pipeline API at `http://localhost:8000`
- API docs at `http://localhost:8000/docs`

---

## Detailed Installation

### Step 1: Install Prerequisites

**On macOS (using Homebrew):**

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Node.js and pnpm
brew install node
npm install -g pnpm

# Install Python 3.11
brew install python@3.11
```

**On Ubuntu/Debian:**

```bash
# Update package list
sudo apt update

# Install Node.js 18+
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs

# Install pnpm
npm install -g pnpm

# Install Python 3.11
sudo apt install -y python3.11 python3.11-venv python3-pip
```

### Step 2: Set Up the Web Dashboard

Navigate to the project root and install Node.js dependencies:

```bash
cd football-dashboard

# Install dependencies
pnpm install

# Initialize the SQLite database
pnpm db:push
```

This creates a local SQLite database file that stores all your analyses, videos, and results.

### Step 3: Set Up the Python Pipeline

The computer vision pipeline requires a separate Python environment:

```bash
# Navigate to the pipeline directory
cd pipeline

# Create a virtual environment
python3.11 -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows (PowerShell):
# .\venv\Scripts\Activate.ps1

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

The installation includes PyTorch, Ultralytics YOLOv8, OpenCV, FastAPI, and other libraries.

### Step 4: Add Model Weights

Place your trained model files in the `pipeline/models/` directory:

```
pipeline/models/
├── player_detection.pt    # Player detection model
├── ball_detection.pt      # Ball detection model (optional)
└── pitch_detection.pt     # Pitch keypoint model (optional)
```

### Step 5: Verify the Installation

Test that everything is installed correctly:

```bash
# Test the dashboard (from project root)
cd ..
pnpm test

# Test the Python pipeline
cd pipeline
source venv/bin/activate
python main.py --help
```

---

## Running the Application

### Option A: Using Make (Recommended)

```bash
# Start everything at once
make run

# Or start components individually:
make dashboard    # Start web dashboard only (port 3000)
make api          # Start FastAPI pipeline server (port 8000)
make worker       # Start pipeline worker only
```

### Option B: Manual Start (Two Terminals)

**Terminal 1 - Web Dashboard:**

```bash
cd football-dashboard
pnpm dev
```

The dashboard starts at `http://localhost:3000`

**Terminal 2 - Pipeline API:**

```bash
cd football-dashboard/pipeline
source venv/bin/activate
python -m api.server
```

The FastAPI server starts at `http://localhost:8000` with interactive docs at `/docs`

### Option C: Direct Pipeline Processing

If you want to process a video directly without the dashboard:

```bash
cd football-dashboard/pipeline
source venv/bin/activate
python main.py \
  --source-video-path /path/to/your/video.mp4 \
  --target-video-path /path/to/output.mp4 \
  --mode all
```

---

## GPU Acceleration

### Apple Silicon (M1/M2/M3) Macs

The pipeline automatically uses **MPS (Metal Performance Shaders)** for GPU acceleration on Apple Silicon. No additional configuration is needed. Verify MPS is available:

```bash
cd pipeline
source venv/bin/activate
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

Expected output: `MPS available: True`

### NVIDIA GPUs (Linux/Windows)

For NVIDIA GPUs, ensure you have CUDA installed, then install the CUDA version of PyTorch:

```bash
cd pipeline
source venv/bin/activate
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Verify CUDA is available:

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### CPU-Only Mode

If no GPU is available, the pipeline falls back to CPU processing. This is significantly slower but works on any machine. A 30-second video clip may take 5-10 minutes on CPU versus 30-60 seconds on GPU.

---

## Cloud GPU Options

For processing longer videos or when local GPU power is insufficient, consider these pay-as-you-go cloud options suitable for a BSc dissertation budget.

### RunPod (~$0.20/hour)

RunPod offers affordable GPU instances with PyTorch pre-installed.

1. Create an account at [runpod.io](https://runpod.io)
2. Deploy a "PyTorch" template pod with RTX 3090 or A4000
3. Upload your project via the web terminal or `rsync`
4. Run the pipeline as described above

**Cost estimate:** Processing a 90-minute match takes approximately 15-30 minutes, costing roughly $0.05-$0.10.

### Google Colab Pro (~$10/month)

Colab Pro provides access to faster GPUs and longer runtimes.

1. Upload the project to Google Drive
2. Open a new Colab notebook
3. Mount Drive and install dependencies
4. Run the pipeline

**Tip:** Use Google Drive to store videos and results for persistence between sessions.

### Vast.ai (~$0.10-0.30/hour)

Vast.ai offers the cheapest GPU rentals but requires more setup:

1. Create an account at [vast.ai](https://vast.ai)
2. Search for instances with PyTorch and at least 8GB VRAM
3. Use SSH to connect and set up the environment

---

## Troubleshooting

### Common Issues

**"pnpm: command not found"**

Install pnpm globally:
```bash
npm install -g pnpm
```

**"Python version not supported"**

The pipeline requires Python 3.9-3.11. Check your version:
```bash
python3 --version
```

**"torch not compiled with MPS support"**

Reinstall PyTorch for Apple Silicon:
```bash
pip uninstall torch torchvision
pip install torch torchvision
```

**"CUDA out of memory"**

Reduce batch size or video resolution. For very long videos, consider processing in segments.

**Dashboard shows "Connection refused"**

Ensure the dashboard is running (`pnpm dev`) before starting the worker.

### Getting Help

If you encounter issues not covered here:

1. Check the `.manus-logs/` directory for detailed error logs
2. Run `make test` to verify the installation
3. Open an issue on the repository with your error message and system details

---

## Next Steps

Once everything is running:

1. Open `http://localhost:3000` in your browser
2. Click "Upload Video" to analyze your first match
3. Select a pipeline mode (start with "all" for full analysis)
4. Watch the real-time progress updates via WebSocket
5. Explore the visualizations: radar view, heatmaps, pass networks

For your dissertation, consider using the "View Demo" option first to see pre-processed results without waiting for pipeline execution.
