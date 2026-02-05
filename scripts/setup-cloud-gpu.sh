#!/bin/bash
# =============================================================================
# Football Analytics - Cloud GPU Worker Setup
# =============================================================================
# Run this script on a cloud GPU instance (RunPod, Lambda Labs, Vast.ai, etc.)
# It will set up the worker to process videos from your deployed site.
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/esharif20/football-analytics-dashboard/main/scripts/setup-cloud-gpu.sh | bash
#
# Or manually:
#   chmod +x setup-cloud-gpu.sh
#   ./setup-cloud-gpu.sh
# =============================================================================

set -e

echo "=========================================="
echo "Football Analytics - Cloud GPU Worker"
echo "=========================================="

# Configuration
DASHBOARD_URL="${DASHBOARD_URL:-https://aifootball.manus.space}"
REPO_URL="https://github.com/esharif20/football-analytics-dashboard.git"

echo ""
echo "Dashboard URL: $DASHBOARD_URL"
echo ""

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    echo "✅ GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠️  No GPU detected - will run on CPU (slower)"
fi

echo ""
echo "Step 1: Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3-pip python3-venv git ffmpeg libgl1-mesa-glx libglib2.0-0

echo ""
echo "Step 2: Cloning repository..."
if [ -d "football-analytics-dashboard" ]; then
    echo "Repository already exists, pulling latest..."
    cd football-analytics-dashboard
    git pull
else
    git clone --depth 1 $REPO_URL
    cd football-analytics-dashboard
fi

echo ""
echo "Step 3: Setting up Python environment..."
cd backend/pipeline
python3 -m venv venv
source venv/bin/activate

echo ""
echo "Step 4: Installing Python dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo ""
echo "Step 5: Creating directories..."
mkdir -p models input_videos output_videos stubs cache

echo ""
echo "=========================================="
echo "✅ Setup complete!"
echo "=========================================="
echo ""
echo "To start the worker, run:"
echo ""
echo "  cd football-analytics-dashboard/backend/pipeline"
echo "  source venv/bin/activate"
echo "  DASHBOARD_URL=$DASHBOARD_URL python worker.py"
echo ""
echo "The worker will:"
echo "  1. Download ML models automatically (~400MB)"
echo "  2. Poll $DASHBOARD_URL for pending analyses"
echo "  3. Process videos using GPU"
echo "  4. Upload results back to the dashboard"
echo ""
echo "To run in background:"
echo "  nohup python worker.py > worker.log 2>&1 &"
echo ""
