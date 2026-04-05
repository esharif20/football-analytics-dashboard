#!/bin/bash
# =============================================================================
# Football Analytics - Cloud GPU Worker Setup
# =============================================================================
# Run this script ON a cloud GPU instance (RunPod, Lambda Labs, Vast.ai, etc.)
# It sets up the worker to process videos from your dashboard.
#
# Usage (on the pod):
#   curl -sSL https://raw.githubusercontent.com/esharif20/football-analytics-dashboard/main/scripts/setup-cloud-gpu.sh | bash
#
# Or copy & run manually:
#   chmod +x setup-cloud-gpu.sh && ./setup-cloud-gpu.sh [OPTIONS]
#
# Options:
#   --start         Start the worker immediately after setup (requires DASHBOARD_URL)
#   --bg            With --start: run worker in background (nohup)
#   --help          Show this help
#
# Environment variables (set before running or in your .env):
#   DASHBOARD_URL   Required. The URL to poll for jobs, e.g. https://your-ngrok.ngrok-free.dev
#   PIPELINE_SUBPROCESS   Set to 1 automatically by this script
#   PITCH_MODEL_BACKEND   Set to ultralytics automatically by this script
# =============================================================================

set -e

START_WORKER=false
RUN_BG=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --start) START_WORKER=true; shift ;;
    --bg)    RUN_BG=true; shift ;;
    --help)
      sed -n '/^# Usage/,/^# ====/p' "$0" | head -n -1
      exit 0 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

echo "=========================================="
echo "Football Analytics - Cloud GPU Worker"
echo "=========================================="

# ── Validate DASHBOARD_URL ────────────────────────────────────────────────────
DASHBOARD_URL="${DASHBOARD_URL:-}"
if [[ -z "$DASHBOARD_URL" ]]; then
  if $START_WORKER; then
    echo "Error: DASHBOARD_URL is required when using --start"
    echo "  export DASHBOARD_URL=https://your-ngrok-url.ngrok-free.dev"
    exit 1
  else
    echo "⚠  DASHBOARD_URL not set — you'll need to set it before starting the worker"
  fi
else
  echo "Dashboard URL: $DASHBOARD_URL"
fi
echo ""

# ── GPU check ─────────────────────────────────────────────────────────────────
if command -v nvidia-smi &> /dev/null; then
    echo "✅ GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠  No GPU detected - will run on CPU (slower)"
fi
echo ""

# ── Determine install dir ─────────────────────────────────────────────────────
# If we're already inside the repo, use it. Otherwise clone/pull.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || echo '')"
if [[ -f "$SCRIPT_DIR/../backend/pipeline/worker.py" ]]; then
  REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
  echo "✅ Repo already present at $REPO_ROOT"
else
  REPO_URL="https://github.com/esharif20/football-analytics-dashboard.git"
  echo "Step 1: Cloning repository..."
  if [[ -d "football-analytics-dashboard" ]]; then
    echo "  Repository already exists, pulling latest..."
    cd football-analytics-dashboard
    git pull
  else
    git clone --depth 1 "$REPO_URL"
    cd football-analytics-dashboard
  fi
  REPO_ROOT="$(pwd)"
fi

PIPELINE_DIR="$REPO_ROOT/backend/pipeline"

# ── System deps ───────────────────────────────────────────────────────────────
echo ""
echo "Step 2: Installing system dependencies..."
apt-get update -qq 2>/dev/null || true
apt-get install -y -qq python3-pip python3-venv git ffmpeg libgl1-mesa-glx libglib2.0-0 2>/dev/null || true

# ── Python venv ───────────────────────────────────────────────────────────────
echo ""
echo "Step 3: Setting up Python environment..."
cd "$PIPELINE_DIR"
python3 -m venv venv
source venv/bin/activate

# ── Python deps ───────────────────────────────────────────────────────────────
echo ""
echo "Step 4: Installing Python dependencies..."
pip install --upgrade pip -q

pip install -r requirements.txt -q

# Extra deps not in base image that the pipeline actually needs
echo "  Installing extra required packages..."
pip install umap-learn sentencepiece protobuf -q

# timm is required for the ML event detection module (EfficientNetV2-B0)
echo "  Installing timm for ML event detection..."
pip install timm -q

# ── Directories ───────────────────────────────────────────────────────────────
echo ""
echo "Step 5: Creating directories..."
mkdir -p models input_videos output_videos stubs cache

# ── Create restart_worker.sh on the pod ───────────────────────────────────────
echo ""
echo "Step 6: Writing restart_worker.sh helper..."
cat > "$PIPELINE_DIR/restart_worker.sh" <<'RESTART_EOF'
#!/bin/bash
# Kill existing worker and restart it
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXISTING=$(pgrep -f "python.*worker.py" 2>/dev/null || true)
if [[ -n "$EXISTING" ]]; then
  echo "Stopping existing worker (PID: $EXISTING)..."
  kill "$EXISTING" 2>/dev/null || true
  sleep 2
fi

source "$SCRIPT_DIR/venv/bin/activate"

export DASHBOARD_URL="${DASHBOARD_URL:?'DASHBOARD_URL must be set'}"
export PIPELINE_SUBPROCESS=1
export PITCH_MODEL_BACKEND=ultralytics
export MODEL_URL_PLAYER="${MODEL_URL_PLAYER:-skip}"
export MODEL_URL_BALL="${MODEL_URL_BALL:-skip}"
export MODEL_URL_PITCH="${MODEL_URL_PITCH:-skip}"

echo "Starting worker → $DASHBOARD_URL"
cd "$SCRIPT_DIR"
nohup python worker.py > worker.log 2>&1 &
echo "Worker PID: $! — tail -f $SCRIPT_DIR/worker.log"
RESTART_EOF
chmod +x "$PIPELINE_DIR/restart_worker.sh"

echo "  ✅ restart_worker.sh created at $PIPELINE_DIR/restart_worker.sh"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "✅ Setup complete!"
echo "=========================================="
echo ""
echo "Pipeline dir : $PIPELINE_DIR"
echo "Python env   : $PIPELINE_DIR/venv"
echo ""
echo "Required env vars before starting worker:"
echo "  export DASHBOARD_URL=https://your-ngrok-url.ngrok-free.dev"
echo "  export PIPELINE_SUBPROCESS=1"
echo "  export PITCH_MODEL_BACKEND=ultralytics"
echo ""
echo "Quick start options:"
echo "  # Foreground (see logs live):"
echo "  cd $PIPELINE_DIR && source venv/bin/activate"
echo "  DASHBOARD_URL=<url> PIPELINE_SUBPROCESS=1 PITCH_MODEL_BACKEND=ultralytics python worker.py"
echo ""
echo "  # Background (via helper):"
echo "  DASHBOARD_URL=<url> bash $PIPELINE_DIR/restart_worker.sh"
echo ""

# ── Auto-start ────────────────────────────────────────────────────────────────
if $START_WORKER; then
  if [[ -z "$DASHBOARD_URL" ]]; then
    echo "Error: DASHBOARD_URL must be set to use --start"
    exit 1
  fi
  export PIPELINE_SUBPROCESS=1
  export PITCH_MODEL_BACKEND=ultralytics
  export MODEL_URL_PLAYER="${MODEL_URL_PLAYER:-skip}"
  export MODEL_URL_BALL="${MODEL_URL_BALL:-skip}"
  export MODEL_URL_PITCH="${MODEL_URL_PITCH:-skip}"

  echo "Starting worker (DASHBOARD_URL=$DASHBOARD_URL)..."
  cd "$PIPELINE_DIR"
  source venv/bin/activate

  if $RUN_BG; then
    nohup python worker.py > worker.log 2>&1 &
    echo "✅ Worker started in background (PID: $!)"
    echo "   Logs: $PIPELINE_DIR/worker.log"
    echo "   tail -f $PIPELINE_DIR/worker.log"
  else
    python worker.py
  fi
fi
