#!/bin/bash
# =============================================================================
# Football Analysis Dashboard - Standalone Setup Script
# =============================================================================
# This script sets up everything you need to run the dashboard locally on Mac
# WITHOUT any Manus dependencies (OAuth, Forge, external database, etc.)
# =============================================================================

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║   🏈 Football Analysis Dashboard - Standalone Setup        ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check for Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js not found. Please install Node.js 18+ first:"
    echo "   brew install node"
    exit 1
fi

NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 18 ]; then
    echo "❌ Node.js 18+ required. Current version: $(node -v)"
    echo "   brew upgrade node"
    exit 1
fi
echo "✓ Node.js $(node -v)"

# Check for pnpm
if ! command -v pnpm &> /dev/null; then
    echo "Installing pnpm..."
    npm install -g pnpm
fi
echo "✓ pnpm $(pnpm -v)"

# Check for Python (for the CV pipeline)
if ! command -v python3 &> /dev/null; then
    echo "⚠️  Python 3 not found. The CV pipeline requires Python 3.9+"
    echo "   brew install python@3.11"
else
    echo "✓ Python $(python3 --version)"
fi

echo ""
echo "📦 Installing Node.js dependencies..."
pnpm install

# Create required directories
echo ""
echo "📁 Creating directories..."
mkdir -p data uploads backend/models backend/input_videos backend/output_videos backend/stubs

# Create .env file for standalone mode
echo ""
echo "⚙️  Creating .env file..."
cat > .env << 'EOF'
# Standalone Mode Configuration
# No external services required - everything runs locally

# Server
PORT=3000
NODE_ENV=development

# Local mode flags (these bypass Manus dependencies)
LOCAL_DEV_MODE=true
USE_LOCAL_STORAGE=true
USE_SQLITE=true

# Optional: Roboflow API key (only if you want to use Roboflow for pitch detection)
# ROBOFLOW_API_KEY=your_key_here

# Optional: Gemini API key (only if you want AI-generated commentary)
# GEMINI_API_KEY=your_key_here
EOF

echo "✓ .env file created"

# Download models if not present
echo ""
echo "🤖 Checking for ML models..."
if [ ! -f "backend/models/player_detection.pt" ]; then
    echo "⚠️  Custom models not found in backend/models/"
    echo "   Please copy your trained models to backend/models/:"
    echo "   - player_detection.pt"
    echo "   - ball_detection.pt"
    echo "   - pitch_detection.pt"
else
    echo "✓ Models found"
fi

# Setup Python environment for backend
echo ""
echo "🐍 Setting up Python environment for CV pipeline..."
cd backend

if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Created Python virtual environment"
fi

source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt 2>/dev/null || {
    echo "Installing core Python packages..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install ultralytics supervision numpy opencv-python-headless
    pip install transformers umap-learn scikit-learn
    pip install roboflow
}

deactivate
cd ..

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║   ✅ Setup Complete!                                       ║"
echo "╠════════════════════════════════════════════════════════════╣"
echo "║                                                            ║"
echo "║   To start the dashboard:                                  ║"
echo "║   $ pnpm run standalone                                    ║"
echo "║                                                            ║"
echo "║   Then open: http://localhost:3000                         ║"
echo "║                                                            ║"
echo "║   To run the CV pipeline on a video:                       ║"
echo "║   $ cd backend                                             ║"
echo "║   $ source venv/bin/activate                               ║"
echo "║   $ python main.py --source-video-path input_videos/test.mp4 --mode all║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
