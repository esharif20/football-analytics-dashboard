#!/bin/bash
# =============================================================================
# Football Analysis Dashboard - Mac Setup Script
# =============================================================================
# This script sets up everything you need to run the dashboard locally on Mac.
# Usage: ./setup-mac.sh
# =============================================================================

set -e

echo ""
echo "🏈 Football Analysis Dashboard - Mac Setup"
echo "==========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check for required tools
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}❌ $1 is not installed.${NC}"
        echo "   Install with: $2"
        exit 1
    else
        echo -e "${GREEN}✓${NC} $1 found"
    fi
}

echo "Checking prerequisites..."
echo ""

check_command "node" "brew install node"
check_command "pnpm" "npm install -g pnpm"
check_command "python3" "brew install python@3.11"

echo ""
echo "All prerequisites found!"
echo ""

# =============================================================================
# Step 1: Install Node.js dependencies
# =============================================================================
echo "📦 Step 1: Installing Node.js dependencies..."
pnpm install
echo -e "${GREEN}✓${NC} Node.js dependencies installed"
echo ""

# =============================================================================
# Step 2: Create .env file for local development
# =============================================================================
echo "⚙️  Step 2: Setting up environment..."

if [ ! -f .env ]; then
    cat > .env << 'EOF'
# Local Development Mode
# This bypasses Manus OAuth and uses local file storage
LOCAL_DEV_MODE=true

# Database (SQLite for simplest setup, or MySQL)
# For SQLite (no setup needed):
# DATABASE_URL="file:./local.db"
# For MySQL:
DATABASE_URL="mysql://root:@localhost:3306/football"

# JWT Secret (any random string for local dev)
JWT_SECRET="local-dev-secret-change-in-production"

# Optional: Local storage directory
LOCAL_STORAGE_DIR="./uploads"
EOF
    echo -e "${GREEN}✓${NC} Created .env file with local dev settings"
else
    echo -e "${YELLOW}⚠${NC} .env file already exists, skipping"
fi
echo ""

# =============================================================================
# Step 3: Set up Python environment for CV pipeline
# =============================================================================
echo "🐍 Step 3: Setting up Python CV pipeline..."
cd backend

if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓${NC} Created Python virtual environment"
else
    echo -e "${YELLOW}⚠${NC} Virtual environment already exists"
fi

source venv/bin/activate

echo "Installing Python dependencies..."
pip install --upgrade pip > /dev/null 2>&1

# Install PyTorch with MPS support for Mac
if [[ $(uname -m) == 'arm64' ]]; then
    echo "   Detected Apple Silicon - installing PyTorch with MPS support..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu > /dev/null 2>&1
else
    echo "   Detected Intel Mac - installing PyTorch CPU..."
    pip install torch torchvision > /dev/null 2>&1
fi

pip install -r requirements.txt > /dev/null 2>&1
echo -e "${GREEN}✓${NC} Python dependencies installed"

cd ..
echo ""

# =============================================================================
# Step 4: Create directories
# =============================================================================
echo "📁 Step 4: Creating directories..."
mkdir -p uploads
mkdir -p backend/models
mkdir -p backend/input_videos
mkdir -p backend/output_videos
mkdir -p backend/stubs
echo -e "${GREEN}✓${NC} Directories created"
echo ""

# =============================================================================
# Step 5: Download models (if not present)
# =============================================================================
echo "🤖 Step 5: Checking for models..."
cd backend

MODELS_NEEDED=false
if [ ! -f "models/player_detection.pt" ]; then
    echo -e "${YELLOW}⚠${NC} player_detection.pt not found"
    MODELS_NEEDED=true
fi
if [ ! -f "models/ball_detection.pt" ]; then
    echo -e "${YELLOW}⚠${NC} ball_detection.pt not found"
    MODELS_NEEDED=true
fi
if [ ! -f "models/pitch_detection.pt" ]; then
    echo -e "${YELLOW}⚠${NC} pitch_detection.pt not found"
    MODELS_NEEDED=true
fi

if [ "$MODELS_NEEDED" = true ]; then
    echo ""
    echo "   Models need to be downloaded. You can either:"
    echo "   1. Copy your .pt files to backend/models/"
    echo "   2. Run: cd backend && ./setup.sh (downloads from Google Drive)"
    echo ""
else
    echo -e "${GREEN}✓${NC} All models found"
fi

cd ..
echo ""

# =============================================================================
# Done!
# =============================================================================
echo "==========================================="
echo -e "${GREEN}✅ Setup complete!${NC}"
echo "==========================================="
echo ""
echo "To start the dashboard:"
echo "  ${GREEN}pnpm dev${NC}"
echo ""
echo "To run the CV pipeline:"
echo "  ${GREEN}cd backend${NC}"
echo "  ${GREEN}source venv/bin/activate${NC}"
echo "  ${GREEN}python main.py --source-video-path input_videos/test.mp4 --target-video-path output.mp4 --mode all${NC}"
echo ""
echo "Dashboard will be available at: ${GREEN}http://localhost:3000${NC}"
echo ""
