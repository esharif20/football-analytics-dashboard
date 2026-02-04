#!/bin/bash
# Football Analysis Dashboard - Local Development Script
# This script runs the application locally using FastAPI (no Node.js required)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     Football Analysis Dashboard - Local Development        ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check Python version
echo -e "${YELLOW}Checking Python...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is required but not installed.${NC}"
    echo "Please install Python 3.9+ from https://www.python.org/downloads/"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo -e "${GREEN}✓ Python $PYTHON_VERSION found${NC}"

# Check Node.js (for frontend build only)
echo -e "${YELLOW}Checking Node.js...${NC}"
if ! command -v node &> /dev/null; then
    echo -e "${RED}Error: Node.js is required for frontend build.${NC}"
    echo "Please install Node.js 18+ from https://nodejs.org/"
    exit 1
fi

NODE_VERSION=$(node -v)
echo -e "${GREEN}✓ Node.js $NODE_VERSION found${NC}"

# Check pnpm
if ! command -v pnpm &> /dev/null; then
    echo -e "${YELLOW}Installing pnpm...${NC}"
    npm install -g pnpm
fi
echo -e "${GREEN}✓ pnpm found${NC}"

# Project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Setup Python virtual environment
echo ""
echo -e "${YELLOW}Setting up Python environment...${NC}"
if [ ! -d "backend/api/venv" ]; then
    python3 -m venv backend/api/venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Activate virtual environment
source backend/api/venv/bin/activate

# Install Python dependencies
echo -e "${YELLOW}Installing Python dependencies...${NC}"
pip install --quiet --upgrade pip
pip install --quiet fastapi uvicorn python-multipart aiofiles websockets pydantic

# Install CV pipeline dependencies if requirements.txt exists
if [ -f "backend/pipeline/requirements.txt" ]; then
    echo -e "${YELLOW}Installing CV pipeline dependencies...${NC}"
    pip install --quiet -r backend/pipeline/requirements.txt 2>/dev/null || true
fi

echo -e "${GREEN}✓ Python dependencies installed${NC}"

# Setup frontend
echo ""
echo -e "${YELLOW}Setting up frontend...${NC}"
cd frontend

if [ ! -d "node_modules" ]; then
    pnpm install --silent
    echo -e "${GREEN}✓ Frontend dependencies installed${NC}"
else
    echo -e "${GREEN}✓ Frontend dependencies already installed${NC}"
fi

# Build frontend for production
echo -e "${YELLOW}Building frontend...${NC}"
pnpm build --silent 2>/dev/null || pnpm build

cd "$PROJECT_ROOT"
echo -e "${GREEN}✓ Frontend built${NC}"

# Create data directories
mkdir -p backend/data/uploads
mkdir -p backend/data

# Initialize database
echo ""
echo -e "${YELLOW}Initializing database...${NC}"
cd backend/api
python3 -c "from services.database import init_db; init_db()" 2>/dev/null || {
    cd "$PROJECT_ROOT"
    PYTHONPATH="$PROJECT_ROOT/backend" python3 -c "from api.services.database import init_db; init_db()"
}
cd "$PROJECT_ROOT"
echo -e "${GREEN}✓ Database initialized${NC}"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down...${NC}"
    kill $(jobs -p) 2>/dev/null
    exit 0
}
trap cleanup SIGINT SIGTERM

# Start the application
echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    Starting Application                     ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}Dashboard:${NC}  http://localhost:8000"
echo -e "${BLUE}API Docs:${NC}   http://localhost:8000/docs"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
echo ""

# Start FastAPI server
cd "$PROJECT_ROOT/backend"
PYTHONPATH="$PROJECT_ROOT/backend" uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
