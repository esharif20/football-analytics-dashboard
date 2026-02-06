#!/bin/bash
# Football Analytics Dashboard - Local Setup Script
# Works on macOS with Docker Desktop

set -e

echo "=========================================="
echo "Football Analytics Dashboard - Local Setup"
echo "=========================================="
echo ""

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo "Docker not found. Please install Docker Desktop for Mac:"
    echo "   https://www.docker.com/products/docker-desktop/"
    echo ""
    echo "Or use the non-Docker setup instead:"
    echo "   ./setup-no-docker.sh"
    exit 1
fi

echo "Docker found"

# Check if Docker is running
if ! docker info &> /dev/null 2>&1; then
    echo "Docker is not running. Please start Docker Desktop."
    echo "Opening Docker Desktop..."
    open -a Docker 2>/dev/null || true
    echo "Wait for Docker to start, then re-run this script."
    exit 1
fi

echo "Docker is running"

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo ""
    echo "Creating .env file..."
    cp .env.example .env 2>/dev/null || cat > .env << 'ENVEOF'
# Football Analytics Dashboard - Local Configuration

# Optional: OpenAI API key for AI commentary (leave empty to disable)
OPENAI_API_KEY=

# Optional: Roboflow API key for pitch detection
ROBOFLOW_API_KEY=
ENVEOF
    echo "Created .env file (edit to add API keys)"
else
    echo ".env file already exists"
fi

echo ""
echo "=========================================="
echo "Starting services with Docker Compose..."
echo "=========================================="
echo ""

# Clean up any stale containers
docker compose down 2>/dev/null || true

# Build and start
docker compose up --build -d

echo ""
echo "Waiting for services to be ready..."

# Wait for the app to be ready
MAX_ATTEMPTS=60
ATTEMPT=0
while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    if curl -s http://localhost:3000 > /dev/null 2>&1; then
        break
    fi
    ATTEMPT=$((ATTEMPT + 1))
    if [ $((ATTEMPT % 5)) -eq 0 ]; then
        echo "  Still waiting... ($ATTEMPT/$MAX_ATTEMPTS)"
    fi
    sleep 2
done

if [ $ATTEMPT -eq $MAX_ATTEMPTS ]; then
    echo ""
    echo "App is taking longer than expected."
    echo "Check logs with: docker compose logs -f app"
    echo ""
    echo "Common fixes:"
    echo "  docker compose down -v && docker compose up --build"
else
    echo ""
    echo "=========================================="
    echo "Setup Complete!"
    echo "=========================================="
    echo ""
    echo "  Dashboard:  http://localhost:3000"
    echo "  Database:   localhost:3307 (user: root, pass: football123)"
    echo ""
    echo "  You are auto-logged in as 'Local Developer' (admin)"
    echo ""
    echo "Useful commands:"
    echo "  docker compose logs -f app   # View app logs"
    echo "  docker compose down          # Stop services"
    echo "  docker compose up -d         # Start services"
    echo "  docker compose restart app   # Restart app only"
    echo ""
    echo "To connect RunPod worker:"
    echo "  1. Install ngrok: brew install ngrok"
    echo "  2. Run: ngrok http 3000"
    echo "  3. On RunPod: export DASHBOARD_URL=https://YOUR-NGROK-URL"
    echo "  4. On RunPod: python worker.py"
    echo ""
fi
