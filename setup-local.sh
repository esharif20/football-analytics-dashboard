#!/bin/bash
# Football Analytics Dashboard - Local Setup Script
# Run this on your Mac to get everything working

set -e

echo "=========================================="
echo "Football Analytics Dashboard - Local Setup"
echo "=========================================="
echo ""

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker Desktop for Mac:"
    echo "   https://www.docker.com/products/docker-desktop/"
    exit 1
fi

echo "✅ Docker found"

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo "❌ Docker is not running. Please start Docker Desktop."
    exit 1
fi

echo "✅ Docker is running"

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo ""
    echo "Creating .env file..."
    cat > .env << 'EOF'
# Football Analytics Dashboard - Local Configuration
# Modify these values as needed

# Optional: Add your OpenAI API key for AI commentary feature
# Leave empty to disable AI commentary
OPENAI_API_KEY=

# Optional: Add your Roboflow API key for pitch detection
# Leave empty to use fallback detection
ROBOFLOW_API_KEY=
EOF
    echo "✅ Created .env file"
else
    echo "✅ .env file already exists"
fi

echo ""
echo "=========================================="
echo "Starting services..."
echo "=========================================="
echo ""

# Build and start containers
docker compose up --build -d

echo ""
echo "=========================================="
echo "Waiting for services to be ready..."
echo "=========================================="

# Wait for the app to be ready
MAX_ATTEMPTS=30
ATTEMPT=0
while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    if curl -s http://localhost:3000 > /dev/null 2>&1; then
        break
    fi
    ATTEMPT=$((ATTEMPT + 1))
    echo "Waiting for app to start... ($ATTEMPT/$MAX_ATTEMPTS)"
    sleep 2
done

if [ $ATTEMPT -eq $MAX_ATTEMPTS ]; then
    echo ""
    echo "⚠️  App is taking longer than expected to start."
    echo "   Check logs with: docker compose logs -f"
else
    echo ""
    echo "=========================================="
    echo "✅ Setup Complete!"
    echo "=========================================="
    echo ""
    echo "Dashboard:  http://localhost:3000"
    echo "Database:   localhost:3306 (user: root, pass: football123)"
    echo ""
    echo "Useful commands:"
    echo "  docker compose logs -f     # View logs"
    echo "  docker compose down        # Stop services"
    echo "  docker compose up -d       # Start services"
    echo "  docker compose restart     # Restart services"
    echo ""
    echo "To connect RunPod worker:"
    echo "  1. Get your Mac's IP or use ngrok/cloudflare tunnel"
    echo "  2. On RunPod: export DASHBOARD_URL=http://YOUR_IP:3000"
    echo "  3. On RunPod: python worker.py"
    echo ""
fi
