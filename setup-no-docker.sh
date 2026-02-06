#!/bin/bash
# Football Analytics Dashboard - Setup WITHOUT Docker
# For macOS - uses Homebrew to install MySQL and Node.js

set -e

echo "=========================================="
echo "Football Analytics Dashboard - No-Docker Setup"
echo "=========================================="
echo ""

# Check for Homebrew
if ! command -v brew &> /dev/null; then
    echo "Homebrew not found. Installing..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Check for Node.js
if ! command -v node &> /dev/null; then
    echo "Installing Node.js..."
    brew install node
fi
echo "Node.js: $(node --version)"

# Check for pnpm
if ! command -v pnpm &> /dev/null; then
    echo "Installing pnpm..."
    npm install -g pnpm
fi
echo "pnpm: $(pnpm --version)"

# Check for MySQL
if ! command -v mysql &> /dev/null; then
    echo "Installing MySQL..."
    brew install mysql
fi

# Start MySQL if not running
if ! mysqladmin ping -h localhost --silent 2>/dev/null; then
    echo "Starting MySQL..."
    brew services start mysql
    sleep 3
fi
echo "MySQL: running"

# Create database if it doesn't exist
mysql -u root -e "CREATE DATABASE IF NOT EXISTS football_dashboard;" 2>/dev/null || {
    echo ""
    echo "Could not connect to MySQL. Try:"
    echo "  brew services restart mysql"
    echo "  mysql -u root -e 'CREATE DATABASE football_dashboard;'"
    echo ""
    exit 1
}
echo "Database: football_dashboard (created)"

# Create .env file
if [ ! -f .env ]; then
    cat > .env << 'ENVEOF'
# Football Analytics Dashboard - Local Configuration
LOCAL_DEV_MODE=true
DATABASE_URL=mysql://root@localhost:3306/football_dashboard
JWT_SECRET=local-dev-secret-change-in-production
LOCAL_STORAGE_DIR=./uploads

# Optional: OpenAI API key for AI commentary
OPENAI_API_KEY=

# Optional: Roboflow API key for pitch detection
ROBOFLOW_API_KEY=

# Disable Manus services
BUILT_IN_FORGE_API_URL=
BUILT_IN_FORGE_API_KEY=
OAUTH_SERVER_URL=
VITE_APP_ID=
VITE_OAUTH_PORTAL_URL=
VITE_APP_TITLE=Football Analytics
VITE_ANALYTICS_ENDPOINT=
VITE_ANALYTICS_WEBSITE_ID=
ENVEOF
    echo "Created .env file"
else
    echo ".env file already exists"
fi

# Install dependencies
echo ""
echo "Installing dependencies..."
pnpm install
cd backend && pnpm install && cd ..
cd frontend && pnpm install && cd ..

# Create uploads directory
mkdir -p uploads

# Run database migrations
echo ""
echo "Running database migrations..."
cd backend && pnpm db:push && cd ..

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To start the dashboard:"
echo "  pnpm dev"
echo ""
echo "Then open: http://localhost:3000"
echo ""
echo "You are auto-logged in as 'Local Developer' (admin)"
echo ""
echo "To connect RunPod worker:"
echo "  1. Install ngrok: brew install ngrok"
echo "  2. Run: ngrok http 3000"
echo "  3. On RunPod: export DASHBOARD_URL=https://YOUR-NGROK-URL"
echo "  4. On RunPod: python worker.py"
echo ""
