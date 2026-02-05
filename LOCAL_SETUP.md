# Football Analytics Dashboard - Local Setup Guide

This guide will get the Football Analytics Dashboard running on your Mac in under 5 minutes.

---

## Prerequisites

You only need **Docker Desktop** installed on your Mac. Download it from:
https://www.docker.com/products/docker-desktop/

---

## Quick Start (Recommended)

```bash
# Clone the repository
git clone https://github.com/esharif20/football-analytics-dashboard.git
cd football-analytics-dashboard

# Run the setup script
./setup-local.sh
```

That's it! The dashboard will be available at **http://localhost:3000**

---

## Manual Setup

If you prefer to run commands manually:

```bash
# Clone the repository
git clone https://github.com/esharif20/football-analytics-dashboard.git
cd football-analytics-dashboard

# Start all services
docker compose up --build

# Dashboard available at http://localhost:3000
```

---

## What Gets Started

| Service | Port | Description |
|---------|------|-------------|
| Dashboard | 3000 | Web interface for uploading and viewing analyses |
| MySQL | 3306 | Database (user: root, password: football123) |

---

## Optional: Enable AI Commentary

To enable AI-generated tactical commentary, add your OpenAI API key:

1. Create a `.env` file in the project root:
```bash
OPENAI_API_KEY=sk-your-openai-key-here
```

2. Restart the services:
```bash
docker compose restart
```

---

## Connecting the RunPod Worker

The CV pipeline worker runs on RunPod with GPU. To connect it to your local dashboard:

### Option 1: Using ngrok (Easiest)

```bash
# On your Mac, install and run ngrok
brew install ngrok
ngrok http 3000

# Copy the https URL (e.g., https://abc123.ngrok.io)
```

Then on RunPod:
```bash
cd /football-analytics-dashboard/backend/pipeline
export DASHBOARD_URL=https://abc123.ngrok.io
export ROBOFLOW_API_KEY=your-key
python worker.py
```

### Option 2: Direct IP (If on same network)

```bash
# Find your Mac's IP
ipconfig getifaddr en0

# On RunPod
export DASHBOARD_URL=http://YOUR_MAC_IP:3000
python worker.py
```

### Option 3: Cloudflare Tunnel (Production-ready)

```bash
# Install cloudflared
brew install cloudflared

# Create tunnel
cloudflared tunnel --url http://localhost:3000
```

---

## Making Code Changes

The Docker setup mounts your source code, so changes are reflected immediately:

1. Edit files in `client/` for frontend changes
2. Edit files in `backend/` for backend changes
3. Changes hot-reload automatically (no restart needed)

For database schema changes:
```bash
# Edit backend/drizzle/schema.ts, then:
docker compose exec app sh -c "cd /app/backend && pnpm db:push"
```

---

## Useful Commands

```bash
# View logs
docker compose logs -f

# View only app logs
docker compose logs -f app

# Stop all services
docker compose down

# Stop and remove all data (fresh start)
docker compose down -v

# Restart services
docker compose restart

# Rebuild after major changes
docker compose up --build
```

---

## Database Access

Connect to the MySQL database:

```bash
# Using Docker
docker compose exec db mysql -uroot -pfootball123 football_dashboard

# Using any MySQL client
Host: localhost
Port: 3306
User: root
Password: football123
Database: football_dashboard
```

---

## Troubleshooting

### "Port 3000 already in use"
```bash
# Find and kill the process
lsof -i :3000
kill -9 <PID>
```

### "Cannot connect to Docker daemon"
Make sure Docker Desktop is running (check the whale icon in menu bar).

### "Database connection failed"
Wait a few seconds - MySQL takes time to initialize on first run.
```bash
# Check if MySQL is ready
docker compose logs db
```

### Worker can't connect to dashboard
1. Make sure the dashboard is accessible from the internet (use ngrok)
2. Check that `DASHBOARD_URL` is set correctly on RunPod
3. Verify no firewall is blocking the connection

---

## File Structure

```
football-analytics-dashboard/
├── client/                 # React frontend
├── backend/
│   ├── server/            # Express + tRPC backend
│   ├── drizzle/           # Database schema
│   └── pipeline/          # CV pipeline (runs on RunPod)
├── docker-compose.yml     # Docker configuration
├── Dockerfile.local       # Docker build instructions
├── setup-local.sh         # Quick setup script
└── LOCAL_SETUP.md         # This file
```

---

## Next Steps

1. Upload a test video through the dashboard
2. Connect your RunPod worker
3. Watch the analysis complete
4. Explore the code and make modifications!

For more details on the architecture and service replacements, see `MIGRATION_GUIDE.md`.
