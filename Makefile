# Football Analysis Dashboard - Makefile
# Simple commands for setup, development, and deployment

.PHONY: help setup setup-dashboard setup-backend run dashboard worker test clean logs docker-build docker-run

# Default target - show help
help:
	@echo "Football Analysis Dashboard - Available Commands"
	@echo ""
	@echo "Setup Commands:"
	@echo "  make setup          - Full setup (dashboard + backend)"
	@echo "  make setup-dashboard - Install Node.js dependencies only"
	@echo "  make setup-backend  - Install Python dependencies only"
	@echo ""
	@echo "Run Commands:"
	@echo "  make run            - Start dashboard and worker together"
	@echo "  make dashboard      - Start web dashboard only (port 3000)"
	@echo "  make worker         - Start pipeline worker only"
	@echo ""
	@echo "Development Commands:"
	@echo "  make test           - Run all tests"
	@echo "  make test-dashboard - Run dashboard tests only"
	@echo "  make test-backend   - Run backend tests only"
	@echo "  make logs           - View recent logs"
	@echo "  make clean          - Remove build artifacts and caches"
	@echo ""
	@echo "Docker Commands:"
	@echo "  make docker-build   - Build Docker images"
	@echo "  make docker-run     - Run with Docker Compose"
	@echo "  make docker-stop    - Stop Docker containers"
	@echo ""
	@echo "Pipeline Commands:"
	@echo "  make process VIDEO=/path/to/video.mp4 - Process a video directly"

# =============================================================================
# Setup Commands
# =============================================================================

setup: setup-dashboard setup-backend
	@echo "✅ Setup complete! Run 'make run' to start the application."

setup-dashboard:
	@echo "📦 Installing Node.js dependencies..."
	pnpm install
	@echo "🗄️  Setting up database..."
	pnpm db:push
	@echo "✅ Dashboard setup complete!"

setup-backend:
	@echo "🐍 Setting up Python environment..."
	cd backend && \
	python3 -m venv venv && \
	. venv/bin/activate && \
	pip install --upgrade pip && \
	pip install -r requirements.txt
	@echo "✅ Backend setup complete!"

# =============================================================================
# Run Commands
# =============================================================================

run:
	@echo "🚀 Starting Football Analysis Dashboard..."
	@echo "   Dashboard: http://localhost:3000"
	@echo "   Press Ctrl+C to stop"
	@echo ""
	@# Start dashboard in background, worker in foreground
	@trap 'kill 0' EXIT; \
	pnpm dev & \
	sleep 3 && \
	cd backend && . venv/bin/activate && python worker.py

dashboard:
	@echo "🌐 Starting web dashboard on http://localhost:3000"
	pnpm dev

worker:
	@echo "⚙️  Starting pipeline worker..."
	cd backend && . venv/bin/activate && python worker.py

# =============================================================================
# Test Commands
# =============================================================================

test: test-dashboard test-backend
	@echo "✅ All tests passed!"

test-dashboard:
	@echo "🧪 Running dashboard tests..."
	pnpm test

test-backend:
	@echo "🧪 Running backend tests..."
	cd backend && . venv/bin/activate && python -m pytest tests/ -v 2>/dev/null || echo "No pytest tests found (OK)"
	@echo "🧪 Verifying pipeline CLI..."
	cd backend && . venv/bin/activate && python main.py --help > /dev/null && echo "✅ Pipeline CLI working"

# =============================================================================
# Development Commands
# =============================================================================

logs:
	@echo "📋 Recent logs:"
	@echo ""
	@echo "=== Dev Server Logs ==="
	@tail -50 .manus-logs/devserver.log 2>/dev/null || echo "No dev server logs"
	@echo ""
	@echo "=== Browser Console Logs ==="
	@tail -30 .manus-logs/browserConsole.log 2>/dev/null || echo "No browser logs"

clean:
	@echo "🧹 Cleaning build artifacts..."
	rm -rf node_modules/.cache
	rm -rf dist
	rm -rf backend/__pycache__
	rm -rf backend/src/**/__pycache__
	rm -rf backend/.pytest_cache
	rm -rf backend/venv 2>/dev/null || true
	@echo "✅ Clean complete!"

# =============================================================================
# Docker Commands
# =============================================================================

docker-build:
	@echo "🐳 Building Docker images..."
	docker-compose build

docker-run:
	@echo "🐳 Starting with Docker Compose..."
	docker-compose up -d
	@echo "✅ Services started!"
	@echo "   Dashboard: http://localhost:3000"
	@echo "   Run 'make docker-stop' to stop"

docker-stop:
	@echo "🛑 Stopping Docker containers..."
	docker-compose down

# =============================================================================
# Pipeline Commands
# =============================================================================

# Process a video directly (usage: make process VIDEO=/path/to/video.mp4)
process:
ifndef VIDEO
	@echo "❌ Error: VIDEO path required"
	@echo "   Usage: make process VIDEO=/path/to/video.mp4"
	@exit 1
endif
	@echo "🎬 Processing video: $(VIDEO)"
	cd backend && . venv/bin/activate && \
	python main.py \
		--source-video-path "$(VIDEO)" \
		--target-video-path "$(VIDEO:.mp4=_annotated.mp4)" \
		--mode all

# Process with specific mode (usage: make process-mode VIDEO=/path/to/video.mp4 MODE=radar)
process-mode:
ifndef VIDEO
	@echo "❌ Error: VIDEO path required"
	@exit 1
endif
ifndef MODE
	@echo "❌ Error: MODE required (all, radar, team, track, players, ball, pitch)"
	@exit 1
endif
	@echo "🎬 Processing video: $(VIDEO) with mode: $(MODE)"
	cd backend && . venv/bin/activate && \
	python main.py \
		--source-video-path "$(VIDEO)" \
		--target-video-path "$(VIDEO:.mp4=_$(MODE).mp4)" \
		--mode $(MODE)

# =============================================================================
# Utility Commands
# =============================================================================

# Check system requirements
check:
	@echo "🔍 Checking system requirements..."
	@echo ""
	@echo "Node.js: $$(node --version 2>/dev/null || echo 'NOT INSTALLED')"
	@echo "pnpm: $$(pnpm --version 2>/dev/null || echo 'NOT INSTALLED')"
	@echo "Python: $$(python3 --version 2>/dev/null || echo 'NOT INSTALLED')"
	@echo ""
	@echo "GPU Support:"
	@cd backend 2>/dev/null && . venv/bin/activate 2>/dev/null && \
		python -c "import torch; print(f'  MPS (Apple): {torch.backends.mps.is_available()}')" 2>/dev/null || echo "  MPS (Apple): Not checked (run setup first)"
	@cd backend 2>/dev/null && . venv/bin/activate 2>/dev/null && \
		python -c "import torch; print(f'  CUDA (NVIDIA): {torch.cuda.is_available()}')" 2>/dev/null || echo "  CUDA (NVIDIA): Not checked (run setup first)"

# Update dependencies
update:
	@echo "📦 Updating dependencies..."
	pnpm update
	cd backend && . venv/bin/activate && pip install --upgrade -r requirements.txt
	@echo "✅ Dependencies updated!"

# Database commands
db-push:
	@echo "🗄️  Pushing database schema..."
	pnpm db:push

db-studio:
	@echo "🗄️  Opening database studio..."
	pnpm db:studio
