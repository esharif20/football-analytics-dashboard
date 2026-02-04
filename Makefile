# Football Analysis Dashboard - Makefile
# Simple commands for setup, development, and deployment

.PHONY: help setup run test clean

# Default target - show help
help:
	@echo "Football Analysis Dashboard"
	@echo ""
	@echo "Setup:"
	@echo "  make setup     - Install all dependencies"
	@echo ""
	@echo "Run:"
	@echo "  make run       - Start full app (backend + frontend)"
	@echo "  make backend   - Start backend only (port 3000)"
	@echo "  make frontend  - Start frontend only (port 5173)"
	@echo "  make api       - Start FastAPI pipeline (port 8000)"
	@echo ""
	@echo "Test:"
	@echo "  make test      - Run all tests"
	@echo ""
	@echo "Pipeline:"
	@echo "  make process VIDEO=/path/to/video.mp4"

# =============================================================================
# Setup
# =============================================================================

setup:
	@echo "📦 Installing frontend dependencies..."
	cd frontend && pnpm install
	@echo "📦 Installing backend dependencies..."
	cd backend && pnpm install
	@echo "🐍 Setting up Python pipeline..."
	cd backend/pipeline && python3 -m venv venv && \
		. venv/bin/activate && \
		pip install --upgrade pip && \
		pip install -r requirements.txt
	@echo "✅ Setup complete! Run 'make run' to start."

# =============================================================================
# Run
# =============================================================================

run:
	@echo "🚀 Starting Football Analysis Dashboard..."
	@echo "   Backend: http://localhost:3000"
	@echo "   API Docs: http://localhost:8000/docs"
	@echo ""
	cd backend && pnpm dev

backend:
	@echo "🌐 Starting backend on http://localhost:3000"
	cd backend && pnpm dev

frontend:
	@echo "🎨 Starting frontend on http://localhost:5173"
	cd frontend && pnpm dev

api:
	@echo "🔌 Starting FastAPI on http://localhost:8000"
	cd backend/pipeline && . venv/bin/activate && python -m api.server

# =============================================================================
# Test
# =============================================================================

test:
	@echo "🧪 Running tests..."
	cd backend && pnpm test

# =============================================================================
# Pipeline
# =============================================================================

process:
ifndef VIDEO
	@echo "❌ Usage: make process VIDEO=/path/to/video.mp4"
	@exit 1
endif
	@echo "🎬 Processing: $(VIDEO)"
	cd backend/pipeline && . venv/bin/activate && \
		python main.py --source-video-path "$(VIDEO)" --mode all

# =============================================================================
# Utility
# =============================================================================

clean:
	rm -rf frontend/node_modules backend/node_modules
	rm -rf backend/pipeline/venv
	rm -rf dist

check:
	@echo "Node.js: $$(node --version 2>/dev/null || echo 'NOT INSTALLED')"
	@echo "Python: $$(python3 --version 2>/dev/null || echo 'NOT INSTALLED')"
	@echo "pnpm: $$(pnpm --version 2>/dev/null || echo 'NOT INSTALLED')"
