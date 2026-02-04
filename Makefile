# Football Analysis Dashboard - Makefile
# Simple commands for setup, development, and deployment

.PHONY: help setup run local test clean

# Default target - show help
help:
	@echo "╔════════════════════════════════════════════════════════════╗"
	@echo "║         Football Analysis Dashboard - Commands             ║"
	@echo "╚════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "LOCAL DEVELOPMENT (Recommended - Pure FastAPI, no Node.js):"
	@echo "  make local     - Run with FastAPI backend (http://localhost:8000)"
	@echo "  make setup-local - Install Python + frontend dependencies"
	@echo ""
	@echo "MANUS DEVELOPMENT (Node.js + tRPC):"
	@echo "  make setup     - Install all dependencies (Node.js + Python)"
	@echo "  make run       - Start full app with Node.js backend"
	@echo ""
	@echo "PIPELINE:"
	@echo "  make process VIDEO=/path/to/video.mp4"
	@echo ""
	@echo "OTHER:"
	@echo "  make test      - Run all tests"
	@echo "  make check     - Check system requirements"
	@echo "  make clean     - Remove all dependencies"

# =============================================================================
# LOCAL DEVELOPMENT (Pure FastAPI - Recommended for your laptop)
# =============================================================================

local:
	@./run-local.sh

setup-local:
	@echo "📦 Setting up for local development (FastAPI)..."
	@echo ""
	@echo "1️⃣  Creating Python virtual environment..."
	cd backend/api && python3 -m venv venv
	@echo ""
	@echo "2️⃣  Installing Python dependencies..."
	cd backend/api && . venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt
	@echo ""
	@echo "3️⃣  Installing CV pipeline dependencies..."
	cd backend/pipeline && python3 -m venv venv && . venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt 2>/dev/null || true
	@echo ""
	@echo "4️⃣  Installing frontend dependencies..."
	cd frontend && pnpm install
	@echo ""
	@echo "5️⃣  Building frontend..."
	cd frontend && pnpm build
	@echo ""
	@echo "✅ Setup complete! Run 'make local' to start."

# =============================================================================
# MANUS DEVELOPMENT (Node.js + tRPC)
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
	cd backend/api && . venv/bin/activate && PYTHONPATH=.. uvicorn main:app --reload --port 8000

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
	rm -rf backend/pipeline/venv backend/api/venv
	rm -rf dist backend/data

check:
	@echo "╔════════════════════════════════════════════════════════════╗"
	@echo "║                  System Requirements Check                  ║"
	@echo "╚════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Node.js:  $$(node --version 2>/dev/null || echo '❌ NOT INSTALLED')"
	@echo "Python:   $$(python3 --version 2>/dev/null || echo '❌ NOT INSTALLED')"
	@echo "pnpm:     $$(pnpm --version 2>/dev/null || echo '❌ NOT INSTALLED (run: npm i -g pnpm)')"
	@echo ""
	@echo "GPU Support:"
	@python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('MPS:', torch.backends.mps.is_available())" 2>/dev/null || echo "PyTorch: ❌ NOT INSTALLED"
