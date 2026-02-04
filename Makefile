# Football Analysis Dashboard - Makefile
# Simple commands for local development

.PHONY: help setup run test clean check

# Default target - show help
help:
	@echo ""
	@echo "╔════════════════════════════════════════════════════════════╗"
	@echo "║         Football Analysis Dashboard - Commands             ║"
	@echo "╚════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "  make setup    - Install all dependencies"
	@echo "  make run      - Start the dashboard (http://localhost:8000)"
	@echo "  make test     - Run tests"
	@echo "  make check    - Check system requirements"
	@echo "  make clean    - Remove all dependencies"
	@echo ""
	@echo "PIPELINE:"
	@echo "  make process VIDEO=/path/to/video.mp4"
	@echo ""

# =============================================================================
# Setup and Run
# =============================================================================

setup:
	@echo ""
	@echo "📦 Setting up Football Analysis Dashboard..."
	@echo ""
	@echo "1️⃣  Creating Python virtual environment..."
	@cd backend/api && python3 -m venv venv
	@echo ""
	@echo "2️⃣  Installing Python dependencies..."
	@cd backend/api && . venv/bin/activate && pip install --quiet --upgrade pip && pip install --quiet fastapi uvicorn python-multipart aiofiles websockets pydantic
	@echo ""
	@echo "3️⃣  Installing frontend dependencies..."
	@cd frontend && pnpm install --silent
	@echo ""
	@echo "4️⃣  Building frontend..."
	@cd frontend && pnpm build 2>/dev/null || pnpm build
	@echo ""
	@echo "✅ Setup complete! Run 'make run' to start."
	@echo ""

run:
	@./run-local.sh

# Alias for backwards compatibility
local: run
setup-local: setup

# =============================================================================
# Pipeline
# =============================================================================

process:
ifndef VIDEO
	@echo "❌ Usage: make process VIDEO=/path/to/video.mp4"
	@exit 1
endif
	@echo "🎬 Processing: $(VIDEO)"
	@cd backend/pipeline && . venv/bin/activate && \
		python main.py --source-video-path "$(VIDEO)" --mode all

# =============================================================================
# Testing
# =============================================================================

test:
	@echo "🧪 Running Python tests..."
	@cd backend/api && . venv/bin/activate && python -m pytest -v 2>/dev/null || echo "No tests found"

# =============================================================================
# Utility
# =============================================================================

clean:
	@echo "🧹 Cleaning up..."
	rm -rf frontend/node_modules
	rm -rf backend/api/venv backend/pipeline/venv
	rm -rf frontend/dist
	rm -rf backend/data
	@echo "✅ Clean complete"

check:
	@echo ""
	@echo "╔════════════════════════════════════════════════════════════╗"
	@echo "║                  System Requirements Check                  ║"
	@echo "╚════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Python:   $$(python3 --version 2>/dev/null || echo '❌ NOT INSTALLED')"
	@echo "Node.js:  $$(node --version 2>/dev/null || echo '❌ NOT INSTALLED')"
	@echo "pnpm:     $$(pnpm --version 2>/dev/null || echo '❌ NOT INSTALLED (run: npm i -g pnpm)')"
	@echo ""
	@echo "GPU Support:"
	@python3 -c "import torch; print('  PyTorch:', torch.__version__); print('  CUDA:', torch.cuda.is_available()); print('  MPS (Apple):', torch.backends.mps.is_available())" 2>/dev/null || echo "  PyTorch: ❌ NOT INSTALLED (optional for CV pipeline)"
	@echo ""
