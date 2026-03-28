#!/usr/bin/env bash
# start-test-stack.sh — Start DB + FastAPI + Vite for Playwright E2E tests
# Usage: bash scripts/start-test-stack.sh
# Stop:  kill $(cat /tmp/api-server.pid) $(cat /tmp/vite-server.pid)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

warn() { echo "[WARN] $*" >&2; }
info() { echo "[INFO] $*"; }
die()  { echo "[ERROR] $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 1. Start MySQL via Docker Compose (detached)
# ---------------------------------------------------------------------------
info "Starting MySQL container..."

if docker compose -f "$REPO_ROOT/docker-compose.yml" ps db 2>/dev/null | grep -q "healthy"; then
  warn "MySQL container already healthy — skipping start"
else
  docker compose -f "$REPO_ROOT/docker-compose.yml" up db -d

  info "Waiting for MySQL to be healthy (max 30s)..."
  deadline=$(( $(date +%s) + 30 ))
  while true; do
    if docker compose -f "$REPO_ROOT/docker-compose.yml" ps db 2>/dev/null | grep -q "healthy"; then
      info "MySQL is healthy."
      break
    fi
    if (( $(date +%s) >= deadline )); then
      die "MySQL did not become healthy within 30s. Check: docker compose logs db"
    fi
    sleep 2
  done
fi

# ---------------------------------------------------------------------------
# 2. Start FastAPI backend
# ---------------------------------------------------------------------------
info "Checking if FastAPI is already running on :8000..."
if curl -sf http://localhost:8000/docs > /dev/null 2>&1; then
  warn "FastAPI already responding on :8000 — skipping start"
else
  info "Starting FastAPI backend with ENABLE_TEST_SUPPORT=true..."

  # Load .env if present for DB credentials
  ENV_FILE="$REPO_ROOT/.env"
  if [ -f "$ENV_FILE" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
  fi

  (
    cd "$REPO_ROOT/backend"
    ENABLE_TEST_SUPPORT=true LOCAL_DEV_MODE=true \
      uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload \
      > /tmp/api-server.log 2>&1
  ) &
  echo $! > /tmp/api-server.pid
  info "FastAPI PID: $(cat /tmp/api-server.pid)"

  info "Waiting for FastAPI to respond (max 30s)..."
  deadline=$(( $(date +%s) + 30 ))
  while true; do
    if curl -sf http://localhost:8000/docs > /dev/null 2>&1 || \
       curl -sf http://localhost:8000/api/health > /dev/null 2>&1; then
      info "FastAPI is ready."
      break
    fi
    if (( $(date +%s) >= deadline )); then
      echo "--- FastAPI startup log ---"
      cat /tmp/api-server.log
      die "FastAPI did not respond within 30s."
    fi
    sleep 2
  done
fi

# ---------------------------------------------------------------------------
# 3. Start Vite frontend
# ---------------------------------------------------------------------------
info "Checking if Vite is already running on :5173..."
if curl -sf http://localhost:5173 > /dev/null 2>&1; then
  warn "Vite already responding on :5173 — skipping start"
else
  info "Starting Vite frontend..."

  (
    cd "$REPO_ROOT/frontend"
    pnpm dev > /tmp/vite-server.log 2>&1
  ) &
  echo $! > /tmp/vite-server.pid
  info "Vite PID: $(cat /tmp/vite-server.pid)"

  info "Waiting for Vite to respond (max 30s)..."
  deadline=$(( $(date +%s) + 30 ))
  while true; do
    if curl -sf http://localhost:5173 > /dev/null 2>&1; then
      info "Vite is ready."
      break
    fi
    if (( $(date +%s) >= deadline )); then
      echo "--- Vite startup log ---"
      cat /tmp/vite-server.log
      die "Vite did not respond within 30s."
    fi
    sleep 2
  done
fi

# ---------------------------------------------------------------------------
# 4. Success banner
# ---------------------------------------------------------------------------
cat <<'BANNER'

Stack ready:
  DB:       localhost:3307 (MySQL)
  Backend:  http://localhost:8000
  Frontend: http://localhost:5173

Run tests: pnpm exec playwright test
Stop servers: kill $(cat /tmp/api-server.pid) $(cat /tmp/vite-server.pid)
BANNER
