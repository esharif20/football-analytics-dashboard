#!/usr/bin/env bash
# =============================================================================
# Football Analytics — Sync local pipeline code to RunPod (or any SSH pod)
# =============================================================================
# Run this LOCALLY to push unpushed pipeline changes to a running pod.
# Reads SSH config from .env (same keys as pipeline-e2e.sh).
#
# Usage:
#   bash scripts/sync-to-pod.sh [OPTIONS]
#
# Options:
#   --restart    Kill and restart the worker on the pod after sync
#   --dry-run    Show what would be synced without actually copying
#   --help       Show this help
#
# Required in .env (or exported before running):
#   RUNPOD_SSH_HOST      e.g. 107.150.186.62
#   RUNPOD_SSH_PORT      e.g. 12610
#   RUNPOD_SSH_KEY       e.g. ~/.runpod/ssh/RunPod-Key-Go
#   RUNPOD_WORKER_DIR    e.g. /workspace/pipeline  (pipeline dir on pod)
#   DASHBOARD_URL        e.g. https://marcie-nonchurchgoing-sonya.ngrok-free.dev
# =============================================================================

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ── Colours ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'
CYAN='\033[0;36m'; BOLD='\033[1m'; DIM='\033[2m'; RESET='\033[0m'

info()    { echo -e "  ${DIM}→${RESET} $1"; }
pass()    { echo -e "  ${GREEN}✓${RESET} $1"; }
warn()    { echo -e "  ${YELLOW}⚠${RESET} $1"; }
fail()    { echo -e "  ${RED}✗${RESET} $1"; exit 1; }
section() { echo -e "\n${BOLD}${CYAN}▶ $1${RESET}"; }

# ── Load .env ─────────────────────────────────────────────────────────────────
if [[ -f "$ROOT/.env" ]]; then
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
    [[ "$line" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]] || continue
    line="${line%%  #*}"; line="${line%% #*}"; line="${line%%	*}"
    line="${line%"${line##*[![:space:]]}"}"
    export "$line" 2>/dev/null || true
  done < "$ROOT/.env"
fi

# ── Defaults / args ───────────────────────────────────────────────────────────
RESTART_AFTER=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --restart)  RESTART_AFTER=true; shift ;;
    --dry-run)  DRY_RUN=true; shift ;;
    --help)
      sed -n '/^# Usage/,/^# ====/p' "$0" | head -n -1
      exit 0 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# ── Read required SSH config ──────────────────────────────────────────────────
SSH_HOST="${RUNPOD_SSH_HOST:-}"
SSH_PORT="${RUNPOD_SSH_PORT:-22}"
SSH_KEY="${RUNPOD_SSH_KEY:-$HOME/.runpod/ssh/RunPod-Key-Go}"
SSH_USER="${RUNPOD_SSH_USER:-root}"
WORKER_DIR="${RUNPOD_WORKER_DIR:-/workspace/pipeline}"
DASHBOARD_URL="${DASHBOARD_URL:-}"

[[ -z "$SSH_HOST" ]] && fail "RUNPOD_SSH_HOST not set (add to .env or export)"
[[ -z "$DASHBOARD_URL" ]] && warn "DASHBOARD_URL not set — worker restart will use whatever is already set on the pod"

SSH_OPTS="-i $SSH_KEY -p $SSH_PORT -o StrictHostKeyChecking=no -o ConnectTimeout=10"
SCP_OPTS="-i $SSH_KEY -P $SSH_PORT -o StrictHostKeyChecking=no -o ConnectTimeout=10"

# ── Validate local pipeline dir ───────────────────────────────────────────────
LOCAL_PIPELINE="$ROOT/backend/pipeline"
LOCAL_EVENT_DETECTOR="$ROOT/backend/event_detector"
[[ -d "$LOCAL_PIPELINE" ]] || fail "Local pipeline dir not found: $LOCAL_PIPELINE"

# event_detector syncs to <WORKER_DIR>/../event_detector on the pod
EVENT_DETECTOR_REMOTE="$(dirname "$WORKER_DIR")/event_detector"

section "Sync local pipeline → ${SSH_HOST}:${WORKER_DIR}"
info "SSH key : $SSH_KEY"
info "Port    : $SSH_PORT"
info "Target  : root@${SSH_HOST}:${WORKER_DIR}"
echo ""

# ── Test SSH connectivity ─────────────────────────────────────────────────────
section "Testing SSH connection"
# shellcheck disable=SC2086
if ssh $SSH_OPTS "${SSH_USER}@${SSH_HOST}" 'echo ok' &>/dev/null; then
  pass "SSH connection successful"
else
  fail "Cannot connect to pod — check RUNPOD_SSH_HOST / RUNPOD_SSH_PORT / RUNPOD_SSH_KEY"
fi

# ── Build rsync excludes list ─────────────────────────────────────────────────
EXCLUDES=(
  --exclude='venv/'
  --exclude='__pycache__/'
  --exclude='*.pyc'
  --exclude='*.pyo'
  --exclude='models/'
  --exclude='input_videos/'
  --exclude='output_videos/'
  --exclude='stubs/'
  --exclude='cache/'
  --exclude='*.log'
  --exclude='.git/'
)

# ── Dry run preview ───────────────────────────────────────────────────────────
if $DRY_RUN; then
  section "Dry run — files that would be synced"
  rsync -av --dry-run "${EXCLUDES[@]}" \
    -e "ssh $SSH_OPTS" \
    "$LOCAL_PIPELINE/" \
    "root@${SSH_HOST}:${WORKER_DIR}/"
  echo ""
  warn "Dry run complete — no files were copied"
  exit 0
fi

# ── Sync files ────────────────────────────────────────────────────────────────
section "Syncing pipeline files"
# shellcheck disable=SC2086
rsync -av --progress "${EXCLUDES[@]}" \
  -e "ssh $SSH_OPTS" \
  "$LOCAL_PIPELINE/" \
  "root@${SSH_HOST}:${WORKER_DIR}/"
pass "Pipeline sync complete"

if [[ -d "$LOCAL_EVENT_DETECTOR" ]]; then
  section "Syncing event_detector module"
  # shellcheck disable=SC2086
  ssh $SSH_OPTS "${SSH_USER}@${SSH_HOST}" "mkdir -p $EVENT_DETECTOR_REMOTE"
  rsync -av --progress \
    --exclude='__pycache__/' --exclude='*.pyc' \
    -e "ssh $SSH_OPTS" \
    "$LOCAL_EVENT_DETECTOR/" \
    "root@${SSH_HOST}:${EVENT_DETECTOR_REMOTE}/"
  pass "event_detector sync complete"
else
  warn "backend/event_detector not found locally — skipping"
fi

# ── Restart worker ────────────────────────────────────────────────────────────
if $RESTART_AFTER; then
  section "Restarting worker on pod"

  RESTART_CMD="DASHBOARD_URL='${DASHBOARD_URL}' bash ${WORKER_DIR}/restart_worker.sh"

  # shellcheck disable=SC2086
  ssh $SSH_OPTS "${SSH_USER}@${SSH_HOST}" "$RESTART_CMD"
  pass "Worker restarted"
  info "Tail logs: ssh -i $SSH_KEY -p $SSH_PORT root@${SSH_HOST} 'tail -f ${WORKER_DIR}/worker.log'"
else
  echo ""
  info "To restart the worker, run:"
  echo ""
  echo "    ssh -i $SSH_KEY -p $SSH_PORT root@${SSH_HOST} \\"
  echo "      \"DASHBOARD_URL='${DASHBOARD_URL:-<url>}' bash ${WORKER_DIR}/restart_worker.sh\""
  echo ""
  info "Or rerun this script with --restart"
fi

echo ""
echo -e "${GREEN}${BOLD}Done!${RESET}"
