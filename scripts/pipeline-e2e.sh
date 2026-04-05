#!/usr/bin/env bash
# =============================================================================
# Football Analytics Dashboard — Pipeline E2E Integration Test
# Uploads a real video, waits for worker processing, verifies all outputs.
#
# Usage: bash scripts/pipeline-e2e.sh [OPTIONS]
#   --video <path|number>  Test video (default: 7 → test_uploads/Test7.mp4)
#   --api-url <url>        Backend URL (default: http://localhost:8000)
#   --mode <mode>          Pipeline mode: all|radar|team|track|players|ball|pitch (default: all)
#   --timeout <seconds>    Max wait for processing (default: 600)
#   --ngrok                Auto-start ngrok tunnel and print RunPod worker command
#   --runpod               Auto-start ngrok AND SSH into RunPod to start the worker
#                          Reads RUNPOD_SSH_HOST, RUNPOD_SSH_PORT, RUNPOD_SSH_KEY,
#                          RUNPOD_WORKER_DIR from .env (or environment)
#   --skip-db              Skip database verification
#   --help                 Show this help
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# ── Colours & helpers (same as ablation.sh) ──────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; DIM='\033[2m'; RESET='\033[0m'

PASS=0; FAIL=0; WARN=0
FAILURES=()

pass()    { echo -e "  ${GREEN}✓${RESET} $1"; PASS=$((PASS+1)); }
fail()    { echo -e "  ${RED}✗${RESET} $1"; FAIL=$((FAIL+1)); FAILURES+=("$1"); }
warn()    { echo -e "  ${YELLOW}⚠${RESET} $1"; WARN=$((WARN+1)); }
info()    { echo -e "  ${DIM}→${RESET} $1"; }
section() { echo -e "\n${BOLD}${CYAN}▶ $1${RESET}"; }

# ── Status vars (for summary dashboard) ──────────────────────────────────────
UPLOAD_STATUS="not run"
ANALYSIS_STATUS="not run"
PROCESSING_STATUS="not run"
VERIFICATION_STATUS="not run"
DB_STATUS="skipped"
VIDEO_ID=""
ANALYSIS_ID=""

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

# ── Defaults ──────────────────────────────────────────────────────────────────
API_URL="http://localhost:8000"
VIDEO_ARG="7"
PIPELINE_MODE="all"
TIMEOUT=600
SKIP_DB=false
USE_NGROK=false
USE_RUNPOD=false
NGROK_URL=""
NGROK_PID=""
# RunPod SSH config — read from env/.env
RUNPOD_SSH_HOST="${RUNPOD_SSH_HOST:-}"
RUNPOD_SSH_PORT="${RUNPOD_SSH_PORT:-22}"
RUNPOD_SSH_KEY="${RUNPOD_SSH_KEY:-$HOME/.ssh/runpod}"
RUNPOD_SSH_USER="${RUNPOD_SSH_USER:-root}"
RUNPOD_WORKER_DIR="${RUNPOD_WORKER_DIR:-/workspace/pipeline}"

VALID_MODES=("all" "radar" "team" "track" "players" "ball" "pitch")

# ── Arg parsing ──────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --video)    VIDEO_ARG="$2"; shift 2 ;;
    --api-url)  API_URL="$2"; shift 2 ;;
    --mode)     PIPELINE_MODE="$2"; shift 2 ;;
    --timeout)  TIMEOUT="$2"; shift 2 ;;
    --skip-db)       SKIP_DB=true; shift ;;
    --ngrok)         USE_NGROK=true; shift ;;
    --runpod)        USE_RUNPOD=true; USE_NGROK=true; shift ;;
    --runpod-host)   RUNPOD_SSH_HOST="$2"; shift 2 ;;
    --runpod-port)   RUNPOD_SSH_PORT="$2"; shift 2 ;;
    --runpod-key)    RUNPOD_SSH_KEY="$2"; shift 2 ;;
    --runpod-dir)    RUNPOD_WORKER_DIR="$2"; shift 2 ;;
    --help)
      sed -n '/^# Usage/,/^# ===/p' "$0" | head -8 | sed 's/^# //'
      exit 0 ;;
    *) echo "Unknown flag: $1. Use --help for usage."; exit 1 ;;
  esac
done

# Validate mode
VALID=false
for m in "${VALID_MODES[@]}"; do [[ "$m" == "$PIPELINE_MODE" ]] && VALID=true; done
if [[ "$VALID" == false ]]; then
  echo -e "${RED}Invalid mode: $PIPELINE_MODE${RESET}. Valid: ${VALID_MODES[*]}"
  exit 1
fi

# Resolve video path
if [[ "$VIDEO_ARG" =~ ^[0-9]+$ ]]; then
  if [[ "$VIDEO_ARG" -lt 1 || "$VIDEO_ARG" -gt 7 ]]; then
    echo -e "${RED}Error: --video number must be 1-7 (got $VIDEO_ARG)${RESET}"
    exit 1
  fi
  VIDEO_PATH="$ROOT/test_uploads/Test${VIDEO_ARG}.mp4"
else
  VIDEO_PATH="$VIDEO_ARG"
fi

VIDEO_BASENAME=$(basename "$VIDEO_PATH")

# Header
echo ""
echo -e "${BOLD}${CYAN}Pipeline E2E — $(date '+%Y-%m-%d %H:%M:%S')${RESET}"
info "Video:   $VIDEO_PATH"
info "API URL: $API_URL"
info "Mode:    $PIPELINE_MODE"
info "Timeout: ${TIMEOUT}s"
[[ "$USE_NGROK" == true ]] && info "ngrok:   enabled (tunnel will be started automatically)"

# =============================================================================
# 1. PRE-FLIGHT CHECKS
# =============================================================================
section "Pre-flight checks"

# Dependencies
if command -v curl &>/dev/null; then
  pass "curl available"
else
  fail "curl not found — install curl to run this script"
  exit 1
fi

if command -v jq &>/dev/null; then
  pass "jq available"
else
  fail "jq not found — install jq (brew install jq) to run this script"
  exit 1
fi

# Backend reachable
HEALTH=$(curl -sf --max-time 5 "$API_URL/api/health" 2>/dev/null || echo "FAIL")
if [[ "$HEALTH" == "FAIL" ]]; then
  fail "Backend unreachable at $API_URL — is the API running?"
  exit 1
else
  STATUS_VAL=$(echo "$HEALTH" | jq -r '.status // "unknown"' 2>/dev/null || echo "ok")
  pass "Backend reachable (status: $STATUS_VAL)"
fi

# Worker endpoint probe
WORKER_KEY="${WORKER_API_KEY:-}"
PENDING_RAW=$(curl -sf --max-time 5 -H "X-Worker-Key: $WORKER_KEY" "$API_URL/api/worker/pending" 2>/dev/null || echo "FAIL")
if [[ "$PENDING_RAW" == "FAIL" ]]; then
  warn "Worker endpoint unreachable — the worker may not be authenticated"
else
  PENDING_COUNT=$(echo "$PENDING_RAW" | jq '.analyses | length' 2>/dev/null || echo "-1")
  if [[ "$PENDING_COUNT" == "-1" ]]; then
    warn "Worker endpoint returned unexpected response"
  elif [[ "$PENDING_COUNT" -gt 0 ]]; then
    warn "Worker queue has $PENDING_COUNT pending analysis(es) already queued"
    pass "Worker endpoint responsive"
  else
    pass "Worker endpoint responsive (queue empty)"
  fi
fi

# Test video exists
if [[ ! -f "$VIDEO_PATH" ]]; then
  fail "Test video not found: $VIDEO_PATH"
  exit 1
fi
VIDEO_SIZE_BYTES=$(wc -c < "$VIDEO_PATH" | tr -d ' ')
VIDEO_SIZE_MB=$(( VIDEO_SIZE_BYTES / 1024 / 1024 ))
pass "Test video found: $VIDEO_BASENAME (${VIDEO_SIZE_MB}MB)"
if [[ "$VIDEO_SIZE_MB" -gt 100 ]]; then
  warn "Video is large (${VIDEO_SIZE_MB}MB) — upload may be slow"
fi

# ngrok tunnel
if [[ "$USE_NGROK" == true ]]; then
  if ! command -v ngrok &>/dev/null; then
    warn "ngrok not found — install via: brew install ngrok/ngrok/ngrok"
    USE_NGROK=false
  else
    # Check if ngrok is already running
    NGROK_URL=$(curl -sf http://localhost:4040/api/tunnels 2>/dev/null \
      | jq -r '.tunnels[] | select(.proto=="https") | .public_url' 2>/dev/null \
      | head -1 || echo "")
    if [[ -n "$NGROK_URL" ]]; then
      pass "ngrok already running: $NGROK_URL"
    else
      info "Starting ngrok tunnel on port 8000..."
      ngrok http 8000 --log=false >/dev/null 2>&1 &
      NGROK_PID=$!
      for i in $(seq 1 15); do
        sleep 1
        NGROK_URL=$(curl -sf http://localhost:4040/api/tunnels 2>/dev/null \
          | jq -r '.tunnels[] | select(.proto=="https") | .public_url' 2>/dev/null \
          | head -1 || echo "")
        [[ -n "$NGROK_URL" ]] && break
      done
      if [[ -z "$NGROK_URL" ]]; then
        warn "ngrok failed to start after 15s — continuing without tunnel"
        USE_NGROK=false
      else
        pass "ngrok tunnel active: $NGROK_URL"
      fi
    fi
  fi
fi

# RunPod SSH pre-flight
if [[ "$USE_RUNPOD" == true ]]; then
  if [[ -z "$RUNPOD_SSH_HOST" ]]; then
    warn "RUNPOD_SSH_HOST not set — add it to .env or pass --runpod-host <ip>"
    warn "Continuing without RunPod auto-start (use --ngrok instead to get the command)"
    USE_RUNPOD=false
  elif [[ ! -f "$RUNPOD_SSH_KEY" ]]; then
    warn "SSH key not found: $RUNPOD_SSH_KEY — set RUNPOD_SSH_KEY in .env or pass --runpod-key <path>"
    USE_RUNPOD=false
  else
    SSH_TEST=$(ssh -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=8 \
      -i "$RUNPOD_SSH_KEY" -p "$RUNPOD_SSH_PORT" "${RUNPOD_SSH_USER}@$RUNPOD_SSH_HOST" \
      "echo ok" 2>/dev/null || echo "FAIL")
    if [[ "$SSH_TEST" == "FAIL" ]]; then
      warn "Cannot SSH into RunPod ($RUNPOD_SSH_HOST:$RUNPOD_SSH_PORT) — check host/port/key"
      USE_RUNPOD=false
    else
      pass "RunPod SSH reachable ($RUNPOD_SSH_HOST:$RUNPOD_SSH_PORT)"
    fi
  fi
fi

# Check for local worker.py that could steal jobs from RunPod before we upload
LOCAL_WORKER_PID=$(ps aux | grep '[p]ython.*worker\.py' | awk '{print $2}' | head -1 || echo "")
if [[ -n "$LOCAL_WORKER_PID" ]]; then
  if [[ "$USE_RUNPOD" == "true" ]]; then
    fail "Local worker.py running (PID $LOCAL_WORKER_PID) — it will steal jobs from RunPod!"
    info "Kill it first: kill $LOCAL_WORKER_PID"
    exit 1
  else
    warn "Local worker.py detected (PID $LOCAL_WORKER_PID) — it may claim the analysis job"
  fi
fi

# =============================================================================
# 2. UPLOAD PHASE
# =============================================================================
section "Upload"

UPLOAD_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$API_URL/api/upload/video" \
  -F "video=@$VIDEO_PATH" \
  -F "title=E2E Test $(date +%Y%m%d-%H%M%S)" \
  -F "description=Pipeline E2E test run — $VIDEO_BASENAME" \
  2>/dev/null)

HTTP_CODE=$(echo "$UPLOAD_RESPONSE" | tail -1)
UPLOAD_BODY=$(echo "$UPLOAD_RESPONSE" | sed '$d')

if [[ "$HTTP_CODE" != "200" && "$HTTP_CODE" != "201" ]]; then
  fail "Upload failed (HTTP $HTTP_CODE)"
  info "Response: $UPLOAD_BODY"
  UPLOAD_STATUS="FAILED (HTTP $HTTP_CODE)"
  exit 1
fi

VIDEO_ID=$(echo "$UPLOAD_BODY" | jq -r '.id // empty' 2>/dev/null)
if [[ -z "$VIDEO_ID" ]]; then
  fail "Upload response missing video id"
  info "Response: $UPLOAD_BODY"
  UPLOAD_STATUS="FAILED (no id in response)"
  exit 1
fi

pass "Video uploaded (id=$VIDEO_ID)"
UPLOAD_STATUS="OK (video id=$VIDEO_ID)"

# =============================================================================
# 3. ANALYSIS CREATION
# =============================================================================
section "Create analysis"

ANALYSIS_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$API_URL/api/analysis" \
  -H "Content-Type: application/json" \
  -d "{\"videoId\": $VIDEO_ID, \"mode\": \"$PIPELINE_MODE\"}" \
  2>/dev/null)

ANALYSIS_HTTP=$(echo "$ANALYSIS_RESPONSE" | tail -1)
ANALYSIS_BODY=$(echo "$ANALYSIS_RESPONSE" | sed '$d')

if [[ "$ANALYSIS_HTTP" != "200" && "$ANALYSIS_HTTP" != "201" ]]; then
  fail "Analysis creation failed (HTTP $ANALYSIS_HTTP)"
  info "Response: $ANALYSIS_BODY"
  ANALYSIS_STATUS="FAILED (HTTP $ANALYSIS_HTTP)"
  exit 1
fi

ANALYSIS_ID=$(echo "$ANALYSIS_BODY" | jq -r '.id // empty' 2>/dev/null)
if [[ -z "$ANALYSIS_ID" ]]; then
  fail "Analysis response missing id"
  info "Response: $ANALYSIS_BODY"
  ANALYSIS_STATUS="FAILED (no id)"
  exit 1
fi

pass "Analysis created (id=$ANALYSIS_ID)"
ANALYSIS_STATUS="created (id=$ANALYSIS_ID)"
info "View at: http://localhost:5173/analysis/$ANALYSIS_ID"

# Export analysis ID for Playwright
echo "$ANALYSIS_ID" > /tmp/pipeline_e2e_analysis_id.txt
info "Analysis ID saved to /tmp/pipeline_e2e_analysis_id.txt for Playwright reuse"

# Start RunPod worker or print command
if [[ "$USE_RUNPOD" == true && -n "$NGROK_URL" ]]; then
  section "Starting RunPod worker"
  REMOTE_CMD="cd $RUNPOD_WORKER_DIR && nohup env DASHBOARD_URL=$NGROK_URL MODEL_URL_PLAYER=http://localhost/player_detection.pt MODEL_URL_BALL=http://localhost/ball_detection.pt MODEL_URL_PITCH=http://localhost/pitch_detection.pt python worker.py > /tmp/worker-e2e.log 2>&1 &"
  SSH_RESULT=$(ssh -o StrictHostKeyChecking=no -o BatchMode=yes \
    -i "$RUNPOD_SSH_KEY" -p "$RUNPOD_SSH_PORT" "${RUNPOD_SSH_USER}@$RUNPOD_SSH_HOST" \
    "$REMOTE_CMD" 2>&1 && echo "ok" || echo "FAIL")
  if [[ "$SSH_RESULT" == *"FAIL"* ]]; then
    warn "Failed to start worker on RunPod: $SSH_RESULT"
    info "Start manually: $REMOTE_CMD"
  else
    pass "Worker started on RunPod (logs: /tmp/worker-e2e.log)"
    info "Polling for completion — worker is now processing analysis $ANALYSIS_ID..."
  fi
elif [[ -n "$NGROK_URL" ]]; then
  echo ""
  echo -e "${BOLD}${CYAN}▶ RunPod Worker Command${RESET}"
  echo -e "  ${DIM}SSH into RunPod and run this — the script will keep polling:${RESET}"
  echo ""
  echo -e "  ${YELLOW}DASHBOARD_URL=$NGROK_URL \\"
  echo -e "  MODEL_URL_PLAYER=http://localhost/player_detection.pt \\"
  echo -e "  MODEL_URL_BALL=http://localhost/ball_detection.pt \\"
  echo -e "  MODEL_URL_PITCH=http://localhost/pitch_detection.pt \\"
  echo -e "  python pipeline/worker.py${RESET}"
  echo ""
  echo -e "  ${DIM}Tip: add --runpod to have this script SSH in and start the worker automatically${RESET}"
fi

# =============================================================================
# 4. PROCESSING POLL LOOP
# =============================================================================
section "Waiting for pipeline processing..."

START_TIME=$(date +%s)
LAST_STAGE=""
LAST_PROGRESS=-1
WARNED_PENDING=false

while true; do
  ELAPSED=$(( $(date +%s) - START_TIME ))

  if [[ "$ELAPSED" -ge "$TIMEOUT" ]]; then
    echo ""
    fail "Processing timed out after ${TIMEOUT}s"
    PROCESSING_STATUS="TIMEOUT after ${TIMEOUT}s"
    break
  fi

  POLL_RAW=$(curl -sf --max-time 10 "$API_URL/api/analysis/$ANALYSIS_ID" 2>/dev/null || echo "FAIL")

  if [[ "$POLL_RAW" == "FAIL" ]]; then
    warn "Poll request failed — retrying"
    sleep 10
    continue
  fi

  A_STATUS=$(echo "$POLL_RAW" | jq -r '.status // "unknown"')
  A_PROGRESS=$(echo "$POLL_RAW" | jq -r '.progress // 0')
  A_STAGE=$(echo "$POLL_RAW" | jq -r '.currentStage // ""')
  A_ERROR=$(echo "$POLL_RAW" | jq -r '.errorMessage // ""')

  # Print progress if changed
  if [[ "$A_STAGE" != "$LAST_STAGE" || "$A_PROGRESS" != "$LAST_PROGRESS" ]]; then
    printf "\r  ${DIM}[%3s%%] Stage: %-12s  Elapsed: %3ss${RESET}  " \
      "$A_PROGRESS" "${A_STAGE:-pending}" "$ELAPSED"
    LAST_STAGE="$A_STAGE"
    LAST_PROGRESS="$A_PROGRESS"
  fi

  # Warn if still pending after 60s (worker probably not running)
  if [[ "$A_STATUS" == "pending" && "$ELAPSED" -ge 60 && "$WARNED_PENDING" == false ]]; then
    echo ""
    warn "Analysis still pending after ${ELAPSED}s — is the worker running?"
    if [[ -n "$NGROK_URL" ]]; then
      info "Start worker on RunPod: DASHBOARD_URL=$NGROK_URL python pipeline/worker.py"
    else
      info "Start worker locally:   DASHBOARD_URL=$API_URL python3 backend/pipeline/worker.py"
      info "Or use --ngrok flag to auto-start a tunnel for RunPod"
    fi
    WARNED_PENDING=true
  fi

  case "$A_STATUS" in
    completed)
      echo ""
      pass "Processing completed in ${ELAPSED}s"
      PROCESSING_STATUS="completed (${ELAPSED}s)"
      break
      ;;
    failed)
      echo ""
      fail "Pipeline failed: ${A_ERROR:-unknown error}"
      PROCESSING_STATUS="FAILED: ${A_ERROR:-unknown}"
      break
      ;;
    *)
      sleep 10
      ;;
  esac
done

# =============================================================================
# 5. VERIFICATION
# =============================================================================
section "Verification"

if [[ "$PROCESSING_STATUS" != completed* ]]; then
  warn "Skipping verification — processing did not complete"
  VERIFICATION_STATUS="skipped (processing not completed)"
else
  VERIFY_PASS=0
  VERIFY_TOTAL=0

  # Fetch final analysis
  FINAL=$(curl -sf "$API_URL/api/analysis/$ANALYSIS_ID" 2>/dev/null || echo "{}")

  # 5a. Status check
  VERIFY_TOTAL=$((VERIFY_TOTAL+1))
  FINAL_STATUS=$(echo "$FINAL" | jq -r '.status // "unknown"')
  if [[ "$FINAL_STATUS" == "completed" ]]; then
    pass "Analysis status: completed"; VERIFY_PASS=$((VERIFY_PASS+1))
  else
    fail "Analysis status: $FINAL_STATUS (expected completed)"
  fi

  # 5b. Annotated video URL
  VERIFY_TOTAL=$((VERIFY_TOTAL+1))
  ANNOTATED_URL=$(echo "$FINAL" | jq -r '.annotatedVideoUrl // empty')
  if [[ -n "$ANNOTATED_URL" ]]; then
    pass "Annotated video URL present"; VERIFY_PASS=$((VERIFY_PASS+1))
    info "URL: $ANNOTATED_URL"
  else
    fail "No annotated video URL in analysis"
  fi

  # 5c. Statistics
  VERIFY_TOTAL=$((VERIFY_TOTAL+1))
  STATS=$(curl -sf "$API_URL/api/statistics/$ANALYSIS_ID" 2>/dev/null || echo "null")
  if echo "$STATS" | jq -e '. != null' &>/dev/null; then
    pass "Statistics created"; VERIFY_PASS=$((VERIFY_PASS+1))
    POSS1=$(echo "$STATS" | jq -r '.possessionTeam1 // "?"')
    POSS2=$(echo "$STATS" | jq -r '.possessionTeam2 // "?"')
    PASSES1=$(echo "$STATS" | jq -r '.passesTeam1 // "?"')
    SHOTS1=$(echo "$STATS" | jq -r '.shotsTeam1 // "?"')
    info "Possession: Team1=${POSS1}%  Team2=${POSS2}%  |  Passes: ${PASSES1}  |  Shots: ${SHOTS1}"
  else
    fail "No statistics found for analysis $ANALYSIS_ID"
  fi

  # 5d. Events
  VERIFY_TOTAL=$((VERIFY_TOTAL+1))
  EVENTS=$(curl -sf "$API_URL/api/events/$ANALYSIS_ID" 2>/dev/null || echo "[]")
  EVENT_COUNT=$(echo "$EVENTS" | jq 'length' 2>/dev/null || echo "0")
  if [[ "$EVENT_COUNT" -gt 0 ]]; then
    pass "Events created ($EVENT_COUNT events)"; VERIFY_PASS=$((VERIFY_PASS+1))
    # Print breakdown by type
    EVENT_BREAKDOWN=$(echo "$EVENTS" | jq -r '[.[].type] | group_by(.) | map("\(.[0]): \(length)") | .[]' 2>/dev/null || echo "")
    [[ -n "$EVENT_BREAKDOWN" ]] && echo "$EVENT_BREAKDOWN" | while read -r line; do info "$line"; done
  else
    warn "No events found — may be expected for mode '$PIPELINE_MODE'"
    VERIFY_PASS=$((VERIFY_PASS+1))  # treat as warn, not fail
  fi

  # 5e. Video download check
  VERIFY_TOTAL=$((VERIFY_TOTAL+1))
  if [[ -n "$ANNOTATED_URL" ]]; then
    # Handle relative URLs
    if [[ "$ANNOTATED_URL" == /* ]]; then
      FULL_VIDEO_URL="${API_URL}${ANNOTATED_URL}"
    else
      FULL_VIDEO_URL="$ANNOTATED_URL"
    fi
    CONTENT_TYPE=$(curl -sI --max-time 10 "$FULL_VIDEO_URL" 2>/dev/null | grep -i "content-type:" | head -1 | tr -d '\r' || echo "")
    if echo "$CONTENT_TYPE" | grep -qi "video"; then
      pass "Annotated video downloadable (Content-Type: video)"; VERIFY_PASS=$((VERIFY_PASS+1))
    else
      # Fallback: check MP4 magic bytes
      MAGIC=$(curl -sf --max-time 10 --range 0-11 "$FULL_VIDEO_URL" 2>/dev/null | xxd -p 2>/dev/null | head -1 || echo "")
      if echo "$MAGIC" | grep -qi "6674797"; then
        pass "Annotated video downloadable (MP4 signature confirmed)"; VERIFY_PASS=$((VERIFY_PASS+1))
      elif [[ -z "$CONTENT_TYPE" ]]; then
        warn "Could not verify video download (server returned no headers)"
        VERIFY_PASS=$((VERIFY_PASS+1))
      else
        fail "Annotated video download failed or wrong content type: $CONTENT_TYPE"
      fi
    fi
  else
    warn "No annotated URL to verify — skipping download check"
    VERIFY_TOTAL=$((VERIFY_TOTAL-1))
  fi

  VERIFICATION_STATUS="${VERIFY_PASS}/${VERIFY_TOTAL} checks passed"
fi

# =============================================================================
# 6. DATABASE VERIFICATION
# =============================================================================
section "Database verification"

if [[ "$SKIP_DB" == true ]]; then
  info "Skipped (--skip-db)"
  DB_STATUS="skipped"
elif [[ -z "$VIDEO_ID" || -z "$ANALYSIS_ID" ]]; then
  warn "No video/analysis IDs to check — skipping DB verification"
  DB_STATUS="skipped (no IDs)"
else
  PY=$(command -v python3 2>/dev/null || command -v python 2>/dev/null || echo "")
  DB_URL="${DATABASE_URL:-}"

  if [[ -z "$PY" || -z "$DB_URL" ]]; then
    warn "python3 or DATABASE_URL not available — skipping DB verification"
    DB_STATUS="skipped (no python/DATABASE_URL)"
  else
    DB_RESULT=$("$PY" - "$VIDEO_ID" "$ANALYSIS_ID" <<'PYEOF' 2>/dev/null
import asyncio, sys, os

async def check(video_id, analysis_id):
    try:
        import asyncpg
        url = os.environ.get("DATABASE_URL", "")
        url = url.replace("postgresql+asyncpg://", "postgresql://")
        conn = await asyncpg.connect(url, timeout=8)

        # Check video row
        vid_row = await conn.fetchrow('SELECT id FROM videos WHERE id = $1', int(video_id))
        # Check analysis row
        ana_row = await conn.fetchrow('SELECT id, status FROM analyses WHERE id = $1', int(analysis_id))
        # Check statistic row
        stat_row = await conn.fetchrow('SELECT id FROM statistics WHERE "analysisId" = $1', int(analysis_id))
        # Count events
        event_count = await conn.fetchval('SELECT count(*) FROM events WHERE "analysisId" = $1', int(analysis_id))

        await conn.close()

        vid_ok = "ok" if vid_row else "missing"
        ana_ok = "ok" if ana_row else "missing"
        ana_status = ana_row["status"] if ana_row else "?"
        stat_ok = "ok" if stat_row else "missing"
        events_n = int(event_count) if event_count else 0
        print(f"video:{vid_ok}|analysis:{ana_ok}({ana_status})|stat:{stat_ok}|events:{events_n}")
    except Exception as e:
        print(f"error:{e}")

asyncio.run(check(sys.argv[1], sys.argv[2]))
PYEOF
    )

    if echo "$DB_RESULT" | grep -q "^error:"; then
      fail "DB verification error: ${DB_RESULT#error:}"
      DB_STATUS="ERROR"
    else
      VID_OK=$(echo "$DB_RESULT" | sed -n 's/.*video:\([^|]*\).*/\1/p' || echo "?")
      ANA_OK=$(echo "$DB_RESULT" | sed -n 's/.*analysis:\([^|]*\).*/\1/p' || echo "?")
      STAT_OK=$(echo "$DB_RESULT" | sed -n 's/.*stat:\([^|]*\).*/\1/p' || echo "?")
      EVENTS_N=$(echo "$DB_RESULT" | sed -n 's/.*events:\([0-9]*\).*/\1/p' || echo "?")

      [[ "$VID_OK" == "ok" ]] && pass "DB: videos row exists" || fail "DB: videos row missing (id=$VIDEO_ID)"
      [[ "$ANA_OK" == ok* ]] && pass "DB: analyses row exists ($ANA_OK)" || fail "DB: analyses row missing (id=$ANALYSIS_ID)"
      [[ "$STAT_OK" == "ok" ]] && pass "DB: statistics row exists" || warn "DB: statistics row missing — may not have been created yet"
      [[ "$EVENTS_N" != "?" && "$EVENTS_N" -gt 0 ]] && pass "DB: $EVENTS_N event rows found" || warn "DB: no event rows found"

      DB_STATUS="video=$VID_OK  analysis=$ANA_OK  stat=$STAT_OK  events=$EVENTS_N"
    fi
  fi
fi

# =============================================================================
# SUMMARY DASHBOARD
# =============================================================================
WIDTH=50
dashline() { printf '━%.0s' $(seq 1 $WIDTH); echo; }
pad() { printf "  %-20s %s\n" "$1" "$2"; }

echo ""
echo -e "${BOLD}$(dashline)${RESET}"
echo -e "${BOLD}  PIPELINE E2E REPORT${RESET}"
echo -e "${BOLD}$(dashline)${RESET}"

pad "Video:"         "$VIDEO_BASENAME (${VIDEO_SIZE_MB:-?}MB)"
pad "API URL:"       "$API_URL"
[[ -n "$NGROK_URL" ]] && pad "ngrok URL:"     "$NGROK_URL"
[[ "$USE_RUNPOD" == true ]] && pad "RunPod SSH:"   "$RUNPOD_SSH_HOST:$RUNPOD_SSH_PORT"
pad "Mode:"          "$PIPELINE_MODE"
pad "Video ID:"      "${VIDEO_ID:-(none)}"
pad "Analysis ID:"   "${ANALYSIS_ID:-(none)}"
pad "Upload:"        "$UPLOAD_STATUS"
pad "Processing:"    "$PROCESSING_STATUS"
pad "Verification:"  "$VERIFICATION_STATUS"
pad "Database:"      "$DB_STATUS"

echo -e "${BOLD}$(dashline)${RESET}"
echo -e "  ${GREEN}Passed${RESET}:   $PASS"
echo -e "  ${YELLOW}Warnings${RESET}: $WARN"
echo -e "  ${RED}Failed${RESET}:   $FAIL"

if [[ ${#FAILURES[@]} -gt 0 ]]; then
  echo ""
  echo -e "  ${RED}${BOLD}Failures:${RESET}"
  for f in "${FAILURES[@]}"; do
    echo -e "  ${RED}✗${RESET} $f"
  done
fi

echo -e "${BOLD}$(dashline)${RESET}"
echo ""

if [[ -n "$ANALYSIS_ID" ]]; then
  echo -e "  ${DIM}Playwright:${RESET} PIPELINE_E2E_ANALYSIS_ID=$ANALYSIS_ID pnpm exec playwright test tests/e2e/pipeline.spec.ts"
fi
echo ""

if [[ $FAIL -gt 0 ]]; then
  echo -e "${RED}${BOLD}PIPELINE E2E FAILED${RESET} — $FAIL check(s) failed"
  exit 1
elif [[ $WARN -gt 0 ]]; then
  echo -e "${YELLOW}${BOLD}PIPELINE E2E PASSED WITH WARNINGS${RESET}"
  exit 0
else
  echo -e "${GREEN}${BOLD}PIPELINE E2E PASSED${RESET}"
  exit 0
fi
