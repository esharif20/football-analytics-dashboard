#!/usr/bin/env bash
# =============================================================================
# Football Analytics Dashboard — Full System Ablation
# Checks every integration point so you know if the system is ready to go.
#
# Usage: bash scripts/ablation.sh [--quick] [--skip-build]
#   --quick       Env + structure + API keys only (no network, no build)
#   --skip-build  Everything except frontend build/lint/tsc/tests
# =============================================================================

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# ── Flags ────────────────────────────────────────────────────────────────────
QUICK=false
SKIP_BUILD=false
for arg in "$@"; do
  case $arg in
    --quick)      QUICK=true ;;
    --skip-build) SKIP_BUILD=true ;;
  esac
done

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; DIM='\033[2m'; RESET='\033[0m'

PASS=0; FAIL=0; WARN=0
FAILURES=()

# Status-line state for the summary dashboard
AUTH_MODE="unknown"
DB_STATUS="unknown"
LLM_STATUS="unknown"
ROBOFLOW_STATUS="unknown"
GPU_STATUS="unknown"
NGROK_STATUS="not running"
NGROK_URL=""
BACKEND_STATUS="unknown"
FRONTEND_STATUS="unknown"
BUILD_STATUS="skipped"
TEST_STATUS="skipped"

pass()    { echo -e "  ${GREEN}✓${RESET} $1"; PASS=$((PASS+1)); }
fail()    { echo -e "  ${RED}✗${RESET} $1"; FAIL=$((FAIL+1)); FAILURES+=("$1"); }
warn()    { echo -e "  ${YELLOW}⚠${RESET} $1"; WARN=$((WARN+1)); }
info()    { echo -e "  ${DIM}→${RESET} $1"; }
section() { echo -e "\n${BOLD}${CYAN}▶ $1${RESET}"; }

# ── Load .env (safe — only KEY=VALUE lines, skip comments/blanks) ─────────────
if [[ -f "$ROOT/.env" ]]; then
  while IFS= read -r line || [[ -n "$line" ]]; do
    # Skip blank lines and comments
    [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
    # Only process lines that look like KEY=VALUE
    [[ "$line" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]] || continue
    # Strip inline comments (space + #) and trailing whitespace
    line="${line%%  #*}"   # double-space comment
    line="${line%% #*}"    # single-space comment
    line="${line%%	*}"  # tab
    line="${line%"${line##*[![:space:]]}"}"  # trailing whitespace
    export "$line" 2>/dev/null || true
  done < "$ROOT/.env"
fi

# Resolve python
PY=$(command -v python3 2>/dev/null || command -v python 2>/dev/null || echo "")

# HTTP probe helper (accepts 2xx, 3xx, 401, 403 as "reachable")
probe() {
  local label="$1" url="$2"
  local code
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "$url" 2>/dev/null || echo "000")
  if [[ "$code" =~ ^(2|3|401|403) ]]; then
    pass "$label reachable (HTTP $code)"
    echo "$code"
  else
    fail "$label unreachable (HTTP $code) — $url"
    echo "000"
  fi
}

# =============================================================================
# 1. ENVIRONMENT & CONFIG
# =============================================================================
section "Environment & config"

if [[ -f "$ROOT/.env" ]]; then
  pass ".env file exists"
else
  fail ".env file MISSING — copy env.example and fill in values"
fi

DB_URL="${DATABASE_URL:-}"
if [[ -z "$DB_URL" ]]; then
  fail "DATABASE_URL not set"
elif [[ "$DB_URL" == *"postgresql+asyncpg://"* ]]; then
  pass "DATABASE_URL uses postgresql+asyncpg driver"
elif [[ "$DB_URL" == *"mysql"* ]]; then
  fail "DATABASE_URL still uses MySQL — update to postgresql+asyncpg://"
else
  warn "DATABASE_URL set but driver unclear: $DB_URL"
fi

LOCAL_DEV="${LOCAL_DEV_MODE:-true}"
JWT="${JWT_SECRET:-dev-secret}"
if [[ "$LOCAL_DEV" == "true" ]]; then
  pass "Auth mode: LOCAL_DEV_MODE=true (auto-login, no JWT required)"
  AUTH_MODE="Local dev (auto-login)"
else
  if [[ "$JWT" == "dev-secret" ]]; then
    fail "LOCAL_DEV_MODE=false but JWT_SECRET is still 'dev-secret' — set a real secret"
    AUTH_MODE="JWT (INSECURE SECRET)"
  else
    pass "Auth mode: JWT (LOCAL_DEV_MODE=false, custom JWT_SECRET set)"
    AUTH_MODE="JWT (production)"
  fi
fi

STORAGE_DIR="${LOCAL_STORAGE_DIR:-./uploads}"
STORAGE_ABS="$ROOT/${STORAGE_DIR#./}"
if [[ -d "$STORAGE_ABS" ]]; then
  if [[ -w "$STORAGE_ABS" ]]; then
    pass "LOCAL_STORAGE_DIR ($STORAGE_DIR) exists and is writable"
  else
    fail "LOCAL_STORAGE_DIR ($STORAGE_DIR) exists but is NOT writable"
  fi
else
  warn "LOCAL_STORAGE_DIR ($STORAGE_DIR) does not exist yet — will be created on first run"
fi

# =============================================================================
# 2. FILE STRUCTURE
# =============================================================================
section "File structure"

must_exist() {
  [[ -e "$ROOT/$1" ]] && pass "$1" || fail "$1 MISSING"
}
must_not_exist() {
  [[ ! -e "$ROOT/$1" ]] && pass "$1 correctly absent" || fail "$1 should have been removed"
}

must_exist "backend/api/main.py"
must_exist "backend/api/services/llm_providers.py"
must_exist "backend/api/services/tactical.py"
must_exist "backend/api/routers/worker.py"
must_exist "backend/api/ws.py"
must_exist "backend/pipeline/worker.py"
must_exist "docker/Dockerfile.worker"
must_exist "frontend/src/hooks/useAuth.ts"
must_exist "frontend/src/types.ts"
must_exist "frontend/src/App.tsx"
must_exist "env.example"
must_exist "supabase"

must_not_exist "docker-compose.yml"
must_not_exist "docker/mysql-dev.cnf"
must_not_exist "scripts/start-test-stack.sh"
must_not_exist "frontend/src/_core"
must_not_exist "frontend/src/shared"
must_not_exist "frontend/src/pages/Analysis.tsx"

section "Import integrity"

check_no_import() {
  local pattern="$1" label="$2"
  if grep -r "$pattern" frontend/src/ --include="*.ts" --include="*.tsx" -q 2>/dev/null; then
    fail "Stale import: $label"
  else
    pass "No stale '$label' imports"
  fi
}
check_no_import "@/_core/hooks" "@/_core/hooks"
check_no_import "@/shared/types" "@/shared/types"
check_no_import "mysql://" "mysql:// connection string"

# =============================================================================
# 3. API KEYS & PROVIDER CONNECTIVITY
# =============================================================================
section "API keys & provider connectivity"

GEMINI_KEY="${GEMINI_API_KEY:-}"
OPENAI_KEY="${OPENAI_API_KEY:-}"
HF_KEY="${HF_API_KEY:-}"
ROBOFLOW_KEY="${ROBOFLOW_API_KEY:-}"
LLM_PROV="${LLM_PROVIDER:-gemini}"

[[ -n "$GEMINI_KEY" ]] && pass "GEMINI_API_KEY set"       || warn "GEMINI_API_KEY not set (Gemini unavailable)"
[[ -n "$OPENAI_KEY" ]] && pass "OPENAI_API_KEY set"       || warn "OPENAI_API_KEY not set (OpenAI unavailable)"
[[ -n "$HF_KEY" ]]     && pass "HF_API_KEY set"           || warn "HF_API_KEY not set (HuggingFace unavailable)"
[[ -n "$ROBOFLOW_KEY" ]] && pass "ROBOFLOW_API_KEY set"   || warn "ROBOFLOW_API_KEY not set (pitch detection degraded)"

# Check at least one LLM key
if [[ -z "$GEMINI_KEY" && -z "$OPENAI_KEY" && -z "$HF_KEY" ]]; then
  fail "No LLM API key set — set at least one of GEMINI_API_KEY / OPENAI_API_KEY / HF_API_KEY"
  LLM_STATUS="no provider configured"
else
  info "Active LLM provider: $LLM_PROV"
fi

# Live LLM probe (skip in --quick mode)
if [[ "$QUICK" == false && -n "$PY" ]]; then
  info "Probing LLM provider ($LLM_PROV) with a test prompt..."
  LLM_PROBE=$("$PY" - <<PYEOF 2>/dev/null
import os, sys, asyncio
provider = os.getenv("LLM_PROVIDER", "gemini").lower()
try:
    if provider == "gemini":
        key = os.getenv("GEMINI_API_KEY", "")
        if not key:
            print("no-key"); sys.exit(0)
        import google.generativeai as genai
        genai.configure(api_key=key)
        m = genai.GenerativeModel("gemini-2.0-flash")
        r = m.generate_content("Reply with exactly: OK")
        print("ok" if r.text else "empty")
    elif provider == "openai":
        key = os.getenv("OPENAI_API_KEY", "")
        if not key:
            print("no-key"); sys.exit(0)
        from openai import OpenAI
        c = OpenAI(api_key=key)
        r = c.chat.completions.create(model="gpt-4o-mini",
            messages=[{"role":"user","content":"Reply with exactly: OK"}],
            max_tokens=5)
        print("ok" if r.choices[0].message.content else "empty")
    elif provider in ("huggingface", "hf"):
        key = os.getenv("HF_API_KEY", "")
        if not key:
            print("no-key"); sys.exit(0)
        from openai import OpenAI
        c = OpenAI(base_url="https://router.huggingface.co/hf-inference/v1", api_key=key)
        model = os.getenv("HF_MODEL", "Qwen/Qwen2.5-72B-Instruct")
        r = c.chat.completions.create(model=model,
            messages=[{"role":"user","content":"Reply with exactly: OK"}],
            max_tokens=5)
        print("ok" if r.choices[0].message.content else "empty")
    else:
        print("unknown-provider")
except Exception as e:
    print(f"error: {e}")
PYEOF
  )
  case "$LLM_PROBE" in
    ok)              pass "LLM ($LLM_PROV) responded successfully"; LLM_STATUS="$LLM_PROV — connected" ;;
    no-key)          warn "LLM probe skipped (no API key for $LLM_PROV)"; LLM_STATUS="$LLM_PROV — no key" ;;
    unknown-provider) warn "LLM_PROVIDER='$LLM_PROV' not recognised"; LLM_STATUS="unknown provider" ;;
    *)               fail "LLM ($LLM_PROV) probe failed: $LLM_PROBE"; LLM_STATUS="$LLM_PROV — ERROR" ;;
  esac
else
  LLM_STATUS="$LLM_PROV (not probed)"
fi

# Roboflow probe
if [[ "$QUICK" == false && -n "$ROBOFLOW_KEY" ]]; then
  RF_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 \
    "https://api.roboflow.com/?api_key=$ROBOFLOW_KEY" 2>/dev/null || echo "000")
  if [[ "$RF_CODE" =~ ^(200|302) ]]; then
    pass "Roboflow API key valid (HTTP $RF_CODE)"
    ROBOFLOW_STATUS="connected"
  else
    fail "Roboflow API key invalid or unreachable (HTTP $RF_CODE)"
    ROBOFLOW_STATUS="ERROR ($RF_CODE)"
  fi
elif [[ -n "$ROBOFLOW_KEY" ]]; then
  ROBOFLOW_STATUS="key set (not probed)"
else
  ROBOFLOW_STATUS="not configured"
fi

# =============================================================================
# 4. DATABASE — READ + WRITE TEST
# =============================================================================
section "Database (Supabase/Postgres)"

if [[ -n "$DB_URL" && -n "$PY" ]]; then
  DB_PROBE=$("$PY" - <<PYEOF 2>/dev/null
import asyncio, sys
async def check():
    try:
        import asyncpg
        url = "$DB_URL".replace("postgresql+asyncpg://", "postgresql://")
        conn = await asyncpg.connect(url, timeout=8)
        # Read test
        await conn.fetchval("SELECT 1")
        # Write test — temp table
        await conn.execute("CREATE TEMP TABLE _ablation_test (id serial, val text)")
        await conn.execute("INSERT INTO _ablation_test (val) VALUES ('ablation')")
        result = await conn.fetchval("SELECT val FROM _ablation_test LIMIT 1")
        await conn.execute("DROP TABLE _ablation_test")
        await conn.close()
        if result == "ablation":
            print("read-write")
        else:
            print("read-only")
    except asyncpg.InvalidPasswordError:
        print("auth-failed")
    except (asyncpg.CannotConnectNowError, OSError, asyncio.TimeoutError):
        print("unreachable")
    except Exception as e:
        print(f"error: {e}")
asyncio.run(check())
PYEOF
  )
  case "$DB_PROBE" in
    read-write)  pass "Database: connected, read ✓, write ✓"; DB_STATUS="Supabase (read/write OK)" ;;
    read-only)   warn "Database: connected, read OK but write test failed"; DB_STATUS="connected (read-only?)" ;;
    auth-failed) fail "Database: authentication failed — check DATABASE_URL credentials"; DB_STATUS="auth failed" ;;
    unreachable) fail "Database: unreachable — is Supabase project running?"; DB_STATUS="unreachable" ;;
    *)           fail "Database probe error: $DB_PROBE"; DB_STATUS="ERROR" ;;
  esac
else
  warn "DATABASE_URL or python3 not available — skipping DB probe"
  DB_STATUS="not probed"
fi

# =============================================================================
# 5. GPU & WORKER DETECTION
# =============================================================================
section "GPU & worker"

RUNPOD_URL="${DASHBOARD_URL:-}"
MODELS_DIR="$ROOT/backend/pipeline/models"

if [[ -n "$RUNPOD_URL" ]]; then
  pass "DASHBOARD_URL set — RunPod worker mode: $RUNPOD_URL"
  GPU_STATUS="RunPod (remote)"
else
  # Check for local GPU
  if command -v nvidia-smi &>/dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "")
    if [[ -n "$GPU_INFO" ]]; then
      pass "Local GPU detected: $GPU_INFO"
      GPU_STATUS="Local GPU ($GPU_INFO)"
    else
      warn "nvidia-smi found but no GPU info returned"
      GPU_STATUS="GPU detection unclear"
    fi
  elif [[ -n "$PY" ]]; then
    TORCH_GPU=$("$PY" -c "
try:
    import torch
    print('cuda' if torch.cuda.is_available() else 'cpu')
except ImportError:
    print('no-torch')
" 2>/dev/null || echo "no-torch")
    case "$TORCH_GPU" in
      cuda)     pass "Local GPU: PyTorch CUDA available"; GPU_STATUS="Local GPU (CUDA via torch)" ;;
      cpu)      warn "No GPU: PyTorch found but CUDA unavailable — worker will be CPU-only (very slow)"; GPU_STATUS="CPU only (no GPU)" ;;
      no-torch) warn "No GPU check possible — torch not installed. Set DASHBOARD_URL for RunPod or install torch"; GPU_STATUS="unknown (no torch)" ;;
    esac
  else
    warn "No GPU detected and no python3 available — set DASHBOARD_URL for RunPod"
    GPU_STATUS="unknown"
  fi
fi

# Model files
MODEL_URL_PLAYER="${MODEL_URL_PLAYER:-}"
MODEL_URL_BALL="${MODEL_URL_BALL:-}"
MODEL_URL_PITCH="${MODEL_URL_PITCH:-}"
[[ -n "$MODEL_URL_PLAYER" ]] && pass "MODEL_URL_PLAYER set" || warn "MODEL_URL_PLAYER not set"
[[ -n "$MODEL_URL_BALL" ]]   && pass "MODEL_URL_BALL set"   || warn "MODEL_URL_BALL not set"
[[ -n "$MODEL_URL_PITCH" ]]  && pass "MODEL_URL_PITCH set"  || warn "MODEL_URL_PITCH not set (pitch detection off)"

if [[ -d "$MODELS_DIR" ]]; then
  MODEL_COUNT=$(find "$MODELS_DIR" -name "*.pt" 2>/dev/null | wc -l | tr -d ' ')
  if [[ "$MODEL_COUNT" -gt 0 ]]; then
    pass "Model files: $MODEL_COUNT .pt file(s) found in pipeline/models/"
  else
    warn "No .pt model files in pipeline/models/ — worker will download on first run"
  fi
else
  warn "pipeline/models/ directory missing — will be created on first run"
fi

# =============================================================================
# 6. NETWORK & SERVICES  (skipped with --quick)
# =============================================================================
if [[ "$QUICK" == false ]]; then
  section "Backend (FastAPI)"

  BACKEND_URL="${BACKEND_URL:-http://localhost:8000}"
  BE_HEALTH=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "$BACKEND_URL/api/health" 2>/dev/null || echo "000")
  if [[ "$BE_HEALTH" =~ ^(200|404) ]]; then
    # 404 is fine — health endpoint may not exist but server is up
    pass "Backend running at $BACKEND_URL"
    BACKEND_STATUS="running"
    probe "Backend /docs"       "$BACKEND_URL/docs" > /dev/null
    probe "Backend /api/videos" "$BACKEND_URL/api/videos" > /dev/null
  else
    fail "Backend not reachable at $BACKEND_URL (HTTP $BE_HEALTH) — run: npm run dev:api"
    BACKEND_STATUS="not running"
  fi

  section "Frontend (Vite)"

  FRONTEND_URL="${FRONTEND_URL:-http://localhost:5173}"
  FE_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "$FRONTEND_URL" 2>/dev/null || echo "000")
  if [[ "$FE_CODE" =~ ^(200|304) ]]; then
    pass "Frontend running at $FRONTEND_URL"
    FRONTEND_STATUS="running"
  else
    warn "Frontend not running (HTTP $FE_CODE) — run: npm run dev"
    FRONTEND_STATUS="not running"
  fi

  section "Ngrok & RunPod"

  NGROK_API="${NGROK_API:-http://localhost:4040}"
  if curl -s --max-time 3 "$NGROK_API/api/tunnels" &>/dev/null; then
    NGROK_URL=$([[ -n "$PY" ]] && "$PY" -c "
import sys, json, urllib.request
try:
    data = json.loads(urllib.request.urlopen('$NGROK_API/api/tunnels', timeout=3).read())
    tunnels = data.get('tunnels', [])
    https = [t for t in tunnels if t.get('proto') == 'https']
    print((https or tunnels)[0]['public_url'] if tunnels else '')
except: print('')
" 2>/dev/null || echo "")

    if [[ -n "$NGROK_URL" ]]; then
      pass "Ngrok tunnel active: $NGROK_URL"
      NGROK_STATUS="active"
      # Probe backend through ngrok
      NGROK_BE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 8 "$NGROK_URL/api/videos" 2>/dev/null || echo "000")
      if [[ "$NGROK_BE" =~ ^(200|401|403) ]]; then
        pass "Backend reachable through ngrok (HTTP $NGROK_BE)"
      else
        fail "Backend not reachable through ngrok (HTTP $NGROK_BE) — is backend running?"
      fi
    else
      warn "Ngrok running but no active tunnels"
      NGROK_STATUS="running, no tunnels"
    fi
  else
    warn "Ngrok not running — RunPod worker cannot reach backend"
    info "Start with: ngrok http 8000"
    NGROK_STATUS="not running"
  fi

  if [[ -n "$RUNPOD_URL" ]]; then
    RPOD_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 8 "$RUNPOD_URL/api/worker/pending" 2>/dev/null || echo "000")
    if [[ "$RPOD_CODE" =~ ^(200|401|403) ]]; then
      pass "RunPod → backend /api/worker/pending reachable (HTTP $RPOD_CODE)"
    else
      fail "RunPod cannot reach backend at DASHBOARD_URL=$RUNPOD_URL (HTTP $RPOD_CODE)"
    fi
  else
    info "DASHBOARD_URL not set — skipping RunPod→backend probe"
  fi

  # WebSocket handshake test
  section "WebSocket"
  if command -v wscat &>/dev/null; then
    WS_URL=$(echo "$BACKEND_URL" | sed 's|^http|ws|')
    if echo "" | wscat -c "${WS_URL}/ws/ablation-test" --timeout 3 &>/dev/null; then
      pass "WebSocket endpoint reachable"
    else
      warn "WebSocket not reachable (backend may not be running)"
    fi
  elif [[ -n "$PY" ]]; then
    WS_CHECK=$("$PY" -c "
import socket, sys
try:
    s = socket.create_connection(('localhost', 8000), timeout=3)
    s.close()
    print('port-open')
except: print('closed')
" 2>/dev/null)
    [[ "$WS_CHECK" == "port-open" ]] && pass "Backend port 8000 open (WS endpoint available)" \
                                      || warn "Backend port 8000 closed — WebSocket unavailable"
  else
    warn "wscat and python3 unavailable — skipping WebSocket check"
  fi
fi

# =============================================================================
# 7. BUILD & TESTS  (skipped with --quick or --skip-build)
# =============================================================================
if [[ "$QUICK" == false && "$SKIP_BUILD" == false ]]; then
  section "Frontend — lint & typecheck"

  cd "$ROOT/frontend"
  TS_OUT=$(pnpm tsc --noEmit 2>&1 || true)
  TS_ERRORS=$(echo "$TS_OUT" | grep -c "error TS" || true)
  if [[ "$TS_ERRORS" -eq 0 ]]; then
    pass "TypeScript: no errors"
  else
    fail "TypeScript: $TS_ERRORS error(s)"
    echo "$TS_OUT" | grep "error TS" | head -10 | sed 's/^/    /'
  fi

  LINT_OUT=$(pnpm lint 2>&1 || true)
  LINT_ERRORS=$(echo "$LINT_OUT" | grep -c " error " || true)
  if [[ "$LINT_ERRORS" -eq 0 ]]; then
    pass "ESLint: no errors"
  else
    fail "ESLint: $LINT_ERRORS error(s)"
  fi

  section "Frontend — build"
  if pnpm build > /tmp/ablation-build.log 2>&1; then
    pass "Vite production build succeeded"
    BUILD_STATUS="OK"
  else
    fail "Vite build failed — see /tmp/ablation-build.log"
    tail -15 /tmp/ablation-build.log | sed 's/^/    /'
    BUILD_STATUS="FAILED"
  fi

  section "Frontend — unit tests"
  TEST_OUT=$(pnpm test --run 2>&1 || true)
  TEST_PASS=$(echo "$TEST_OUT" | grep -oE "[0-9]+ passed" | head -1 || true)
  TEST_FAIL=$(echo "$TEST_OUT" | grep -oE "[0-9]+ failed" | head -1 || true)
  if [[ -n "$TEST_FAIL" ]]; then
    fail "Vitest: $TEST_FAIL"
    TEST_STATUS="$TEST_PASS, $TEST_FAIL"
  elif [[ -n "$TEST_PASS" ]]; then
    pass "Vitest: $TEST_PASS"
    TEST_STATUS="$TEST_PASS"
  else
    warn "Vitest: no test files found"
    TEST_STATUS="no tests"
  fi

  cd "$ROOT"

  section "Backend — lint & tests"
  cd "$ROOT/backend"

  if command -v ruff &>/dev/null; then
    if ruff check api/ > /tmp/ablation-ruff.log 2>&1; then
      pass "Ruff: no lint errors"
    else
      RUFF_N=$(grep -c "." /tmp/ablation-ruff.log || true)
      fail "Ruff: $RUFF_N issue(s) — see /tmp/ablation-ruff.log"
    fi
  else
    warn "ruff not installed — skipping Python lint"
  fi

  if [[ -n "$PY" ]]; then
    if [[ -d api/tests ]]; then
      if "$PY" -m pytest api/tests/ -q --tb=short > /tmp/ablation-pytest-api.log 2>&1; then
        PT=$(grep -oE "[0-9]+ passed" /tmp/ablation-pytest-api.log | head -1 || echo "passed")
        pass "API tests: $PT"
      else
        fail "API tests failed — see /tmp/ablation-pytest-api.log"
        tail -10 /tmp/ablation-pytest-api.log | sed 's/^/    /'
      fi
    else
      warn "backend/api/tests/ not found"
    fi

    if [[ -d tests ]]; then
      if "$PY" -m pytest tests/ -q --tb=short > /tmp/ablation-pytest-pipeline.log 2>&1; then
        PT=$(grep -oE "[0-9]+ passed" /tmp/ablation-pytest-pipeline.log | head -1 || echo "passed")
        pass "Pipeline tests: $PT"
      else
        fail "Pipeline tests failed — see /tmp/ablation-pytest-pipeline.log"
        tail -10 /tmp/ablation-pytest-pipeline.log | sed 's/^/    /'
      fi
    else
      warn "backend/tests/ not found"
    fi
  fi

  cd "$ROOT"
fi

# =============================================================================
# SUMMARY DASHBOARD
# =============================================================================
WIDTH=50
line() { printf '━%.0s' $(seq 1 $WIDTH); echo; }

echo ""
echo -e "${BOLD}$(line)${RESET}"
echo -e "${BOLD}  ABLATION REPORT${RESET}"
echo -e "${BOLD}$(line)${RESET}"

pad() { printf "  %-18s %s\n" "$1" "$2"; }

pad "Auth mode:"    "$AUTH_MODE"
pad "Database:"     "$DB_STATUS"
pad "LLM provider:" "$LLM_STATUS"
pad "Roboflow:"     "$ROBOFLOW_STATUS"
pad "GPU / worker:" "$GPU_STATUS"
pad "Ngrok:"        "$NGROK_STATUS${NGROK_URL:+ ($NGROK_URL)}"
pad "Backend:"      "$BACKEND_STATUS"
pad "Frontend:"     "$FRONTEND_STATUS"
pad "Build:"        "$BUILD_STATUS"
pad "Tests:"        "$TEST_STATUS"

echo -e "${BOLD}$(line)${RESET}"
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

echo -e "${BOLD}$(line)${RESET}"
echo ""

if [[ "$QUICK" == true ]]; then
  echo -e "  ${DIM}Tip: run without --quick to probe live services${RESET}"
elif [[ "$SKIP_BUILD" == true ]]; then
  echo -e "  ${DIM}Tip: run without --skip-build to also run frontend build & tests${RESET}"
fi
echo -e "  ${DIM}Pipeline E2E: bash scripts/pipeline-e2e.sh --video 7${RESET}"

echo ""
if [[ $FAIL -gt 0 ]]; then
  echo -e "${RED}${BOLD}SYSTEMS NOT READY${RESET} — $FAIL critical check(s) failed"
  exit 1
elif [[ $WARN -gt 0 ]]; then
  echo -e "${YELLOW}${BOLD}MOSTLY READY${RESET} — $WARN warning(s), but no failures"
  exit 0
else
  echo -e "${GREEN}${BOLD}ALL SYSTEMS GO${RESET}"
  exit 0
fi
