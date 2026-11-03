#!/usr/bin/env bash
# run.sh — one command that runs the whole platform and opens the dashboard.
#
# Builds every stage that is missing (Tasks 1-8 plus the quantum models),
# starts the backend API, warms it so the first page load shows real numbers
# instead of a cold error, starts the frontend, prints a few real predictions
# to the terminal, and opens the dashboard in your browser.
#
# Usage: ./run.sh [--force-retrain] [--skip-qml]
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Overridable, so a contributor with something already on 8000/3000 can move
# this project out of the way instead of killing their process:
#   BACKEND_PORT=8001 FRONTEND_PORT=3001 ./run.sh
# Whether the caller picked these, which decides what happens when the port
# turns out to be taken: a port you asked for is honoured or the script stops,
# a default moves to the next free one.
BACKEND_PORT_EXPLICIT=${BACKEND_PORT:+1}
FRONTEND_PORT_EXPLICIT=${FRONTEND_PORT:+1}
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"
BACKEND_LOG="/tmp/f1_backend.log"
FRONTEND_LOG="/tmp/f1_frontend.log"

FORCE_RETRAIN=0
SKIP_QML=0
for arg in "$@"; do
  case "$arg" in
    --force-retrain) FORCE_RETRAIN=1 ;;
    --skip-qml)      SKIP_QML=1 ;;
    -h|--help)
      cat <<'USAGE'
Usage: ./run.sh [--force-retrain] [--skip-qml]

  --force-retrain  Rebuild every stage from scratch before starting.
  --skip-qml       Skip the quantum stage (it needs pennylane, and it is the
                   slowest thing in a cold build).

Environment:
  BACKEND_PORT   (default 8000)
  FRONTEND_PORT  (default 3000)
USAGE
      exit 0 ;;
    *) echo "Unknown argument: $arg (try --help)" >&2; exit 2 ;;
  esac
done

# Status is carried by a coloured word, not by a glyph: a tick and a bang say
# nothing when the terminal has no colour, and nothing at all to a screen reader.
info()  { printf "\033[1;34m==>\033[0m %s\n" "$1"; }
ok()    { printf "\033[1;32mOK\033[0m    %s\n" "$1"; }

# ---------------------------------------------------------------------------
# 1. Python environment
# ---------------------------------------------------------------------------
if [[ ! -d .venv ]]; then
  info "Creating Python virtual environment..."
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

if ! python -c "import fastapi, sklearn, pandas" >/dev/null 2>&1; then
  info "Installing Python dependencies (first run only)..."
  pip install -q -r requirements.txt
fi
pip show f1-quantum-strategy >/dev/null 2>&1 || pip install -q -e . >/dev/null
ok "Python environment ready ($(python --version))"

# ---------------------------------------------------------------------------
# 2. Choose ports
#
# This project's own servers from an earlier ./run.sh are restarted. Anything
# else holding a port is left running: the script never kills a process it
# does not own.
# ---------------------------------------------------------------------------
# The backend is recognised by its uvicorn target and the frontend by its
# working directory, because `next-server` rewrites its argv to something that
# does not name the project.
is_own_server() {
  local pid="$1" args cwd
  args=$(ps -p "$pid" -o args= 2>/dev/null || true)
  [[ "$args" == *app.api.main:app* ]] && return 0
  cwd=$(lsof -a -p "$pid" -d cwd -Fn 2>/dev/null | sed -n 's/^n//p' | head -1 || true)
  [[ -n "$cwd" && "$cwd" == "$ROOT_DIR"* ]] && return 0
  return 1
}

# Everything this says goes to stderr, because stdout is the resolved port.
resolve_port() {
  local label="$1" port="$2" explicit="$3" pid

  while :; do
    for pid in $(lsof -ti:"$port" -sTCP:LISTEN 2>/dev/null || true); do
      if is_own_server "$pid"; then
        info "Restarting this project's $label server on port $port..." >&2
        kill "$pid" 2>/dev/null || true
        sleep 1
      fi
    done

    if [[ -z "$(lsof -ti:"$port" -sTCP:LISTEN 2>/dev/null || true)" ]]; then
      echo "$port"
      return 0
    fi

    if [[ -n "$explicit" ]]; then
      echo "Port $port is held by another program. Choose a free one, for example:" >&2
      echo "  BACKEND_PORT=8001 FRONTEND_PORT=3001 ./run.sh" >&2
      exit 1
    fi

    info "Port $port is held by another program; trying $((port + 1)) for the $label." >&2
    port=$((port + 1))
  done
}

BACKEND_PORT=$(resolve_port backend "$BACKEND_PORT" "$BACKEND_PORT_EXPLICIT")
FRONTEND_PORT=$(resolve_port frontend "$FRONTEND_PORT" "$FRONTEND_PORT_EXPLICIT")
BACKEND_URL="http://127.0.0.1:${BACKEND_PORT}"
FRONTEND_URL="http://127.0.0.1:${FRONTEND_PORT}"

# ---------------------------------------------------------------------------
# 3. Build every stage the dashboard reads
#
# build_all.py is incremental: each stage checks for its own artifacts and
# logs "Already built" instead of redoing the work. So we always run it,
# rather than gating on the ML registry alone. That gate meant a checkout
# with trained models but no XAI or quantum artifacts started anyway, and
# the pages for those tasks read "Not generated yet" with nothing to say why.
# ---------------------------------------------------------------------------
# A plain string, not an array: these flags never contain spaces, and an
# empty array expanded under `set -u` is a portability trap on bash 3.2,
# which is still what ships with macOS.
BUILD_ARGS=""
[[ $FORCE_RETRAIN -eq 1 ]] && BUILD_ARGS="$BUILD_ARGS --force"
[[ $SKIP_QML -eq 1 ]] && BUILD_ARGS="$BUILD_ARGS --skip-qml"

if [[ $FORCE_RETRAIN -eq 1 ]]; then
  info "Rebuilding every stage (--force-retrain). This takes a while."
else
  info "Building anything that is missing (Tasks 1-8 and the quantum models)..."
fi
# shellcheck disable=SC2086  # deliberate word splitting of the flag list
python scripts/build_all.py $BUILD_ARGS

# ---------------------------------------------------------------------------
# 4. Start the backend API
# ---------------------------------------------------------------------------
# The API's CORS allow-list defaults to :3000. The moment the dashboard lands
# anywhere else - because Docker holds :3000, or you set FRONTEND_PORT - every
# fetch the browser makes is blocked and the client-rendered pages show their
# "Backend unreachable" state while the server-rendered ones look fine. So the
# port the dashboard actually got is added to the list the backend starts with.
# An allow-list you set yourself is kept, with this origin appended.
FRONTEND_ORIGINS="http://127.0.0.1:${FRONTEND_PORT},http://localhost:${FRONTEND_PORT}"
if [[ -n "${F1_ALLOWED_ORIGINS:-}" ]]; then
  F1_ALLOWED_ORIGINS="${F1_ALLOWED_ORIGINS},${FRONTEND_ORIGINS}"
else
  F1_ALLOWED_ORIGINS="$FRONTEND_ORIGINS"
fi
export F1_ALLOWED_ORIGINS

info "Starting backend API on $BACKEND_URL ..."
nohup uvicorn app.api.main:app --host 127.0.0.1 --port "$BACKEND_PORT" \
  > "$BACKEND_LOG" 2>&1 &
BACKEND_PID=$!
disown "$BACKEND_PID"

for _ in $(seq 1 60); do
  curl -sf "$BACKEND_URL/api/health" >/dev/null 2>&1 && break
  sleep 0.5
done
if ! curl -sf "$BACKEND_URL/api/health" >/dev/null 2>&1; then
  echo "Backend failed to start — see $BACKEND_LOG" >&2
  tail -n 30 "$BACKEND_LOG" >&2
  exit 1
fi
ok "Backend ready (pid $BACKEND_PID, log: $BACKEND_LOG)"

# ---------------------------------------------------------------------------
# 4b. Warm the API
#
# The first request to each of these loads and caches a model or parses an
# artifact, and some take tens of seconds. Opening the dashboard cold meant
# the first page you looked at rendered its error or empty state while the
# backend was still thinking. Warming here costs the same time once, before
# the browser opens, instead of in front of you.
# ---------------------------------------------------------------------------
info "Warming the API so the first page load shows real numbers..."
WARM_ENDPOINTS=(
  /api/health
  /api/tasks/evidence
  /api/data/options
  /api/ml/models  /api/ml/metrics  /api/ml/comparison  /api/ml/feature-importance
  /api/dl/models  /api/dl/metrics  /api/dl/comparison  /api/dl/history
  /api/reasoning/knowledge  /api/reasoning/expert-system  /api/reasoning/search
  /api/xai/summary  /api/xai/fairness  /api/xai/stratification
  /api/qml/summary
)
for endpoint in "${WARM_ENDPOINTS[@]}"; do
  # A 404 is fine and expected when a stage was skipped; we only care that
  # the process has paid the load cost, so the status is not checked.
  curl -s -o /dev/null -m 180 "${BACKEND_URL}${endpoint}" || true
  printf "."
done
echo
ok "API warm (${#WARM_ENDPOINTS[@]} endpoints)"

# ---------------------------------------------------------------------------
# 5. Start the frontend
# ---------------------------------------------------------------------------
if [[ ! -d frontend/node_modules ]]; then
  info "Installing frontend dependencies (first run only)..."
  (cd frontend && npm install)
fi

# NEXT_PUBLIC_* values are inlined when the app is compiled, so a production
# build left behind by `npm run build` has the old API URL baked into it and
# `next dev` will not override it. Clear it and let dev rebuild.
if [[ -f frontend/.next/BUILD_ID ]]; then
  info "Clearing a stale production build in frontend/.next ..."
  rm -rf frontend/.next
fi

info "Starting frontend on $FRONTEND_URL ..."
NEXT_PUBLIC_API_BASE_URL="$BACKEND_URL" nohup npm --prefix frontend run dev -- -p "$FRONTEND_PORT" \
  > "$FRONTEND_LOG" 2>&1 &
FRONTEND_PID=$!
disown "$FRONTEND_PID"

for _ in $(seq 1 60); do
  curl -sf "$FRONTEND_URL" >/dev/null 2>&1 && break
  sleep 0.5
done
if ! curl -sf "$FRONTEND_URL" >/dev/null 2>&1; then
  echo "Frontend failed to start — see $FRONTEND_LOG" >&2
  tail -n 30 "$FRONTEND_LOG" >&2
  exit 1
fi
# `next dev` compiles a page on its first visit, which is several seconds in
# front of an audience unless it has already happened.
info "Compiling the dashboard pages..."
PAGES=(strategy data reasoning models model-comparison explainability rai)
for page in "${PAGES[@]}"; do
  curl -s -o /dev/null -m 180 "$FRONTEND_URL/$page" || true
  printf "."
done
echo
ok "Frontend ready (pid $FRONTEND_PID, log: $FRONTEND_LOG)"

# Compiling those pages sends the backend its first real traffic. A native crash
# leaves no line in its log, so say here that it died rather than let the next
# step fail with a connection traceback that points at the wrong script.
if ! curl -sf "$BACKEND_URL/api/health" >/dev/null 2>&1; then
  echo "The backend stopped while the dashboard pages were compiling." >&2
  echo "Its last log lines are below. If they show no error it crashed natively;" >&2
  echo "macOS keeps a report in ~/Library/Logs/DiagnosticReports." >&2
  tail -n 15 "$BACKEND_LOG" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# 6. Prove the ML models actually work: run a few real predictions
#
# Feature payloads are built dynamically from whatever the live model
# registry says the selected features are (see scripts/demo_predict.py) —
# never hard-coded here, since the exact feature list depends on whether the
# platform was trained on the synthetic demo data or a real fetched session.
# ---------------------------------------------------------------------------
python scripts/demo_predict.py --base-url "$BACKEND_URL"

# ---------------------------------------------------------------------------
# 7. Open the dashboard
# ---------------------------------------------------------------------------
echo
info "Opening $FRONTEND_URL in your browser..."
if command -v open >/dev/null 2>&1; then
  open "$FRONTEND_URL"
elif command -v xdg-open >/dev/null 2>&1; then
  xdg-open "$FRONTEND_URL"
else
  echo "Open this URL manually: $FRONTEND_URL"
fi

echo
ok "Backend:  $BACKEND_URL  (docs at $BACKEND_URL/docs, log: $BACKEND_LOG, pid $BACKEND_PID)"
ok "Frontend: $FRONTEND_URL  (log: $FRONTEND_LOG, pid $FRONTEND_PID)"
echo
echo "Stop both with:"
echo "  kill $BACKEND_PID $FRONTEND_PID"
