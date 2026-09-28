#!/usr/bin/env bash
# run.sh — one command that runs the whole platform and opens the dashboard.
#
# Builds every stage that is missing (Tasks 1-8 plus the quantum models),
# starts the backend API, warms it so the first page load shows real numbers
# instead of a cold error, starts the frontend, prints a few real predictions
# to the terminal, and opens the dashboard in your browser.
#
# Usage: ./run.sh [--force-retrain] [--force-ports] [--skip-qml]
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Overridable, so a contributor with something already on 8000/3000 can move
# this project out of the way instead of killing their process:
#   BACKEND_PORT=8001 FRONTEND_PORT=3001 ./run.sh
# Whether the caller picked these, which decides what happens when the port
# turns out to be taken: a port you asked for is honoured or the script stops,
# a default is just a starting point and may move.
BACKEND_PORT_EXPLICIT=${BACKEND_PORT:+1}
FRONTEND_PORT_EXPLICIT=${FRONTEND_PORT:+1}
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"
BACKEND_LOG="/tmp/f1_backend.log"
FRONTEND_LOG="/tmp/f1_frontend.log"

FORCE_RETRAIN=0
FORCE_PORTS=0
SKIP_QML=0
for arg in "$@"; do
  case "$arg" in
    --force-retrain) FORCE_RETRAIN=1 ;;
    --force-ports)   FORCE_PORTS=1 ;;
    --skip-qml)      SKIP_QML=1 ;;
    -h|--help)
      cat <<'USAGE'
Usage: ./run.sh [--force-retrain] [--force-ports] [--skip-qml]

  --force-retrain  Rebuild every stage from scratch before starting.
  --force-ports    Kill whatever is listening on BACKEND_PORT/FRONTEND_PORT
                   without asking. Without this flag the script asks first,
                   and refuses in a non-interactive shell.
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

info()  { printf "\033[1;34m==>\033[0m %s\n" "$1"; }
ok()    { printf "\033[1;32m✓\033[0m %s\n" "$1"; }

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
# 2. Free the ports this script needs
#
# This used to kill whatever was listening, with no prompt and no check that
# the process belonged to this project - a contributor running an unrelated
# dev server on :3000 lost it silently, with unsaved state. Now the script
# shows what it found and asks; --force-ports restores the old behaviour, and
# BACKEND_PORT/FRONTEND_PORT let you avoid the collision entirely.
# ---------------------------------------------------------------------------
warn() { printf "\033[1;33m!\033[0m %s\n" "$1"; }

# A port can have more than one listener - a stale dev server and Docker's
# port proxy, say - so this walks them one at a time. It used to hold them
# all in one variable and pass that to `ps`, which failed on the embedded
# newline and, under `set -e`, killed the script with no message at all.
other_port_advice() {
  echo "Set BACKEND_PORT/FRONTEND_PORT to use different ports:" >&2
  echo "  BACKEND_PORT=8001 FRONTEND_PORT=3001 ./run.sh" >&2
}

# Is this pid one of our own servers from an earlier ./run.sh?
#
# The backend is recognised by its uvicorn target and the frontend by its
# working directory, which lsof reports even though `next-server` rewrites
# its own argv to something that says nothing about which project it serves.
is_own_server() {
  local pid="$1" args cwd
  args=$(ps -p "$pid" -o args= 2>/dev/null || true)
  [[ "$args" == *app.api.main:app* ]] && return 0
  cwd=$(lsof -a -p "$pid" -d cwd -Fn 2>/dev/null | sed -n 's/^n//p' | head -1 || true)
  [[ -n "$cwd" && "$cwd" == "$ROOT_DIR"* ]] && return 0
  return 1
}

# Return a port we can actually bind, starting from the one asked for.
#
# Everything it says goes to stderr, because stdout is the resolved port.
resolve_port() {
  local label="$1" port="$2" explicit="$3"
  local pids pid args blocked

  while :; do
    pids=$(lsof -ti:"$port" -sTCP:LISTEN 2>/dev/null || true)
    if [[ -z "$pids" ]]; then
      echo "$port"
      return 0
    fi

    blocked=0
    # A plain for-loop, not `while read ... <<< "$pids"`: a here-string is the
    # loop body's stdin, so the tty check below would see a pipe and the
    # prompt would read a pid instead of the answer. Pids are
    # whitespace-separated integers, so word splitting is exactly right.
    for pid in $pids; do
      args=$(ps -p "$pid" -o args= 2>/dev/null | head -1 | cut -c1-100 || true)

      # Our own leftovers get restarted without asking. Re-running ./run.sh is
      # a request for a fresh pair of servers, and making you confirm the
      # death of the server the last ./run.sh started is noise, not safety.
      if is_own_server "$pid"; then
        info "Restarting this project's own $label server on port $port (pid $pid)..." >&2
        kill "$pid" 2>/dev/null || true
        sleep 1
        continue
      fi

      warn "Port $port is in use by pid $pid: ${args:-unknown process}" >&2

      # Shared infrastructure that happens to hold the port. Killing Docker's
      # proxy to free :3000 breaks every other container the user has
      # running, which is never a fair trade for a demo script.
      if [[ "$args" == *docker* || "$args" == *Docker* ]]; then
        warn "That is Docker, so this script will not kill it." >&2
        blocked=1
        break
      fi

      if [[ $FORCE_PORTS -eq 1 ]]; then
        info "Stopping it (--force-ports)..." >&2
        kill "$pid" 2>/dev/null || true
        sleep 1
        continue
      fi

      if [[ ! -t 0 ]]; then
        echo "Refusing to kill pid $pid on port $port in a non-interactive shell." >&2
        echo "Re-run with --force-ports, or set the port variable for the $label to a free port." >&2
        exit 1
      fi

      read -r -p "Kill pid $pid to free port $port? [y/N] " reply
      if [[ "$reply" =~ ^[Yy]$ ]]; then
        kill "$pid" 2>/dev/null || true
        sleep 1
      else
        blocked=1
        break
      fi
    done

    if [[ $blocked -eq 0 ]]; then
      echo "$port"
      return 0
    fi

    # You named this port, so the script does not quietly serve a different
    # one. An untouched default is only a starting point, and moving on beats
    # refusing to run at all.
    if [[ -n "$explicit" ]]; then
      echo "Port $port was requested but cannot be freed." >&2
      other_port_advice
      exit 1
    fi

    port=$((port + 1))
    info "Moving the $label to port $port instead." >&2
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
ok "Frontend ready (pid $FRONTEND_PID, log: $FRONTEND_LOG)"

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
info "The dashboard has six pages:"
printf "     %-18s %s\n" \
  "Overview"       "$FRONTEND_URL/" \
  "Strategy"       "$FRONTEND_URL/strategy" \
  "Reasoning"      "$FRONTEND_URL/reasoning" \
  "Models"         "$FRONTEND_URL/models" \
  "Explainability" "$FRONTEND_URL/explainability" \
  "Data"           "$FRONTEND_URL/data"
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
