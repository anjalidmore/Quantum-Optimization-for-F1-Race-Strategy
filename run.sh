#!/usr/bin/env bash
# run.sh — one-command demo of the F1 Race Strategy Intelligence platform.
#
# Ensures models are trained, starts the backend API and the frontend dev
# server, opens the frontend in your browser, and prints a few real
# predictions from the trained models straight to the terminal so you can
# see the ML pipeline actually working end to end.
#
# Usage: ./run.sh [--force-retrain]
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Overridable, so a contributor with something already on 8000/3000 can move
# this project out of the way instead of killing their process:
#   BACKEND_PORT=8001 FRONTEND_PORT=3001 ./run.sh
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"
BACKEND_URL="http://127.0.0.1:${BACKEND_PORT}"
FRONTEND_URL="http://127.0.0.1:${FRONTEND_PORT}"
BACKEND_LOG="/tmp/f1_backend.log"
FRONTEND_LOG="/tmp/f1_frontend.log"

FORCE_RETRAIN=0
FORCE_PORTS=0
for arg in "$@"; do
  case "$arg" in
    --force-retrain) FORCE_RETRAIN=1 ;;
    --force-ports)   FORCE_PORTS=1 ;;
    -h|--help)
      cat <<'USAGE'
Usage: ./run.sh [--force-retrain] [--force-ports]

  --force-retrain  Retrain all models before starting.
  --force-ports    Kill whatever is listening on BACKEND_PORT/FRONTEND_PORT
                   without asking. Without this flag the script asks first,
                   and refuses in a non-interactive shell.

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

for port in "$BACKEND_PORT" "$FRONTEND_PORT"; do
  pid=$(lsof -ti:"$port" -sTCP:LISTEN 2>/dev/null || true)
  [[ -z "$pid" ]] && continue

  # Show the caller *what* they would be killing, not just a bare pid.
  desc=$(ps -p "$pid" -o comm=,args= 2>/dev/null | head -1 | cut -c1-100)
  warn "Port $port is in use by pid $pid: ${desc:-unknown process}"

  if [[ $FORCE_PORTS -eq 1 ]]; then
    info "Stopping it (--force-ports)..."
    kill "$pid" 2>/dev/null || true
    sleep 1
    continue
  fi

  if [[ ! -t 0 ]]; then
    echo "Refusing to kill pid $pid on port $port in a non-interactive shell." >&2
    echo "Re-run with --force-ports, or set BACKEND_PORT/FRONTEND_PORT to free ports." >&2
    exit 1
  fi

  read -r -p "Kill pid $pid to free port $port? [y/N] " reply
  if [[ "$reply" =~ ^[Yy]$ ]]; then
    kill "$pid" 2>/dev/null || true
    sleep 1
  else
    echo "Leaving pid $pid alone. Set BACKEND_PORT/FRONTEND_PORT to use different ports:" >&2
    echo "  BACKEND_PORT=8001 FRONTEND_PORT=3001 ./run.sh" >&2
    exit 1
  fi
done

# ---------------------------------------------------------------------------
# 3. Train models if they don't exist yet (or --force-retrain was passed)
# ---------------------------------------------------------------------------
if [[ $FORCE_RETRAIN -eq 1 ]]; then
  info "Retraining all models (--force-retrain)..."
  python scripts/build_all.py --force
elif [[ ! -f artifacts/metadata/model_registry.json ]]; then
  info "No trained models found — running the full build pipeline..."
  python scripts/build_all.py
else
  ok "Trained models already present (artifacts/metadata/model_registry.json)"
fi

# ---------------------------------------------------------------------------
# 4. Start the backend API
# ---------------------------------------------------------------------------
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
# 5. Start the frontend
# ---------------------------------------------------------------------------
if [[ ! -d frontend/node_modules ]]; then
  info "Installing frontend dependencies (first run only)..."
  (cd frontend && npm install)
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
# 7. Open the frontend
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
