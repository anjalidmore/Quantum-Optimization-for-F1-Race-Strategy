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

BACKEND_PORT=8000
FRONTEND_PORT=3000
BACKEND_URL="http://127.0.0.1:${BACKEND_PORT}"
FRONTEND_URL="http://127.0.0.1:${FRONTEND_PORT}"
BACKEND_LOG="/tmp/f1_backend.log"
FRONTEND_LOG="/tmp/f1_frontend.log"

FORCE_RETRAIN=0
if [[ "${1:-}" == "--force-retrain" ]]; then
  FORCE_RETRAIN=1
fi

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
# 2. Free the ports this script needs, so re-running is always idempotent
# ---------------------------------------------------------------------------
for port in "$BACKEND_PORT" "$FRONTEND_PORT"; do
  pid=$(lsof -ti:"$port" -sTCP:LISTEN 2>/dev/null || true)
  if [[ -n "$pid" ]]; then
    info "Stopping existing process on port $port (pid $pid)..."
    kill "$pid" 2>/dev/null || true
    sleep 1
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
