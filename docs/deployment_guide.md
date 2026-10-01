# Deployment guide

A standalone reference for installing, building, running, serving and troubleshooting the F1 Race
Strategy Intelligence platform — expanded from [`DEPLOYMENT.md`](DEPLOYMENT.md) with more detail on
environment variables, the choice between building from raw data and using the committed artifacts,
and what to do when something in the chain isn't there yet. Where `DEPLOYMENT.md` states a fact,
this document explains why it's true and what depends on it; read this one first if you're setting
up a new environment, `DEPLOYMENT.md` remains the terser reference.

## Contents

1. [Prerequisites](#1-prerequisites)
2. [First-time setup](#2-first-time-setup)
3. [Building the artifacts: from raw data vs. using what's committed](#3-building-the-artifacts-from-raw-data-vs-using-whats-committed)
4. [Running it](#4-running-it)
5. [Environment variables](#5-environment-variables)
6. [Serving it to other people](#6-serving-it-to-other-people)
7. [Health checks](#7-health-checks)
8. [Retraining on a different race](#8-retraining-on-a-different-race)
9. [Docker deployment](#9-docker-deployment)
10. [Troubleshooting](#10-troubleshooting)
11. [What to back up](#11-what-to-back-up)

## 1. Prerequisites

| Requirement | Version | Why |
|---|---|---|
| Python | 3.12+ (developed and its committed artifacts generated on 3.14.6) | the whole backend |
| Node.js | 20+ | the Next.js dashboard |
| `libomp` | any | XGBoost only; the build skips XGBoost with a stated reason ("XGBoost unavailable — skipped") if it is missing, rather than failing |
| pandoc | optional | converts a generated Markdown strategy report to PDF (`pandoc report.md -o report.pdf`) |

No database, no message queue, no cloud service. The system reads and writes plain files under
`data/` and `artifacts/`, and that is its entire persistence layer.

The Python dependency set that produces the committed metrics is version-pinned in
`requirements.txt` — notably `scikit-learn==1.9.0`, `xgboost==3.4.1`, `keras==3.15.1`,
`torch==2.14.0`, `shap==0.52.0`, `lime==0.2.0.1`. This is deliberate: an unpinned install could
silently drift the committed numbers on a future release of any of these libraries. If you need to
upgrade one, expect to rebuild and re-verify against `docs/TESTING_REPORT.md`'s recorded run rather
than assuming the numbers still hold.

**A note on the deep-learning stack.** Keras 3 is backend-agnostic, and this project runs it on the
**PyTorch** backend, not TensorFlow — TensorFlow currently publishes no wheel for Python 3.14
(`requirements.txt`'s own comment records this was verified with `pip index versions tensorflow`).
`app/core/runtime.py` sets `KERAS_BACKEND=torch` and, importantly, imports XGBoost *before* Keras/
PyTorch to claim the OpenMP runtime first — PyTorch ships its own copy of `libomp`, and importing
XGBoost after PyTorch has already initialized it can segfault on macOS. Any code path that needs
both must go through `app.core.runtime.prepare_dl_runtime()` rather than importing `keras`/`torch`
directly; see [Troubleshooting](#10-troubleshooting) if you hit this.

## 2. First-time setup

```bash
git clone <repo> && cd CIL
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                       # makes `app` importable as a package
cd frontend && npm install && cd ..
```

That's the whole install. There is no separate configuration file to edit for a default local run —
every configurable value has a sensible default and is overridden by an environment variable when
needed (§5).

## 3. Building the artifacts: from raw data vs. using what's committed

**You may not need to build anything.** `artifacts/` (about 15 MB) and `data/processed/` are
committed to the repository, so a fresh clone already has a complete, working system — every
trained model, every metric, every figure and report the dashboard reads already exists on disk.
This is intentional: cloning the repo and running `./run.sh` should show real numbers immediately,
not force a 15-minute rebuild before the first page loads.

You need to build (or rebuild) only when:

- you've changed code in `app/intelligence/` and want to see the effect on the trained models or
  reports;
- you're retraining on a different race session (§8);
- you want to verify reproducibility end-to-end (a `--force` rebuild is the project's own
  correctness check — see `docs/TESTING_REPORT.md`).

```bash
python scripts/build_all.py            # builds what is missing, skips what already exists
python scripts/build_all.py --force    # rebuilds everything from the raw data
python scripts/build_all.py --skip-dl  # Tasks 1-6 only (no Keras/PyTorch needed)
python scripts/build_all.py --skip-qml # skip the quantum stage (no PennyLane needed)
```

Stages run in a fixed order and each one independently skips itself when its own artifacts already
exist — so a partial build (say, Tasks 1–6 already present, Task 7 not yet run) resumes correctly
rather than redoing completed work. **Tasks 7–9 depend on Task 6 and the build refuses to run them
against nothing** — if you see an error about a missing Task 6 registry, run Task 6 first (or just
run `build_all.py` without flags, which orders this correctly for you).

Approximate timings on a laptop CPU, from the project's own committed run (`DEPLOYMENT.md` §3):

| Stage | Time |
|---|---|
| Tasks 1–4 (knowledge, rules, search, cleaning) | seconds |
| Task 5 feature engineering | ~40 s |
| Task 6 machine learning | ~25 s |
| Task 7 deep learning | ~12 min (the hyperparameter search dominates) |
| Task 8 explainable AI | ~2 min |
| Quantum ML | ~30 s |

A full `--force` rebuild is therefore about 15 minutes, nearly all of it Task 7's hyperparameter
search over the cross-validation folds.

### Single stages

When you only want one stage rebuilt (faster iteration on one task's code):

```bash
python scripts/build_knowledge_base.py     # Task 1
python scripts/run_expert_system.py        # Task 2
python scripts/run_search.py               # Task 3
python scripts/run_eda.py                  # Task 4
python scripts/build_features.py           # Task 5
python scripts/run_qml.py                  # Quantum ML
python -c "from app.intelligence.ml import pipeline; pipeline.train_all()"    # Task 6
python -c "from app.intelligence.dl import pipeline; pipeline.train_all()"    # Task 7
python -c "from app.intelligence.xai import pipeline; pipeline.run_all()"     # Task 8
```

## 4. Running it

### One command

```bash
./run.sh
```

This is the supported path for a laptop demo or local development. It does the following, in
order (`run.sh`'s own comment block):

1. Creates and activates `.venv` if it doesn't exist, installs Python dependencies on first run.
2. Resolves the backend and frontend ports — honouring `BACKEND_PORT`/`FRONTEND_PORT` if you set
   them explicitly, otherwise starting from 8000/3000 and walking forward if something else (that
   isn't this project's own leftover process) already holds the port, asking before killing
   anything it doesn't recognise as its own.
3. Sets `F1_ALLOWED_ORIGINS` to include whatever frontend origin it actually resolved, so CORS
   works out of the box even if the port moved.
4. Builds any missing artifact stages.
5. Starts the API, warms it with a few real predictions (proving the trained models actually
   respond before you look at the dashboard), starts the frontend, and opens your browser.

Useful flags and variables:

```bash
./run.sh --force-retrain               # rebuild everything first
./run.sh --force-ports                 # kill whatever holds :8000 / :3000 without asking
./run.sh --skip-qml                    # skip the quantum stage (needs pennylane; slowest cold-build stage)
BACKEND_PORT=8001 FRONTEND_PORT=3001 ./run.sh
```

`./run.sh --help` prints the same summary from the terminal at any time.

### Running backend and frontend separately

Useful when iterating on one side only, or when you want the backend's auto-reload:

```bash
uvicorn app.api.main:app --reload                       # backend, :8000
cd frontend && npm run dev                               # dashboard, :3000
```

The dashboard finds the API through `NEXT_PUBLIC_API_BASE_URL`, which defaults to
`http://localhost:8000`. Point it elsewhere with:

```bash
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8001 npm run dev
```

## 5. Environment variables

Every configurable value the system reads, confirmed against `run.sh` and `app/api/main.py`:

| Variable | Default | Read by | Effect |
|---|---|---|---|
| `BACKEND_PORT` | `8000` | `run.sh` | Which port `uvicorn` binds. Explicitly set values are honoured or the script stops (rather than silently moving); an unset default is free to move if taken. |
| `FRONTEND_PORT` | `3000` | `run.sh` | Which port the Next.js dev/start server binds. |
| `NEXT_PUBLIC_API_BASE_URL` | `http://localhost:8000` | `frontend/lib/api.ts` (baked in at Next.js build time) | Where the dashboard's fetch client sends requests. Must be rebuilt (`npm run build`) if changed for a production build, since Next.js inlines `NEXT_PUBLIC_*` variables at build time. |
| `F1_ALLOWED_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` | `app/api/main.py` (CORS middleware) | Comma-separated list of origins allowed to call the API cross-origin. `run.sh` appends the frontend origin it actually resolved to whatever you set, rather than replacing it. |

CORS is deliberately not a wildcard (`app/api/main.py`'s own comment explains why: "any website the
developer visits in the same browser can issue cross-origin requests to 127.0.0.1:8000 and read the
responses"). If you deploy the dashboard at a real domain, set `F1_ALLOWED_ORIGINS` to that domain
explicitly — omitting it will leave the API rejecting the deployed dashboard's requests.

## 6. Serving it to other people

### Backend

```bash
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

Multiple workers are safe specifically because the API is read-only at request time: trained models
are loaded once per process and cached (`app/services/model_cache.py`), and nothing writes to
`artifacts/` while serving. If you add a write path to the API in the future, revisit this
assumption before scaling workers.

**Set the allowed origins** to the real dashboard domain, not the default:

```bash
export F1_ALLOWED_ORIGINS="https://dashboard.example.com"
```

**What the API does and does not expose.** Only these artifact directories are mounted read-only
under `/artifacts/` (`app/api/main.py`, `PUBLIC_ARTIFACT_DIRS`):

```
figures/  reports/  data_engineering/  knowledge_representation/
expert_system/  search/  deep_learning/  xai/
```

`artifacts/models/` and `artifacts/metadata/` are **never** mounted (`PRIVATE_ARTIFACT_DIRS` in the
same file), so trained weights (`.joblib`, `.h5`, `.npy`) are not downloadable over HTTP — a request
for one returns 404 because no route exists for it, not because of a filtering rule that could be
misconfigured. If you add a new artifact subdirectory that should stay private, do not add it to
`PUBLIC_ARTIFACT_DIRS`.

### Frontend

```bash
cd frontend
NEXT_PUBLIC_API_BASE_URL=https://api.example.com npm run build
npm run start -- --port 3000
```

`npm run build` bakes the API URL into the built assets, so you must rebuild (not just restart)
whenever `NEXT_PUBLIC_API_BASE_URL` changes.

### Behind a reverse proxy

Serving the dashboard at `/` and the API at `/api` on one hostname removes the CORS question
entirely — the browser sees same-origin requests. Nginx sketch:

```nginx
location /api/        { proxy_pass http://127.0.0.1:8000; }
location /artifacts/  { proxy_pass http://127.0.0.1:8000; }
location /            { proxy_pass http://127.0.0.1:3000; }
```

## 7. Health checks

| Check | Command | Expected |
|---|---|---|
| API up | `curl localhost:8000/api/health` | `200` with a status body (`status`, `models_trained`, `model_count`, `xgboost_available`) |
| Models loaded | `curl localhost:8000/api/ml/models` | the model registry, not a 404 |
| Artifacts served | `curl -I localhost:8000/artifacts/figures/qml_circuit.png` | `200` |
| Weights **not** served | `curl -I localhost:8000/artifacts/models/qml/vqc_weights.npy` | `404` |
| Dashboard up | `curl -I localhost:3000` | `200` |
| Tests pass | `pytest -q` | all green — 284 tests as of the 2026-09-27 audit (`docs/TESTING_REPORT.md`) |

If you extend `GET /api/health` with a richer per-model breakdown (a `models: {ml, dl,
xai_available}` shape was under discussion at the time this guide was written but had not landed in
this worktree — see [`sdd.md`](sdd.md) §2.1), update this table's first row to match whatever shape
actually ships rather than assuming the extension is live.

## 8. Retraining on a different race

```bash
python scripts/fetch_real_session.py --year 2023 --event Monza --session R
python scripts/build_all.py --force
```

Step 1 fetches the new session via FastF1 (needs network access; cached locally afterward, so a
second run against the same session is fast) and writes a provenance marker
(`data/raw/.data_source.json`). Step 2 re-cleans Task 4 on the new raw data, rebuilds Task 5's
feature contract as a pipeline stage (it used to be a presence check that silently left Task 6
training on the *previous* race's features — this was fixed and is now a real, verified rebuild
stage, per `docs/architecture.md`'s "Rebuilding the feature contract" section), and retrains every
downstream model and report. The dashboard then shows the new race automatically, with its
provenance badge updated — nothing needs editing by hand, because every number flows from
`data/processed/data_source.json` through the API to the frontend rather than being assumed
anywhere.

**Caveat found in the committed real-data run** (`docs/architecture.md`, "Synthetic vs. real
data"): with 20 real drivers and 10 real teams, Task 5's feature-selection funnel tends to keep more
one-hot driver/team identity dummies than the synthetic demo data did (45 regression features vs. 6
for the synthetic case), which is a real overfitting risk on a session with only ~800 development
rows that the selection funnel does not itself guard against. Retraining on a different single race
will likely reproduce this pattern; it is not specific to Bahrain.

## 9. Docker deployment

`Dockerfile` (backend), `frontend/Dockerfile`, and `docker-compose.yml` live at the repository root.
Both images are code-only — `artifacts/` and `data/` are excluded via `.dockerignore` and bind-mounted
at runtime instead, since they're produced by `scripts/build_all.py` and meant to be regenerated, not
shipped as a stale image layer. `./run.sh` is unaffected and remains the primary local path; Docker is
additive.

```bash
docker compose build        # builds the backend and frontend images
docker compose up           # starts both; frontend waits on the backend's healthcheck
```

- Backend: `http://localhost:8000` by default, with `data/` and `artifacts/` bind-mounted read-write so
  `docker compose exec backend python scripts/build_all.py` can (re)generate them from inside the
  container exactly as it would on the host.
- Frontend: `http://localhost:3001` by default — **not** 3000, since this machine commonly has an
  unrelated dev server (`./run.sh` or a plain `npm run dev`) already holding that port, and
  `docker compose up` shouldn't have to fight it. `NEXT_PUBLIC_API_BASE_URL` is baked into the client
  bundle at `docker compose build` time (a Next.js constraint — `NEXT_PUBLIC_*` values are inlined at
  build, not read at runtime) via a Docker build arg in `docker-compose.yml`, mirroring how `run.sh`
  passes the same variable to `next dev`.
- Both host ports are overridable without editing the compose file:
  `BACKEND_HOST_PORT=8001 FRONTEND_HOST_PORT=3002 docker compose up` — the backend's CORS
  (`F1_ALLOWED_ORIGINS`) and the frontend's build-time `NEXT_PUBLIC_API_BASE_URL` both pick up the
  override automatically.
- **Verified in this repo** (2026-10-01): `docker compose build` succeeds for both images;
  `docker compose up -d` brings up `backend` (healthy via its own `GET /api/health`, serving the full
  per-model breakdown) and `frontend` (HTTP 200, server-rendering real data fetched from the backend
  container — confirmed by curling the rendered page for the real dataset-source badge).
- The backend's healthcheck hits `GET /api/health` with a 90-second `start_period`, since the process
  imports the full ML/DL/XAI/QML stack (torch, keras, shap, lime, pennylane) at startup.
- `docker compose exec backend python scripts/record_environment.py` writes
  `artifacts/deployment/environment.txt` from inside the container, if you want a record of exactly
  what the containerized build installed.
- `docker compose down` stops both services. Neither image's `artifacts`/`data` volumes are removed
  by `down` alone (they're host bind mounts, not named Docker volumes) — your trained models and data
  stay on disk between runs.

If `artifacts/`/`data/` are empty on first `docker compose up` (a fresh clone with no prior
`./run.sh` or `build_all.py` run), the API will report models as untrained via `GET /api/health`
rather than fabricating predictions — run the build step first, on the host or inside the container.

## 10. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Task 5 outputs missing` | Run `python scripts/build_features.py`, or `python scripts/build_all.py`. |
| `No trained model registered` | Task 6 has not run: `python scripts/build_all.py`. |
| Dashboard shows "Backend unreachable" | API isn't running, or `NEXT_PUBLIC_API_BASE_URL` points at the wrong port — check with `curl localhost:8000/api/health`. |
| A page says "Not generated yet" | That task's artifacts are missing on disk. The page itself names the exact command to run — it is read live from `artifacts/`, not a stale message. |
| `XGBoost unavailable — skipped` | `libomp` is not installed. This is by design, not an error: nine models train instead of ten, and the report says so. |
| Segfault when importing Keras with XGBoost | Import order. Always go through `app.core.runtime.prepare_dl_runtime()` rather than importing `keras`/`torch` directly — see §1's note on the deep-learning stack. |
| `TensorFlow not found` | Expected. Keras runs on the PyTorch backend here; TensorFlow has no wheel for Python 3.14. |
| Port already in use | `./run.sh --force-ports`, or set `BACKEND_PORT`/`FRONTEND_PORT` to move out of the way instead. Under Docker, set `BACKEND_HOST_PORT`/`FRONTEND_HOST_PORT` (§9) — the frontend already defaults to 3001, not 3000, for exactly this reason. |
| CORS error in the browser console | The dashboard's actual origin isn't in `F1_ALLOWED_ORIGINS`. If you're running `./run.sh` with a non-default `FRONTEND_PORT`, it handles this for you; if you started the backend and frontend separately with custom ports, set `F1_ALLOWED_ORIGINS` yourself. |
| A downloaded strategy report is missing the SHAP/trust section | Explanations are opt-in on `/api/strategy/predict` (`explain: false` by default) but always forced on for `/api/strategy/report` — if a report is missing that section, check the live explainer didn't fail (`e.get("reason")` in the response) rather than assuming it was skipped. |

## 11. What to back up

| Path | Back up? | Why |
|---|---|---|
| `data/raw/`, `data/processed/` | yes | the session data and the Task 5 feature contract — losing these means re-fetching from FastF1 |
| `artifacts/` | yes | every trained model, metric, figure and report — losing these means a full ~15-minute rebuild |
| `.venv/`, `frontend/node_modules/`, `frontend/.next/` | no | reinstallable from `requirements.txt` / `package.json` |
| `fastf1_cache/` | no | re-downloads on demand from FastF1 |

`artifacts/` is about 15 MB and is already committed to the repository, so in practice a git clone
of this project already constitutes a backup of the current trained state.
