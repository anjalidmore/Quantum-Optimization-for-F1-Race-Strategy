# Deployment guide

How to build, run and serve this system. Two commands cover the normal case:

```bash
python scripts/build_all.py     # build every artifact (Tasks 1-9)
./run.sh                        # start the backend and the dashboard
```

Everything else in this document is detail for a different machine, a different
race, or a server rather than a laptop.

---

## 1. What has to be installed

| Requirement | Version | Why |
|---|---|---|
| Python | 3.12+ (developed on 3.14.6) | the whole backend |
| Node.js | 20+ | the Next.js dashboard |
| `libomp` | any | XGBoost only; the build skips XGBoost with a stated reason if it is missing |
| pandoc | optional | converts a generated strategy report to PDF |

No database, no message queue, no cloud service. The system reads and writes
plain files under `data/` and `artifacts/`.

## 2. First-time setup

```bash
git clone <repo> && cd CIL
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                       # makes `app` importable
cd frontend && npm install && cd ..
```

The libraries that produce committed metrics (keras, torch, shap, lime,
scikit-learn, xgboost, pennylane) are pinned to exact versions in
`requirements.txt`, so a fresh install reproduces the numbers in `artifacts/`
rather than drifting with a new release.

## 3. Building the artifacts

```bash
python scripts/build_all.py            # builds what is missing, skips what exists
python scripts/build_all.py --force    # rebuilds everything from the raw data
python scripts/build_all.py --skip-dl  # Tasks 1-6 only (no Keras needed)
python scripts/build_all.py --skip-qml # skip the quantum stage (no PennyLane needed)
```

Stages run in order, and each one skips itself when its artifacts already exist.
Tasks 7-9 depend on Task 6, and the build refuses to run them against nothing.

Approximate timings on a laptop CPU, from the committed run:

| Stage | Time |
|---|---|
| Tasks 1-4 (knowledge, rules, search, cleaning) | seconds |
| Task 5 feature engineering | ~40 s |
| Task 6 machine learning | ~25 s |
| Task 7 deep learning | ~12 min (the hyperparameter search dominates) |
| Task 8 explainable AI | ~2 min |
| Quantum ML | ~30 s |

A full `--force` rebuild is therefore about 15 minutes, nearly all of it Task 7.

Single stages, when you only want one:

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

It creates the virtual environment if needed, installs dependencies on first
run, builds the artifacts if none exist, starts the API on `:8000` and the
dashboard on `:3000`, prints a few live predictions to prove the models work,
and opens the browser.

Useful flags and variables:

```bash
./run.sh --force-retrain               # rebuild everything first
./run.sh --force-ports                 # kill whatever holds :8000 / :3000
BACKEND_PORT=8001 FRONTEND_PORT=3001 ./run.sh
```

### Separately

```bash
uvicorn app.api.main:app --reload                       # backend, :8000
cd frontend && npm run dev                              # dashboard, :3000
```

The dashboard finds the API through `NEXT_PUBLIC_API_BASE_URL`, default
`http://localhost:8000`. Point it elsewhere with:

```bash
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8001 npm run dev
```

## 5. Serving it to other people

### Backend

```bash
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

Workers are safe: the API is read-only at request time. It loads trained models
once per process and caches them (`app/services/model_cache.py`); nothing writes
to `artifacts/` while serving.

**Set the allowed origins.** CORS is not a wildcard — it reads a list, so the
dashboard's real origin has to be included:

```bash
export F1_ALLOWED_ORIGINS="https://dashboard.example.com"
```

**What the API does and does not expose.** Only these artifact directories are
mounted read-only under `/artifacts/`:

```
figures/  reports/  data_engineering/  knowledge_representation/
expert_system/  search/  deep_learning/  xai/
```

`artifacts/models/` and `artifacts/metadata/` are **not** mounted, so trained
weights (`.joblib`, `.h5`, `.npy`) are never downloadable over HTTP. A request
for one returns 404. Keep it that way if you add directories.

### Frontend

```bash
cd frontend
NEXT_PUBLIC_API_BASE_URL=https://api.example.com npm run build
npm run start -- --port 3000
```

`npm run build` bakes the API URL in at build time, so rebuild when it changes.

### Behind a reverse proxy

Serve the dashboard at `/` and the API at `/api` on one hostname, which removes
the CORS question entirely. Nginx sketch:

```nginx
location /api/        { proxy_pass http://127.0.0.1:8000; }
location /artifacts/  { proxy_pass http://127.0.0.1:8000; }
location /            { proxy_pass http://127.0.0.1:3000; }
```

## 6. Training on a different race

```bash
python scripts/fetch_real_session.py --year 2023 --event Monza --session R
python scripts/build_all.py --force
```

Step 1 fetches the session and writes a provenance marker; step 2 re-cleans,
rebuilds the features, retrains every model and regenerates every report. The
dashboard then shows the new race, and the provenance badge shows which session
it is. Nothing needs editing by hand.

## 7. Health checks

| Check | Command | Expected |
|---|---|---|
| API up | `curl localhost:8000/api/health` | `200` with a status body |
| Models loaded | `curl localhost:8000/api/ml/models` | the model registry |
| Artifacts served | `curl -I localhost:8000/artifacts/figures/qml_circuit.png` | `200` |
| Weights **not** served | `curl -I localhost:8000/artifacts/models/qml/vqc_weights.npy` | `404` |
| Dashboard up | `curl -I localhost:3000` | `200` |
| Tests pass | `pytest -q` | all green |

## 8. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Task 5 outputs missing` | Run `python scripts/build_features.py`, or `build_all.py`. |
| `No trained model registered` | Task 6 has not run: `python scripts/build_all.py`. |
| Dashboard shows "Backend unreachable" | API not running, or `NEXT_PUBLIC_API_BASE_URL` points at the wrong port. |
| A page says "Not generated yet" | That task's artifacts are missing. The page names the command to run. |
| `XGBoost unavailable — skipped` | `libomp` is not installed. By design, not an error: nine models train instead of ten. |
| Segfault when importing Keras with XGBoost | Import order. Always go through `app.core.runtime.prepare_dl_runtime()`. |
| `TensorFlow not found` | Expected. Keras runs on the PyTorch backend here; TensorFlow has no wheel for Python 3.14. |
| Port already in use | `./run.sh --force-ports`, or set `BACKEND_PORT` / `FRONTEND_PORT`. |

## 9. What to back up

| Path | Back up? | Why |
|---|---|---|
| `data/raw/`, `data/processed/` | yes | the session data and the Task 5 feature contract |
| `artifacts/` | yes | every trained model, metric, figure and report |
| `.venv/`, `frontend/node_modules/`, `frontend/.next/` | no | reinstallable |
| `fastf1_cache/` | no | re-downloads on demand |

`artifacts/` is about 15 MB and is committed to the repository, so a clone
already has a working system before anything is built.
