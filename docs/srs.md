# Software Requirements Specification

F1 Race Strategy Intelligence — a computational-intelligence platform built around one real
Formula 1 session, attacking two prediction problems (lap time, pit-stop decision) three ways:
symbolic reasoning, machine learning, and explainable AI, served behind a FastAPI backend and shown
on a six-page Next.js dashboard.

This document derives every requirement below from what is actually implemented — the API routers
under `app/api/routers/`, their Pydantic schemas in `app/api/schemas.py`, the six dashboard pages
under `frontend/app/`, and the project's own stated rules in `README.md`.

## 1. Purpose and scope

### 1.1 Purpose

To specify the functional and non-functional requirements of the F1 Race Strategy Intelligence
system: what it computes, what it exposes over HTTP, what a user sees on the dashboard, and the
constraints (data, reproducibility, leakage) the project holds itself to.

### 1.2 Scope

The system is a **decision-support tool**, not an autonomous strategy system. It:

- represents Formula 1 domain knowledge as an ontology and knowledge graph (Task 1);
- reasons over race state with a forward/backward-chaining rule base (Task 2);
- plans a pit strategy with state-space search (Task 3);
- cleans and analyses one real race session (Task 4);
- engineers and selects a feature set from that session (Task 5);
- trains and compares classical ML models for lap time and pit decision (Task 6);
- trains and compares deep neural networks on the same problems and the same holdout (Task 7);
- explains those models' predictions with SHAP, LIME, counterfactuals, a trust score and
  performance stratification (Task 8);
- integrates all of the above into one API and dashboard, including a downloadable strategy report
  (Task 9);
- documents itself for evaluation (Task 10, this document among others);
- as an extension beyond the lab specification, trains simulated quantum models on the same split
  for comparison (Quantum ML).

It is explicitly out of scope for the system to control a vehicle, communicate with live timing
systems, or generalize beyond the single race session it is trained on (see §4, Constraints).

## 2. Intended audience

- **Race engineers / dashboard users** — read [`user_manual.md`](user_manual.md) instead; this
  document is not written for that audience.
- **Developers extending the system** — the primary audience of this document, together with
  [`sdd.md`](sdd.md) and [`architecture.md`](architecture.md).
- **Evaluators / auditors** — verifying that what was built matches what was specified; cross-check
  against [`DELIVERABLES_CHECKLIST.md`](DELIVERABLES_CHECKLIST.md), which is the authoritative,
  audited list of what artifact exists at what path.

## 3. Functional requirements

Each requirement is grounded in a specific module, router or dashboard page. Numbering follows the
project's own ten-task structure.

### FR-1 — Knowledge representation (Task 1)

- **FR-1.1** The system shall represent the F1 domain as a set of typed entities, attributes and
  relationships, implemented in `app/intelligence/knowledge_representation/schema.py`.
- **FR-1.2** The system shall encode this schema as an OWL 2 ontology
  (`artifacts/knowledge_representation/ontology/formula1.owl`) and a populated instance knowledge
  graph (`.ttl` / `.graphml`).
- **FR-1.3** The system shall expose the entity, relationship and attribute counts, grouped by
  category, through `GET /api/reasoning/knowledge`.
- **FR-1.4** The system shall validate the knowledge base against a fixed set of static checks and
  record the result in `artifacts/knowledge_representation/reports/validation_report.md`.

### FR-2 — Rule-based expert system (Task 2)

- **FR-2.1** The system shall maintain a rule base of production rules with conditions, actions,
  category and salience, defined in `app/intelligence/expert_system/rule_base.py` and persisted at
  `artifacts/expert_system/rules/rule_base.json`.
- **FR-2.2** The system shall provide a forward-chaining inference engine
  (`app/intelligence/expert_system/inference.py`, class `InferenceEngine`) that accepts a race-state
  input dict and returns which rules fired, in what order, with which conditions matched, and what
  was concluded.
- **FR-2.3** The system shall expose the full rule base — id, name, category, salience, condition
  count and asserted actions — through `GET /api/reasoning/expert-system`.
- **FR-2.4** The system shall run the expert system live as part of every strategy prediction (see
  FR-9.2), not only as a static, pre-generated report.

### FR-3 — State-space search (Task 3)

- **FR-3.1** The system shall formulate the remaining-race pit strategy as a state-space search
  problem (`app/intelligence/search/problem.py`, class `RaceProblem`) with defined states, actions,
  transition costs and a goal test.
- **FR-3.2** The system shall implement and compare at least five search algorithms — BFS, DFS, UCS,
  Greedy Best-First and A\* — over the same problem instance, in `app/intelligence/search/algorithms.py`.
- **FR-3.3** The system shall run A\* live, from the caller's actual race state (current lap, tyre
  compound, track temperature, wet/dry), as part of every strategy prediction, returning the found
  plan, its total expected cost in seconds, and the next recommended action.
- **FR-3.4** The system shall expose the five-algorithm comparison (cost, nodes expanded, frontier
  size, wall-clock time) through `GET /api/reasoning/search`.

### FR-4 — Data preparation & EDA (Task 4)

- **FR-4.1** The system shall clean raw per-lap session data (dedupe, dtype coercion, time-format
  normalisation, categorical normalisation, missing-value imputation, IQR-based outlier capping)
  with every step recorded in an audit trail
  (`artifacts/data_engineering/reports/cleaning_audit.md`).
- **FR-4.2** The system shall record and expose the provenance of the session data — real vs.
  synthetic, year, event, session type — as a first-class fact
  (`data/processed/data_source.json`), never assumed, and surface it as `dataset_source` in every
  API response that depends on it.
- **FR-4.3** The system shall generate domain-specific exploratory analyses (driver, constructor,
  circuit, pit, tyre, lap time, weather, season, safety car) and a data-quality report, both served
  to the dashboard's Data page.

### FR-5 — Feature engineering & selection (Task 5)

- **FR-5.1** The system shall derive features from the cleaned lap data in six blocks (tyre, stint,
  pace, field, environment, interaction), implemented in `app/intelligence/features/build.py`.
- **FR-5.2** The system shall exclude same-lap and post-lap fields that would leak the prediction
  target — `Sector1Time`, `Sector2Time`, `Sector3Time`, `SpeedFL`, `SpeedST`, `IsPersonalBest` —
  from every feature set, and record this exclusion and its reason in
  `data/processed/feature_metadata.json` (`excluded_as_leakage`).
- **FR-5.3** The system shall run a four-stage feature-selection funnel — near-zero-variance,
  correlation, VIF, then importance-with-fold-stability — and record what each stage dropped.
- **FR-5.4** The system shall persist the selected feature set as a versioned contract
  (`data/processed/feature_metadata.json`), read by every downstream training and serving module
  through `app/intelligence/features/contract.py`, and shall never allow a duplicate, independently
  maintained feature list to exist elsewhere in the codebase.

### FR-6 — Machine learning (Task 6)

- **FR-6.1** The system shall train at least five regression models (for `target_laptime`) and at
  least four classification models (for `target_pit_next_lap`) from the Task 5 feature contract,
  including at minimum linear/logistic regression, a decision tree, a random forest and an
  SVM/SVR, with XGBoost included when its native dependency is available.
- **FR-6.2** The system shall validate every model with expanding-window, lap-forward
  cross-validation and a chronologically later, untouched holdout — never a random shuffle-split —
  implemented in `app/intelligence/ml/splits.py`.
- **FR-6.3** The system shall select the best model per task by a stated primary metric (MAE for
  regression, PR-AUC for classification) and record the selection rule, not just the winner, in
  `artifacts/reports/model_selection_report.md`.
- **FR-6.4** The system shall persist every trained pipeline (`.joblib`) together with its full
  reproducibility metadata (seed, dataset version, hyperparameters, feature list, metrics, software
  versions) in `artifacts/metadata/model_registry.json`.
- **FR-6.5** The system shall serve model listings, metrics, comparisons, feature importance and
  single-row predictions through `GET/POST /api/ml/*`, returning HTTP 404/503 with an actionable
  message — never a fabricated number — when no trained model is available.

### FR-7 — Deep learning (Task 7)

- **FR-7.1** The system shall train a Keras neural network for each of the two prediction targets,
  on the identical folds and holdout Task 6 uses (imported directly from `app.intelligence.ml.splits`
  and `.evaluation`, not reimplemented), so any comparison between the two model families is
  attributable to the model rather than the evaluation harness.
- **FR-7.2** The system shall select hyperparameters (architecture, learning rate, batch size,
  dropout, optimizer, L2, loss, class weighting) using only the cross-validation folds, leaving the
  held-out test laps untouched until final evaluation.
- **FR-7.3** The system shall apply overfitting-prevention measures (dropout, L2 regularisation,
  early stopping with best-weights restoration) and report a computed, evidence-based overfitting
  verdict per target rather than an asserted one.
- **FR-7.4** The system shall persist each trained network in HDF5 format with its scaler(s) and
  a `model_spec.json`, and verify that the reloaded model reproduces its trained predictions exactly
  (`reload_verified`) before treating training as complete.
- **FR-7.5** The system shall serve Task 7 model listings, metrics, training history, artifact
  listings and single-row predictions through `GET/POST /api/dl/*`, mirroring the Task 6 router's
  shape and using the tuned decision threshold (not a hard-coded 0.5) for the pit-decision class.
- **FR-7.6** The system shall never serve trained model weights (`.h5`, `.joblib`, `.npy`) over
  HTTP; only metrics, reports and figures derived from them are exposed.

### FR-8 — Explainable AI (Task 8)

- **FR-8.1** The system shall compute permutation feature importance, SHAP attributions and LIME
  local surrogates for the trained Task 7 network (and, as a second opinion, the Task 6
  selected-best model) on a fixed, rule-selected set of representative test laps plus the one lap
  where a real pit stop occurred.
- **FR-8.2** The system shall compute a tyre-age counterfactual scan for each explained lap, holding
  compound and derived interaction features physically consistent with the swept tyre age, and
  report whether any age in the observed training range flips the decision.
- **FR-8.3** The system shall compute a project-defined trust score per prediction, from a documented,
  weighted combination of confidence, cross-model agreement, explanation stability and input
  validity, and report the score's own empirical correlation with actual prediction error rather
  than asserting the score is reliable.
- **FR-8.4** The system shall compute F1-specific performance stratification (error by driver, team
  and tyre compound — the only groupings this single-circuit, no-demographic-data project has) and
  flag any group below a stated sample-size threshold as descriptive-only.
- **FR-8.5** The system shall serve every Task 8 result — importance, SHAP, LIME, counterfactual,
  trust score, fairness/stratification, and a plain-English narrative per prediction — through
  `GET /api/xai/*`, and shall compute a live, reduced-budget explanation for an arbitrary caller-
  supplied race state (not only the fixed committed set) through `app/intelligence/xai/live.py`.
- **FR-8.6** The system shall raise a explicit, typed error (`ExplainerUnavailableError`) rather
  than substitute a stand-in model whenever the model it is asked to explain is missing, so that
  every explanation served is provably an explanation of the deployed model.

### FR-9 — System integration (Task 9)

- **FR-9.1** The system shall validate every incoming race-state request (`RaceStateRequest`,
  `app/api/schemas.py`) both at the schema level (lap/tyre-age consistency, numeric bounds) and
  against the real dataset's known values (driver, team, tyre compound must be one of
  `GET /api/data/options`'s real, dataset-derived choices — never accepted as free text).
- **FR-9.2** The system shall combine, for a single submitted race state, the Task 6 ML prediction,
  the Task 2 expert-system inference and the Task 3 search plan into one response through
  `POST /api/strategy/predict`, with each component's output separately attributable in the
  response body (as implemented in `app/services/strategy_service.run_strategy_analysis`).
- **FR-9.3** The system shall, when a Task 6 model needs an engineered feature that a single
  race-state snapshot cannot supply (a feature requiring multi-lap history), fill it from a
  training-data median and report explicitly, per feature, that it did so
  (`approximated_features` in the response) — never silently substituting a value.
- **FR-9.4** The system shall flag, per feature, when a value derived from the caller's input falls
  outside the range the underlying model was trained on (`out_of_range`), with the training range
  included, so a caller can tell interpolation from extrapolation.
- **FR-9.5** The system shall generate a self-contained, downloadable strategy report
  (`POST /api/strategy/report`, Markdown by default or `?format=html` for a self-contained HTML
  page — `app/services/strategy_report.render_markdown` / `render_html`) containing the race state,
  the recommendation (including the ML/DL agreement or disagreement), triggered rules and their
  conclusions, the search plan, a Task 8 explanation of the Task 7 network's prediction on the same
  state, and a provenance table naming which module produced each section.
- **FR-9.6** The system shall present six dashboard pages — Overview, Strategy, Reasoning, Models,
  Explainability, Data — each of which renders exclusively from live API responses, and shall show
  "Not generated yet" / "Not available" with the exact command to remedy it, never a placeholder
  value, whenever a requested artifact does not exist.
- **FR-9.7** The system shall run a Task 7 DL prediction alongside the Task 6 ML prediction on every
  `/api/strategy/predict` call (`strategy_service._run_dl`, response field `dl_prediction`), using
  the tuned decision threshold Task 7's own endpoints use — never a hard-coded 0.5 — and shall
  combine the expert system, ML and DL evidence with the A* search plan into one `recommendation`
  object (`action` — `"PIT NOW"` / `"STAY OUT"` / `"PIT IN N LAPS"`, `confidence`, `source`,
  `disagreement`, `ml_pit_probability`, `dl_pit_probability`, `reason`) via
  `strategy_service._combine_recommendation`, stating explicitly when ML and DL disagree rather than
  silently preferring one.
- **FR-9.8** The system shall report which models are actually trained and loadable, per target and
  per model family, through `GET /api/health`'s `models: {ml, dl, xai_available}` breakdown, distinct
  from the task-level "was Task N ever built" ledger (`GET /api/tasks/evidence`).

### FR-10 — Evaluation & documentation (Task 10)

- **FR-10.1** The system shall maintain a traceability matrix (`DELIVERABLES_CHECKLIST.md`) mapping
  every lab-specified deliverable to the file that satisfies it, its build status, and — for
  anything not yet built — where the material for it already exists.
- **FR-10.2** The system shall maintain this SRS, a corresponding SDD, a user manual and a
  deployment guide (this document and its siblings in `docs/`), each traceable to the source code
  and generated artifacts they describe.
- **FR-10.3** The system shall maintain a testing report recording an actual, dated `pytest` run
  (not a paraphrase) with per-file test counts and the slowest test durations.

### Quantum ML extension (beyond the lab specification)

- **FR-EXT.1** The system shall train at least three quantum machine-learning models (a variational
  classifier, a variational regressor and a fidelity-kernel SVM) on a PennyLane noiseless simulator,
  using the same cross-validation folds and holdout as Task 6, with inputs PCA-reduced (fit on
  training rows only) to fit a small number of qubits.
- **FR-EXT.2** The system shall compare every quantum model against a parameter-matched classical
  baseline on the same reduced inputs, and shall not claim quantum advantage — the generated report
  and every dashboard surface state explicitly that this is a noiseless simulation of small circuits
  on one race.

## 4. Non-functional requirements

### 4.1 Performance

Approximate, machine-measured build timings on a laptop CPU, as recorded in
[`DEPLOYMENT.md`](DEPLOYMENT.md) §3:

| Stage | Time |
|---|---|
| Tasks 1–4 (knowledge, rules, search, cleaning) | seconds |
| Task 5 feature engineering | ~40 s |
| Task 6 machine learning | ~25 s |
| Task 7 deep learning | ~12 min (the hyperparameter search dominates) |
| Task 8 explainable AI | ~2 min |
| Quantum ML | ~30 s |

A full `--force` rebuild of every stage is therefore approximately 15 minutes. At serving time, the
API is read-only per request — trained models are loaded once and cached in-process
(`app/services/model_cache.py`), so a prediction request does not retrain or re-read a pipeline from
disk. `uvicorn` may be run with multiple workers safely for the same reason.

### 4.2 Correctness / no fabricated results

Quoted directly from the project's own stated rules (`README.md`, "The rules we held ourselves to"):

> - **No leakage.** Sector times sum to the lap time, so they are excluded along with speed traps
>   and `IsPersonalBest`. A test fails the build if any of them reaches a training matrix.
> - **No random splits.** This is a time-ordered panel, so a shuffled K-fold would train on lap 50
>   and validate on lap 10. Every model uses expanding-window, lap-forward folds with a
>   chronological holdout.
> - **Scalers are fit inside each fold**, never on the whole dataset.
> - **No made-up numbers.** Every value on the dashboard is read from a generated artifact. When
>   something cannot be computed, the UI and the reports say "undefined" and why.

These are enforced by the test suite, not only stated: leakage columns are asserted never to reach
a training matrix, no fold is asserted to train on a later lap than it validates on, a saved model
is asserted to reload and predict identically, and the committed trust scores are asserted to
recompute from their own recorded inputs (`docs/architecture.md`, "Testing").

### 4.3 Reproducibility

Every trained model records its random seed, dataset version, feature metadata, hyperparameters,
training timestamp, validation strategy, feature list, target, metrics and software versions
(`artifacts/metadata/model_registry.json`). Stochastic steps in feature selection and model
training are seeded (`random_state=42`) so a rebuild from the same raw data reproduces the committed
contract and metrics; the last full verification of this is recorded in
[`TESTING_REPORT.md`](TESTING_REPORT.md), which reports metrics reproducing to roughly `1e-14`
after a clean `--force` rebuild.

### 4.4 Testing

As of the audit dated 2026-09-27 (`docs/TESTING_REPORT.md`): **284 tests passed, 0 failed, 0
skipped**, in 57.67 s, across 16 test files, together with a clean `ruff check .` and a passing
frontend production build (`npm run build`, 6 routes). The test suite writes to a temporary
directory (`app.core.paths.ArtifactPaths` redirected root) so running it never modifies the
committed artifacts. (`docs/architecture.md`'s own "120 tests" figure and the top-level
`README.md`'s prose reference to "261 tests" predate this audit and are superseded by the dated,
`pytest --collect-only`-verified count in `TESTING_REPORT.md`.)

### 4.5 Security / data exposure

CORS is not a wildcard: allowed origins are read from `F1_ALLOWED_ORIGINS` (default
`http://localhost:3000,http://127.0.0.1:3000`), and only `GET`/`POST` with a `Content-Type` header
are permitted (`app/api/main.py`). Only specific artifact subdirectories — figures, reports,
data_engineering, knowledge_representation, expert_system, search, deep_learning, xai — are mounted
read-only under `/artifacts/`; `models/` and `metadata/` are never mounted, so trained weights
(`.joblib`, `.h5`, `.npy`) are not downloadable over HTTP by construction (a route that doesn't
exist cannot be bypassed by a filtering rule).

## 5. Constraints

- **Single-race data.** All Task 5–8 training, evaluation, explanation and dashboard numbers are
  derived from one session: the 2023 Bahrain Grand Prix (Race), 1,055 raw laps / 995 usable rows
  after the warm-up trim (`data/processed/feature_metadata.json`). No claim on the dashboard should
  be read as generalising beyond that race.
- **Python 3.12+** is required; the system was developed and its committed artifacts generated on
  Python 3.14.6 (`docs/DEPLOYMENT.md` §1).
- **Node.js 20+** is required for the frontend.
- **No database, no message queue, no cloud service.** The system reads and writes plain files under
  `data/` and `artifacts/` only (`docs/DEPLOYMENT.md` §1).
- **XGBoost is optional**, gated on the native `libomp` runtime; its absence is detected and reported
  ("XGBoost unavailable — skipped") rather than failing the build or fabricating a result.
- **TensorFlow has no wheel for Python 3.14**; Task 7 runs Keras 3 on the PyTorch backend instead
  (`app/core/runtime.py`), and an import-order guard exists because PyTorch and XGBoost's OpenMP
  runtimes can otherwise segfault on macOS.
- **Containerization is optional, additive to the process-based path.** `Dockerfile`,
  `frontend/Dockerfile` and `docker-compose.yml` exist at the repository root; `./run.sh` remains the
  primary local workflow and is unaffected — see [`sdd.md`](sdd.md) §6 and
  [`deployment_guide.md`](deployment_guide.md)'s Docker section.

## 6. External interfaces

The system's only external interface is a REST API served by FastAPI (`app/api/main.py`), consumed
by the Next.js dashboard through a typed client (`frontend/lib/api.ts`) and by
`scripts/demo_predict.py`. Every endpoint below is read directly from the router source under
`app/api/routers/`.

| Router | Method & path | Purpose |
|---|---|---|
| health | `GET /api/health` | Liveness/status: model registry presence, model count, XGBoost availability, and a per-target/per-family `models: {ml, dl, xai_available}` breakdown |
| ml | `GET /api/ml/models` | Task 6 model registry (classical models only) |
| ml | `GET /api/ml/metrics` | Regression + classification metrics |
| ml | `GET /api/ml/comparison` | All Task 6 models ranked on the primary CV metric |
| ml | `GET /api/ml/artifacts` | The generated-artifact manifest |
| ml | `GET /api/ml/feature-importance` | Native feature importance of each task's selected-best model |
| ml | `GET /api/ml/top-features` | Top-N features for a target, human-readable, ranking method stated |
| ml | `POST /api/ml/predict/laptime` | Lap-time prediction from a full Task 5 feature row |
| ml | `POST /api/ml/predict/pit` | Pit-probability prediction from a full Task 5 feature row |
| strategy | `POST /api/strategy/predict` | Combined ML + DL + expert-system + search recommendation from a race state, eight pipeline stages in one response |
| strategy | `POST /api/strategy/report` | Downloadable strategy briefing, `?format=markdown` (default) or `?format=html` (always includes explanations) |
| data | `GET /api/data/options` | Real driver/team/compound dropdown values and the trained track-temperature range |
| dl | `GET /api/dl/models` | Task 7 entries in the shared model registry |
| dl | `GET /api/dl/metrics` | Task 7 evaluation report |
| dl | `GET /api/dl/comparison` | Task 7 vs. Task 6 on the identical holdout |
| dl | `GET /api/dl/history` | Per-epoch training curves |
| dl | `GET /api/dl/artifacts` | Which Task 7 deliverables exist on disk |
| dl | `POST /api/dl/predict/laptime` | Task 7 network lap-time prediction |
| dl | `POST /api/dl/predict/pit` | Task 7 network pit prediction, at the tuned threshold |
| xai | `GET /api/xai/summary` | Headline Task 8 status per target |
| xai | `GET /api/xai/feature-importance` | Permutation importance, DNN vs. classical |
| xai | `GET /api/xai/shap` | Global SHAP ranking and per-row attributions |
| xai | `GET /api/xai/lime` | LIME local surrogates and their local R² |
| xai | `GET /api/xai/counterfactual` | Tyre-age counterfactual scan and DiCE alternatives |
| xai | `GET /api/xai/trust-score` | Trust score, components, formula and weights |
| xai | `GET /api/xai/fairness` | Identity-attribution share vs. an even spread |
| xai | `GET /api/xai/explanation` | The plain-English, per-prediction race-engineer view |
| xai | `GET /api/xai/stratification` | Error by driver, team and tyre compound |
| xai | `GET /api/xai/laps` | Every scored test lap, for the lap inspector |
| xai | `GET /api/xai/lap` | One lap's full explanation, with a live-computed counterfactual |
| qml | `GET /api/qml/summary` | Quantum vs. parameter-matched classical results |
| qml | `GET /api/qml/training` | Per-epoch loss for the variational quantum models |
| qml | `GET /api/qml/search` | Every quantum hyperparameter trial and its CV score |
| reasoning | `GET /api/reasoning/knowledge` | Task 1 ontology/graph counts |
| reasoning | `GET /api/reasoning/expert-system` | Task 2 rule base |
| reasoning | `GET /api/reasoning/search` | Task 3 five-algorithm comparison |
| tasks | `GET /api/tasks/evidence` | Live scan of every task's real generated artifacts |
| charts | `GET /api/charts` | Index of available chart-data JSON files |
| charts | `GET /api/charts/{name}` | One chart's underlying numbers (browser-drawn, not a served PNG) |

Static files: `/artifacts/{figures,reports,data_engineering,knowledge_representation,
expert_system,search,deep_learning,xai}/**` are mounted read-only. `artifacts/models/` and
`artifacts/metadata/` are never mounted.

Every endpoint above returns HTTP 404 (missing artifact) or 503 (no trained model) with an
actionable detail message — never a 200 with a fabricated body — when the underlying pipeline stage
has not been run.

## 7. Assumptions and dependencies

- The committed `artifacts/` and `data/processed/` directories (approximately 15 MB, committed to
  the repository) already contain a complete build, so a fresh clone has a working system before
  anything is rebuilt.
- The pinned library versions in `requirements.txt` (scikit-learn 1.9.0, xgboost 3.4.1, keras
  3.15.1, torch 2.14.0, shap 0.52.0, lime 0.2.0.1) are assumed to reproduce the committed metrics; an
  unpinned upgrade is not guaranteed to.
- `scripts/fetch_real_session.py` assumes network access to FastF1's data source when retraining on
  a different race; there is no offline cache shipped for a session other than the committed one.
