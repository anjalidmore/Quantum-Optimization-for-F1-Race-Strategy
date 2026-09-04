# 🏎️ Quantum Optimization for Formula 1 Race Strategy

> A unified Computational Intelligence system for **Formula 1 race-strategy decision support**, combining knowledge representation, rule-based reasoning, state-space search, data engineering, feature engineering, classical machine learning, deep learning, explainable AI, and quantum optimization.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Machine Learning](https://img.shields.io/badge/ML-Classical%20%2B%20Deep%20Learning-orange)
![Quantum](https://img.shields.io/badge/Quantum-QML%20%2F%20Optimization-purple)
![Status](https://img.shields.io/badge/Status-Tasks%201--6%20Implemented-brightgreen)

> **Dataset note:** the pipeline runs on either a reproducible **synthetic** session or a **real FastF1**
> session, and always says which. The results currently committed in this repository are from a **real
> session — the 2023 Bahrain Grand Prix (Race)** — fetched via `scripts/fetch_real_session.py`. They
> reflect one real race, not a validated general-purpose F1 model — see [§13](#13-synthetic-vs-real-data).

---

## 1. Project Overview

Formula 1 race strategy is a sequential decision problem influenced by:

- tyre compound and tyre age
- degradation
- fuel load and race progress
- driver pace
- team/car performance
- field pace
- weather and track conditions
- pit-stop loss
- safety-car / track status
- remaining race distance
- tactical constraints

This project builds a **single end-to-end computational intelligence platform** that can analyse these factors and produce reproducible race-strategy recommendations.

The project is intentionally designed as **one integrated system**, rather than a collection of unrelated task folders. The individual computational-intelligence tasks remain identifiable through their modules, services, algorithms, reports, and UI sections, but they all operate on shared domain models, shared data, and shared generated artifacts.

### Core pipeline

```text
Raw F1 / FastF1 Data
        │
        ▼
Data Validation + Cleaning + EDA
        │
        ▼
Feature Engineering + Feature Selection
        │
        ├──────────────► Knowledge Representation
        │                       │
        │                       ▼
        │                Knowledge Graph / Ontology
        │                       │
        ├──────────────► Expert Rules ──────► Explainable Rule Decisions
        │
        ├──────────────► State-Space Search ─► Classical Optimal Baseline
        │
        └──────────────► ML / DL / XAI / QML
                                │
                                ▼
                     Unified Strategy Engine
                                │
                                ▼
                     Web Dashboard / Reports
```

---

## 2. Computational Intelligence Tasks

The implementation follows the 10-task journey represented in the project specification.

| Task | Capability | Role in the unified system |
|---|---|---|
| **1** | Knowledge Representation | Formal F1 entities, attributes, relationships, ontology and knowledge graph |
| **2** | Rule-Based Expert System | Symbolic strategy reasoning using production rules |
| **3** | State-Space Search | BFS, DFS, UCS, Greedy and A* strategy optimisation |
| **4** | Data Preparation & EDA | Cleaning, validation, statistics and domain analytics |
| **5** | Feature Engineering & Selection | Causal, model-ready feature matrix and target definitions |
| **6** | Machine Learning Model Development | Train, compare, evaluate and persist classical ML models |
| **7** | Deep Learning Model Development | Neural models for race-performance prediction |
| **8** | Explainable AI | SHAP/LIME/counterfactual explanations and model trust information |
| **9** | System Integration & Deployment | Unified strategy engine and application |
| **10** | System Evaluation & Responsible AI | Performance, usability, explainability, robustness and documentation |

**Important:** these are conceptual task boundaries, not mandatory directory boundaries.

---

# 3. Current Implementation

The supplied implementation already establishes a strong classical foundation.

### Knowledge Representation

The knowledge model contains F1 entities and relationships covering historical championship data and FastF1 concepts such as:

- seasons
- races
- circuits
- drivers
- constructors
- results
- standings
- pit stops
- lap times
- qualifying
- sessions
- stints
- tyre compounds
- weather
- track status
- race-control information
- car data
- telemetry

The ontology is generated from a declarative schema and the knowledge graph is produced using NetworkX/RDF tooling.

### Rule-Based Expert System

The expert system contains a validated production-rule architecture with:

- working memory
- typed conditions/actions/rules
- forward chaining
- backward chaining
- conflict resolution
- rule validation
- provenance
- HOW/WHY explanations

Rules are organised around strategic areas including:

`weather`, `safety_car`, `pit`, `degradation`, `tyre`, `strategy`, `fuel`, `energy`, `tactics`, and `risk`.

### State-Space Search

Race strategy is represented as a shortest-path problem.

A state contains:

```text
(lap, compound, tyre_age, stops_made, fuel_kg, compounds_used)
```

Actions include:

```text
RUN
PIT(compound)
```

The system compares:

- BFS
- DFS
- Uniform-Cost Search
- Greedy Best-First Search
- A*

The current formulation uses UCS as the ground-truth optimal baseline and verifies that A* reaches the same optimal cost when the admissible heuristic is used.

### Data Engineering & EDA

The data pipeline follows:

```text
Load → Clean → Transform → Analyse → Visualise → Report
```

It supports the Ergast-derived/Kaggle F1 schema and FastF1 lap/session data.

The pipeline handles:

- schema validation
- `\N` null conversion
- duplicate removal
- datatype coercion
- lap-time parsing
- categorical normalisation
- missing-value imputation
- outlier detection/capping
- encoding
- scaling
- statistical summaries
- correlation analysis
- data-quality scoring
- driver analysis
- constructor analysis
- circuit analysis
- pit-stop analysis
- tyre analysis
- lap-time analysis
- weather analysis
- season analysis
- safety-car / track-status analysis
- visual dashboards

### Feature Engineering & Feature Selection

Task 5 produces the model-ready feature matrix from the cleaned FastF1 lap dataset.

The current exported feature matrix contains **520 rows and 19 columns**, including identifiers, selected predictors, and targets.

Primary regression features:

```text
tyre_life
gap_roll3_mean
gap_expanding
field_median_lag1
driver_sai
race_progress
```

Pit-decision features:

```text
tyre_life
race_progress
form_vs_baseline
field_median_lag1
team_aston_martin
field_pace_trend
tyrelife_x_medium
tracktemp_dev_x_tyrelife
```

Targets:

```text
target_laptime
target_pit_next_lap
target_laptime_fuel_corrected
```

The feature-selection metadata also records:

- feature provenance
- scaling requirements
- validation strategy
- leakage exclusions
- selection funnel
- stability information

### Critical modelling contract

Task 5 deliberately exports **unscaled** features.

Task 6 and all downstream modelling tasks must fit preprocessing/scalers **inside the training folds**.

The dataset is a time-ordered driver × lap panel, so random K-fold cross-validation must **not** be used.

The intended validation strategy is:

```text
Expanding-window / lap-forward validation
```

with whole laps kept together.

---

# 4. Machine Learning — Task 6

Task 6 converts the selected feature matrix into trained classical ML models.

The system should support two complementary modelling problems.

### A. Lap-time regression

Predict:

```text
target_laptime
```

This answers:

> "Given everything known before the lap begins, what lap time should we expect?"

Candidate models:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Support Vector Regression
- XGBoost Regressor when available

Primary metrics:

- MAE
- RMSE
- R²
- optionally MAPE where numerically appropriate

### B. Pit-decision classification

Predict:

```text
target_pit_next_lap
```

This answers:

> "Given the current race state, should the driver pit at the end of this lap?"

Candidate models:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- SVM
- XGBoost Classifier when available

Primary metrics:

- ROC-AUC
- PR-AUC
- accuracy
- precision
- recall
- F1
- confusion matrix
- calibration where appropriate

Because pit events are relatively rare, accuracy must never be the only metric.

The current synthetic dataset has a near-deterministic stint schedule, so its pit-decision AUC can be artificially high. Results must clearly disclose this limitation and should not be presented as real-world performance.

### Task 6 status: implemented

All five regression models, four classification models (plus XGBoost when available), chronological
holdout, expanding-window CV, bounded hyperparameter search, leakage checks, persisted pipelines, figures,
reports and the model registry are implemented in `app/intelligence/ml/` and exercised by
`tests/test_ml_*.py`. Run `python scripts/build_all.py --force` to reproduce the numbers below from scratch.

Latest run in this repository — **real FastF1 data** (2023 Bahrain GP, Race; see [§13](#13-synthetic-vs-real-data)),
10 models trained, 0 fabricated:

| | Best model | CV metric | Test metric |
|---|---|---|---|
| Lap-time regression | `decision_tree` | MAE 1.19s | MAE 0.87s, R² −0.17 |
| Pit-decision classification | `random_forest` | ROC-AUC 0.85 | ROC-AUC 0.98 |

The synthetic demo path still works identically (`linear_regression` / MAE ~0.27s CV, `random_forest` /
ROC-AUC ~1.0 CV but undefined on test — see [§13](#13-synthetic-vs-real-data) for why real data behaves
so differently, notably that the regression test R² being negative here is an honest result, not a bug: a
single real race, a small feature-selected model, and a fuel/tyre state very different from the training
laps in the final stint is a genuinely hard extrapolation.

Full tables, per-fold metrics, and the full discussion of both datasets' caveats live in
`artifacts/reports/*.md` and are served live by `GET /api/ml/comparison` / the Machine Learning dashboard.

---

# 5. Leakage Prevention

This project treats leakage prevention as a first-class engineering requirement.

The following same-lap/post-lap fields were explicitly excluded by Task 5:

```text
Sector1Time
Sector2Time
Sector3Time
SpeedFL
SpeedST
IsPersonalBest
```

They must not re-enter Task 6 through an alternate preprocessing path.

The following principles must remain true:

1. Only information available before the prediction point may be used.
2. Historical features must be causal.
3. Scalers and preprocessing objects must be fitted only on training data.
4. Validation must respect race/lap chronology.
5. Hyperparameter selection must not use the final test set.
6. Test data must remain untouched until final evaluation.
7. All transformations used by a trained model must be serialised with the model/pipeline.

---

# 6. Unified System Architecture

This is the actual, as-built repository layout — the old `phase1_taskN/` / `phase2_taskN/` folders have been
removed; every module now lives under one shared `app/` package:

```text
f1-quantum-strategy/
│
├── README.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
│
├── app/
│   ├── api/
│   │   ├── main.py                 # FastAPI app, CORS, static /artifacts mount
│   │   ├── schemas.py              # Pydantic request/response models
│   │   └── routers/
│   │       ├── health.py           # GET  /api/health
│   │       ├── ml.py               # GET  /api/ml/{models,metrics,comparison,artifacts,feature-importance}
│   │       │                       # POST /api/ml/predict/{laptime,pit}
│   │       ├── strategy.py         # POST /api/strategy/predict
│   │       ├── data.py             # GET  /api/data/* — dataset options for the simulator
│   │       └── tasks.py            # GET  /api/tasks/evidence — per-task artifact index (Tasks 1-9)
│   │
│   ├── core/
│   │   └── paths.py                # single source of truth for every data/artifact path
│   │
│   ├── services/
│   │   ├── model_cache.py          # loads + caches trained pipelines (no retraining per request)
│   │   ├── feature_approximation.py# builds ML feature rows from a strategy-simulator snapshot
│   │   └── strategy_service.py     # wires ML + Expert System + Search into one recommendation
│   │
│   └── intelligence/
│       ├── knowledge_representation/  # Task 1 (formerly f1kr)
│       ├── expert_system/             # Task 2 (formerly f1es)
│       ├── search/                    # Task 3 (formerly f1search)
│       ├── data/                      # Task 4 (formerly f1data)
│       ├── features/                  # Task 5 contract reader (contract.py)
│       └── ml/                        # Task 6 — data_contract, splits, preprocessing,
│                                       # regression, classification, tuning, evaluation,
│                                       # selection, persistence, registry, visualize,
│                                       # reports, pipeline (orchestrator)
│
├── frontend/                        # Next.js 14 + TypeScript + Tailwind
│   ├── app/                         # 5 routes, all wired into the nav
│   │   ├── page.tsx                 # Dashboard / Overview
│   │   ├── strategy/page.tsx        # Race Strategy Simulator
│   │   ├── machine-learning/page.tsx# Model metrics, comparison, live prediction
│   │   ├── data-analysis/page.tsx   # Task 4 figures + reports, served from artifacts/
│   │   └── evidence/page.tsx        # Per-task evidence index (Tasks 1-9)
│   ├── components/                  # Nav, ArtifactImage, PredictPanel, DatasetBadge,
│   │   └── strategy/                # TaskEvidence, Modal, Tooltip, strategy tabs
│   └── lib/api.ts                   # typed fetch client (no hard-coded data)
│
├── data/
│   ├── raw/                         # Kaggle/Ergast-style + FastF1-like CSVs
│   └── processed/                   # fastf1_laps_clean.csv, f1_features_selected.csv,
│                                     # feature_metadata.json — the Task 5 contract
│
├── artifacts/                       # everything scripts/build_all.py generates
│   ├── knowledge_representation/    # Task 1 reports/diagrams
│   ├── expert_system/               # Task 2 reports/rules
│   ├── search/                      # Task 3 reports/figures
│   ├── data_engineering/            # Task 4 reports/figures
│   ├── models/{laptime,pit_decision}/  # Task 6 persisted pipelines (.joblib)
│   ├── metrics/                     # Task 6 metrics JSON
│   ├── figures/                     # Task 6 PNGs
│   ├── reports/                     # Task 6 markdown reports
│   ├── metadata/model_registry.json
│   └── manifest.json                # artifact-driven frontend manifest
│
├── docs/
│   ├── notebooks/task5_feature_engineering.ipynb
│   └── task{1,2,3,4}_*.md           # per-task documentation (traceability)
│
├── tests/                           # one flat suite — data contract, splits, leakage,
│                                     # training, persistence, API, plus the original
│                                     # Task 1-4 suites, all passing against the new layout
│
├── scripts/
│   ├── build_all.py                 # top-level build: Tasks 1-6 end to end
│   ├── build_knowledge_base.py      # Task 1
│   ├── run_expert_system.py         # Task 2
│   ├── run_search.py                # Task 3
│   ├── run_eda.py                   # Task 4
│   ├── fetch_real_session.py        # replaces synthetic laps with a real FastF1 session
│   └── demo_predict.py              # used by run.sh to prove the trained models respond
│
└── run.sh                           # one-command setup + demo (see §11)
```

> **One application, one domain, one shared data/artifact layer, multiple computational-intelligence engines.**

---

# 7. Frontend

The frontend is not just a landing page — it renders real artifacts and calls the real backend. Built with
Next.js 14 (App Router) + TypeScript + Tailwind, in `frontend/`.

**Implemented today:** Overview, Machine Learning (Task 6 dashboard), Race Strategy Simulator.
**Not yet built** (shown as disabled nav items, not fake pages): Data & EDA, Knowledge, Expert System, Search
Optimisation, Deep Learning, Explainability — their backend modules exist (Tasks 1-4), but no dedicated UI
page reads them yet. Ready for Tasks 7-10 to slot into the same nav/API pattern.

### Overview (implemented)

- current dataset status
- number of races/laps/drivers
- model status
- latest model metrics
- strategy-engine status
- pipeline health
- recent generated artifacts

### Machine Learning (implemented)

Header cards (dataset source, model count, best regression/classification model, last training timestamp),
a lap-time regression section (best-model cards, model-comparison / predicted-vs-actual / residual /
feature-importance figures, full comparison table, a live "try a prediction" form), and a mirrored
pit-decision classification section — all reading `GET /api/ml/{comparison,artifacts,models,feature-importance}`
and rendering `artifacts/figures/*.png` served by the backend. The synthetic-data caveat is shown as a
banner, not buried in a footnote.

### Race Strategy Simulator (implemented)

A form for driver, team, current/total laps, tyre compound, tyre age, track temperature, weather, fuel
state, track status and current position. **RUN STRATEGY ANALYSIS** calls `POST /api/strategy/predict`,
which runs the real pipeline:

```text
Race State
    ↓
Feature Construction   (app/services/feature_approximation.py — see the honesty note below)
    ↓
ML Prediction           (Task 6 cached pipelines)
    ↓
Expert Rules            (Task 2 forward-chaining inference engine)
    ↓
Search / Optimisation   (Task 3 A* over the remaining stint)
    ↓
Unified Recommendation
```

and displays predicted lap time, probability of pit, recommended action, expected cost, the A* plan,
triggered expert rules, and the expert system's evidence — nothing pre-scripted.

**Honesty note:** Task 6's ML models require Task 5's engineered features (rolling gap, field-median lag,
form-vs-baseline, ...), which need multi-lap race history a single form snapshot can't supply. Two
interaction features (`tyrelife_x_medium`, `tracktemp_dev_x_tyrelife`) and tyre/team/driver features are
computed exactly from the form input and real Task 4/5 statistics; the remaining history-dependent features
fall back to the training data's median value. The response's `approximated_features` field names exactly
which ones, every time — this is a stated engineering approximation, not a fabricated prediction.

### Also implemented

**Data & Analysis** (`/data-analysis`) renders Task 4's figures and reports directly from `artifacts/`,
and **Project Evidence** (`/evidence`) indexes the artifacts of Tasks 1-9 via `GET /api/tasks/evidence`.
Both are wired into the nav and were verified building and rendering in this audit.

### Not yet built

Dedicated per-engine dashboard pages for Knowledge Representation, the Expert System and Search
Optimisation. Their artifacts are already reachable through the Project Evidence page, but none of the
three has a purpose-built page with the depth the Machine Learning page has. The backend needs no changes
to add them.

---

# 8. "No AI-Generated Fake Results" Requirement

The frontend must **never fabricate model outputs**.

Do not hard-code:

- fake accuracy
- fake ROC-AUC
- fake feature importance
- fake predictions
- fake strategy recommendations
- fake graph data
- fake search results
- fake EDA values

The UI must consume artifacts and API responses generated by the actual Python pipeline.

For example:

```text
Training code
    ↓
trained model
    ↓
evaluation code
    ↓
metrics.json / reports / figures
    ↓
FastAPI
    ↓
Frontend
```

If a model has not been trained, the frontend must show:

```text
Model not trained
```

rather than inventing a result.

---

# 9. Reproducibility

Every experiment should record:

- random seed
- dataset version
- feature metadata version
- model type
- hyperparameters
- training timestamp
- validation strategy
- feature list
- target
- metrics
- software environment
- model artifact path

Use deterministic seeds where the algorithm permits it.

---

# 10. Testing

The integrated project should include:

### Unit tests

- feature generation
- leakage checks
- target construction
- model wrappers
- metric calculation
- expert rules
- search transitions
- API schemas

### Integration tests

Verify:

```text
Task 4 data
    ↓
Task 5 features
    ↓
Task 6 model
    ↓
API
    ↓
frontend-consumable JSON
```

### Regression tests

Known invariants include:

- A* and UCS agree on the optimal search cost.
- no search algorithm returns a path cheaper than UCS.
- leakage columns never enter the model feature matrix.
- selected Task-5 feature lists remain compatible with the training pipeline.
- saved model can be loaded and used for inference.

---

# 11. Running the Project

### Prerequisites

| | Version | Needed for |
|---|---|---|
| Python | 3.10+ (verified on 3.14.6) | everything |
| Node.js + npm | 18+ (verified on 24.15.0 / npm 11.12.1) | the frontend and `./run.sh` |

`./run.sh` also needs ports **8000** and **3000** free — it will `kill` whatever is listening on them
before it starts, so close anything you care about on those ports first.

**Quickest path:** `./run.sh` — sets up the Python env, trains models if needed, starts the backend and
frontend, runs three real sample predictions in your terminal (lap-time, pit-decision, full strategy
simulation), and opens the dashboard in your browser. `./run.sh --force-retrain` retrains first. It's
idempotent — safe to re-run any time — and always builds its sample requests from whatever the live model
registry says the current features are, so it works unchanged on synthetic or real data.

Manual steps, if you want them individually:

```bash
# 1. Python environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                      # makes `app` importable everywhere

# 2. Build everything (Tasks 1-6). Skips a stage if its artifacts already
#    exist; --force regenerates, --skip-ml skips the slow Task 6 stage.
python scripts/build_all.py
python scripts/build_all.py --force
python scripts/build_all.py --skip-ml

# 3. Run the test suite (120 tests: Tasks 1-4 originals + Task 6 + API)
pytest tests/
#    NOTE: the ML tests retrain models and rewrite 10 committed files under
#    artifacts/ (metrics, reports, manifest, model_registry). A clean clone goes
#    dirty just from running the tests. The values are reproducible to ~1e-14 —
#    only timestamps and float noise move — so `git checkout -- artifacts/`
#    afterwards is safe.

# 4. Backend API (http://localhost:8000, docs at /docs)
uvicorn app.api.main:app --reload

# 5. Frontend (http://localhost:3000) — point it at the backend
cd frontend
npm install
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000 npm run dev
```

XGBoost is optional. If it's installed but its native OpenMP runtime is missing (common on macOS), Task 6
reports "XGBoost unavailable — skipped" rather than failing — installing it is one line:
`brew install libomp` (macOS) or the equivalent OpenMP package on Linux.

---

# 12. Data Sources

The system is designed around two complementary sources.

### Historical F1 data

The current data engineering layer supports Ergast-derived/Kaggle-style CSV tables such as:

```text
races.csv
drivers.csv
constructors.csv
circuits.csv
results.csv
pit_stops.csv
lap_times.csv
fastf1_laps.csv
```

### FastF1

FastF1 can provide session-level information such as:

- lap timing
- stint information
- tyre compounds
- weather
- telemetry
- track status
- race-control context

The pipeline should be capable of working with the current reproducible sample/synthetic data while allowing real data to be supplied without changing downstream model code.

---

# 13. Synthetic vs. Real Data

The Task 4 → Task 5 → Task 6 pipeline runs unchanged on either a reproducible **synthetic** session or a
**real FastF1** session — the data source is recorded as a first-class fact
(`data/processed/data_source.json`, exposed as `dataset_source` in `feature_metadata.json`, the model
registry, every API response, and the frontend badge) rather than assumed. Nothing downstream hard-codes
"synthetic" or "real" — the dashboard, reports, and predictions all read this field and describe themselves
accordingly.

### Using real data

```bash
# 1. Fetch a real session's lap data (needs network access; ~3-5s once cached)
python scripts/fetch_real_session.py --year 2023 --event Bahrain --session R

# 2. Re-clean Task 4 on the new raw data (does NOT touch the other synthetic
#    Kaggle-style tables, and does NOT regenerate synthetic laps — see the
#    --regenerate-synthetic guard in scripts/build_all.py)
python scripts/build_all.py --force

# 3. Re-run Task 5's feature-engineering notebook against the new clean data.
#    It is fully parameterised (paths + TOTAL_LAPS are derived from the data,
#    not hard-coded), so re-running it end to end is enough:
jupyter nbconvert --to notebook --execute --inplace \
  docs/notebooks/task5_feature_engineering.ipynb
# (or open it in Jupyter/VS Code and "Run All")

# 4. Retrain Task 6 on the new, real feature matrix
python scripts/build_all.py --force
```

**This has been done in this repository** — the committed `artifacts/`, `data/processed/`, and model
registry currently reflect the **2023 Bahrain Grand Prix (Race)**, 995 modelling rows across 20 drivers,
not the synthetic demo. `data/processed/data_source.json` and every report say so explicitly.

What changed once real strategic variation entered the data (worth reading before trusting any number):

- **Regression got harder, honestly.** CV MAE went from 0.27s (synthetic) to ~1.2s (real); the best model
  changed from `linear_regression` to `decision_tree`. Real lap times have far more structure a 6-feature
  linear model can't capture — see `artifacts/reports/regression_report.md`.
- **Classification stopped being trivially easy.** Real pit stops are spread across 21 distinct laps
  instead of clustered at 2-3 laps, so — for the first time — the chronological holdout test set actually
  contains pit events, and test-set ROC-AUC/PR-AUC are defined (not `undefined*`) for every model.
- **Feature selection picked 45 regression features**, not 6 — with 20 real drivers and 10 real teams,
  one-hot driver/team identity dummies survived the automated selection funnel. With only ~800 development
  rows, this is a real overfitting risk the funnel doesn't itself guard against; treat the regression
  numbers on this single race with appropriate skepticism (this is exactly the kind of judgment call the
  Task 5 notebook's automated "K\* within 2% of best CV MAE" rule can't make for you).

### Remaining next steps for going further on real data

1. **More races.** One session is not a generalisable model. Fetch several real sessions
   (`scripts/fetch_real_session.py` per race) and concatenate them before Task 5, so `race_progress` and
   the field-pace features aren't fit to one track/weather combination. This needs a small change to Task
   4/5 to accept multiple sessions and add a `RaceId`-like grouping key (currently assumes one session).
2. **Revisit the 45-feature regression set.** Either tighten the near-zero-variance/correlation thresholds
   in the notebook for many-driver real data, or cap the one-hot driver/team dummies, before trusting the
   regression numbers.
3. **Replace the other Kaggle-style tables** (`races/drivers/constructors/circuits/results/pit_stops/lap_times.csv`)
   with real Ergast/Kaggle data too — they're still synthetic. They don't feed Task 5/6 (which reads only
   `fastf1_laps_clean.csv`), but Task 4's driver/constructor/season EDA analyses do, and the frontend's
   planned Data & EDA page would inherit that.
4. **Cache real sessions in CI** or accept that `fetch_real_session.py` requires network access; the
   synthetic path remains the default for fast, offline, deterministic testing.

---

# 14. Research Direction

The long-term objective is not merely to predict lap times.

The system should progress from:

```text
Prediction
```

to:

```text
Prediction + Reasoning + Optimisation
```

and ultimately:

```text
Classical Intelligence
        +
Machine Learning
        +
Deep Learning
        +
Explainability
        +
Quantum Optimisation
        ↓
Race Strategy Decision Support
```

The classical search implementation provides an optimisation baseline for later quantum approaches such as QAOA/hybrid optimisation.

Task 5 deliberately keeps the selected feature count compact because model width is especially important for downstream quantum models.

---

# 15. Design Principles

1. **One integrated application**
2. **Clear computational-intelligence boundaries without artificial folder boundaries**
3. **Single source of truth for domain concepts**
4. **No duplicated datasets**
5. **No duplicated feature engineering**
6. **No data leakage**
7. **Chronology-aware validation**
8. **Reproducible experiments**
9. **Actual generated artifacts only**
10. **Explainability by design**
11. **Frontend consumes backend-generated results**
12. **Classical methods establish baselines before quantum methods**
13. **Synthetic results are explicitly labelled**
14. **Every important recommendation should have traceable evidence**

---

# 16. Project Status

| Capability | Status |
|---|---|
| Knowledge Representation | ✅ Implemented |
| Rule-Based Expert System | ✅ Implemented |
| State-Space Search | ✅ Implemented |
| Data Preparation & EDA | ✅ Implemented |
| Feature Engineering & Selection | ✅ Implemented |
| Classical ML (Task 6) | ✅ Implemented — 10 models trained, backend API, dashboard, strategy simulator |
| Deep Learning | ⏳ Planned (Task 7) |
| Explainable AI | ⏳ Planned (Task 8) |
| Unified Deployment | 🚧 Backend + 5 frontend pages done; dedicated Task 1-3 engine pages still planned (Task 9) |
| Responsible AI Evaluation | ⏳ Planned (Task 10) |
| Quantum Optimisation | ⏳ Planned |

---

## 17. Final Goal

The finished application should feel like **one Formula 1 race-strategy intelligence platform**, not ten laboratory submissions placed beside each other.

A user should be able to open the application, inspect the data, understand the domain, compare reasoning engines, evaluate ML models, run a race scenario, see the evidence behind the recommendation, and eventually compare classical optimisation against quantum optimisation — all from one coherent interface.

---

## Licence

MIT — see [LICENSE](LICENSE).

---

## Academic / Laboratory Context

This project implements the Computational Intelligence workflow represented by the supplied **10-task Intelligent Clinical Decision Support System summary**, adapted to the Formula 1 race-strategy domain.

The mapping is:

```text
Knowledge Representation
        ↓
Rule-Based Expert System
        ↓
State-Space Search
        ↓
Data Preparation & EDA
        ↓
Feature Engineering & Selection
        ↓
Machine Learning
        ↓
Deep Learning
        ↓
Explainable AI
        ↓
System Integration & Deployment
        ↓
System Evaluation / Responsible AI
```

The healthcare/CDSS example is therefore used as the **computational-intelligence task structure**, while the actual domain, data, models and outputs of this project remain Formula 1 race strategy.
