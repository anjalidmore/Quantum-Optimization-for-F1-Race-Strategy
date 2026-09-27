# Architecture

Deep technical reference for the F1 Race Strategy Intelligence platform. See the [README](../README.md) for a quick overview and setup instructions.

## Contents

- [Repository layout](#repository-layout)
- [Task 6 — machine learning](#task-6--machine-learning)
- [Leakage prevention](#leakage-prevention)
- [Frontend](#frontend)
- [No fabricated results](#no-fabricated-results)
- [Reproducibility](#reproducibility)
- [Testing](#testing)
- [Data sources](#data-sources)
- [Synthetic vs. real data](#synthetic-vs-real-data)
- [Design principles](#design-principles)
- [End-to-end flow](#end-to-end-flow)

## Repository layout

```
f1-quantum-strategy/
│
├── app/
│   ├── api/
│   │   ├── main.py                 # FastAPI app, CORS, static /artifacts mount
│   │   ├── schemas.py              # Pydantic request/response models
│   │   └── routers/
│   │       ├── health.py           # GET  /api/health
│   │       ├── ml.py               # GET  /api/ml/{models,metrics,comparison,artifacts,
│   │       │                       #             feature-importance,top-features}
│   │       │                       # POST /api/ml/predict/{laptime,pit}
│   │       ├── strategy.py         # POST /api/strategy/predict
│   │       ├── data.py             # GET  /api/data/options (real driver/team/compound choices)
│   │       └── tasks.py            # GET  /api/tasks/evidence (scans artifacts/ live)
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
│       ├── knowledge_representation/  # Task 1
│       ├── expert_system/             # Task 2
│       ├── search/                    # Task 3
│       ├── data/                      # Task 4
│       ├── features/                  # Task 5 contract reader (contract.py, display.py)
│       └── ml/                        # Task 6 — data_contract, splits, preprocessing,
│                                       # regression, classification, tuning, evaluation,
│                                       # selection, persistence, registry, visualize,
│                                       # reports, pipeline (orchestrator)
│
├── frontend/                        # Next.js 15 + TypeScript + Tailwind
│   ├── app/
│   │   ├── page.tsx                 # Dashboard
│   │   ├── strategy/page.tsx        # Race Strategy Simulator
│   │   ├── models/page.tsx          # Tasks 6 + 7 side by side
│   │   ├── explainability/page.tsx  # Task 8
│   │   └── data/page.tsx            # Task 4 figures + the task evidence index
│   ├── components/                  # one folder per page's sections
│   ├── lib/api.ts                   # typed fetch client (no hard-coded data)
│   └── lib/format.ts                # shared number/path formatting
│
├── data/
│   ├── raw/                         # Kaggle/Ergast-style + FastF1-like CSVs
│   └── processed/                   # fastf1_laps_clean.csv, f1_features_selected.csv,
│                                     # feature_metadata.json, data_source.json — the
│                                     # Task 5 contract + real/synthetic provenance
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
│   ├── architecture.md              # this file
│   ├── screenshots/
│   ├── notebooks/task5_feature_engineering.ipynb
│   └── task{1,2,3,4}_*.md           # per-task documentation (traceability)
│
├── tests/                           # one flat suite — data contract, splits, leakage,
│                                     # training, persistence, API, strategy simulator,
│                                     # plus the original Task 1-4 suites
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
└── run.sh                           # one-command setup + demo
```

**One application, one domain, one shared data/artifact layer, multiple computational-intelligence engines** — not ten separate lab-exercise folders.

## Task 6 — machine learning

Two complementary modelling problems, both built on the same Task 5 feature contract:

**A. Lap-time regression** (`target_laptime`) — "given everything known before the lap begins, what lap time should we expect?" Linear Regression, Decision Tree, Random Forest, SVR, and XGBoost (when available), ranked primarily by MAE.

**B. Pit-decision classification** (`target_pit_next_lap`) — "given the current race state, should the driver pit at the end of this lap?" Logistic Regression, Decision Tree, Random Forest, SVM, and XGBoost (when available), ranked primarily by ROC-AUC/PR-AUC/F1 — never accuracy alone, since pit events are rare.

Both use **expanding-window, lap-forward cross-validation** with a chronologically later, untouched holdout test set — never a random shuffle-split, which would leak future laps into training for this time-ordered panel.

Latest run in this repository — real FastF1 data (2023 Bahrain GP, Race; see [Synthetic vs. real data](#synthetic-vs-real-data)), 10 models trained, 0 fabricated:

| | Best model | CV metric | Test metric |
|---|---|---|---|
| Lap-time regression | `svr` | MAE 1.381 s | MAE 0.781 s, R² 0.302 |
| Pit-decision classification | `random_forest` | PR-AUC 0.386 | PR-AUC 0.25, ROC-AUC 0.983 |

The classifier is selected and scored on PR-AUC, not ROC-AUC: pit events are 4.8% of laps, and at that prevalence ROC-AUC stays high for a model that almost never fires. Its threshold is tuned on pooled out-of-fold predictions (0.4803), not left at 0.5. Full tables, per-fold metrics and the discussion live in `artifacts/reports/*.md`, and are served live by `GET /api/ml/comparison` and the Machine Learning dashboard.

## Leakage prevention

Treated as a first-class engineering requirement, not an afterthought. These same-lap/post-lap fields were explicitly excluded at Task 5 and must never re-enter Task 6 through an alternate preprocessing path:

```
Sector1Time
Sector2Time
Sector3Time
SpeedFL
SpeedST
IsPersonalBest
```

Enforced principles:

1. Only information available before the prediction point may be used.
2. Historical features must be causal.
3. Scalers/preprocessing objects are fit only on training data — inside each CV fold, never on the full dataset up front.
4. Validation respects race/lap chronology.
5. Hyperparameter selection never touches the final test set.
6. Test data remains untouched until final evaluation.
7. All transformations used by a trained model are serialized with the model/pipeline (`joblib`).

## Frontend

Built with Next.js 15 (App Router) + React 19 + TypeScript + Tailwind. Five pages, each rendering real artifacts from the real backend — nothing is mocked. No component is longer than 250 lines, so each one can be read in a sitting.

- **Dashboard** — workflow diagram, honest Task 1–10 progress, clickable task cards showing each task's real generated reports/figures (or "Artifact not generated yet," never a placeholder)
- **Race Strategy Simulator** — driver/team/compound as real dropdowns sourced from the dataset (`GET /api/data/options`), server + client validation, a model selector (best-performing or manual), quick scenario presets built from the data's real ranges, and a "Top Features" tab for a simplified feature-level demo
- **Models** — Tasks 6 and 7 together: comparison tables, ROC/PR curves, confusion matrices, feature importance and live prediction forms for the classical models, then the networks' architectures, training curves, train/validation/test tables and the deep-vs-classical verdict
- **Explainability** — Task 8: global importance, per-group performance, fairness, one card per explained lap, and a lap inspector whose counterfactual is computed live by the saved network
- **Data & Evidence** — Task 4's EDA figures and reports, followed by every task's artifacts, scanned live from `artifacts/`

**Honesty note on the simulator:** Task 6's models need Task 5's engineered features (rolling gap, field-median lag, form-vs-baseline, ...), which require multi-lap race history a single form snapshot can't supply. Driver/team/compound/tyre-age features are computed *exactly* from the form input; history-dependent features fall back to the training data's median. The API response's `approximated_features` field names exactly which ones every time, and an `out_of_range` field flags any feature value that falls outside what the model was actually trained on — a stated engineering approximation, never a silent fabrication.

## No fabricated results

The frontend never hard-codes accuracy, ROC-AUC, feature importance, predictions, strategy recommendations, or EDA values. Everything flows one way:

```
Training code → trained model → evaluation code → metrics.json / reports / figures
    → FastAPI → Frontend
```

If a model hasn't been trained, the UI shows "No trained model available. Run the training pipeline to generate results" — never an invented number.

## Reproducibility

Every trained model records: random seed, dataset version, feature metadata, model type, hyperparameters, training timestamp, validation strategy, feature list, target, metrics, software versions, and artifact path — see `artifacts/metadata/model_registry.json`. Seeds are fixed wherever the algorithm permits it.

## Testing

- **Unit** — feature generation, leakage checks, target construction, model wrappers, metric calculation, expert rules, search transitions, API schemas
- **Integration** — Task 4 data → Task 5 features → Task 6 model → API → frontend-consumable JSON
- **Regression invariants** — A* and UCS agree on optimal search cost; no algorithm returns a cheaper-than-UCS path; leakage columns never enter the training matrix; a saved model reloads and predicts identically; changing the strategy simulator's driver/team dropdown changes at least one feature value (added after that exact bug was found and fixed — see the README's "What I Learned")

Run with `pytest tests/` (120 tests as of this writing).

## Data sources

Two complementary sources, both readable by the same Task 4 pipeline without changing any downstream code:

- **Ergast/Kaggle-style historical tables** — `races.csv`, `drivers.csv`, `constructors.csv`, `circuits.csv`, `results.csv`, `pit_stops.csv`, `lap_times.csv` (used by Task 4's driver/constructor/season EDA, not by Task 5/6)
- **FastF1** — lap timing, stint/tyre info, weather, track status; this is what Task 5/6 actually train on (`fastf1_laps.csv` → `fastf1_laps_clean.csv`)

## Synthetic vs. real data

The Task 4 → Task 5 → Task 6 pipeline runs unchanged on either a reproducible **synthetic** session or a **real FastF1** session — the data source is recorded as a first-class fact (`data/processed/data_source.json`, exposed as `dataset_source` in `feature_metadata.json`, the model registry, every API response, and the frontend badge) rather than assumed.

### Using real data

```bash
# 1. Fetch a real session's lap data (needs network access; ~3-5s once cached)
python scripts/fetch_real_session.py --year 2023 --event Bahrain --session R

# 2. Re-clean Task 4 on the new raw data (does NOT touch the other synthetic
#    Kaggle-style tables, and does NOT regenerate synthetic laps — see the
#    --regenerate-synthetic guard in scripts/build_all.py)
python scripts/build_all.py --force

# 3. Steps 1 and 2 are all that is needed: build_all.py --force rebuilds Task 5
#    as a pipeline stage (app/intelligence/features/build.py) before retraining
#    Tasks 6-8 on the new feature matrix. To rebuild features alone:
python scripts/build_features.py
```

**This has been done in this repository** — the committed `artifacts/`, `data/processed/`, and model registry reflect the **2023 Bahrain Grand Prix (Race)**, 995 modelling rows across 20 drivers, not the synthetic demo.

What changed once real strategic variation entered the data:

- **Regression got harder, honestly.** CV MAE went from 0.27s (synthetic) to ~1.2s (real); the best model changed from `linear_regression` to `svr`. Real lap times have far more structure a 6-feature linear model can't capture.
- **Classification stopped being trivially easy.** Real pit stops are spread across 21 distinct laps instead of clustered at 2–3, so the chronological holdout test set actually contains pit events, and test-set ROC-AUC/PR-AUC are defined (not `undefined*`) for every model.
- **Feature selection picked 45 regression features**, not 6 — with 20 real drivers and 10 real teams, one-hot driver/team identity dummies survived the automated selection funnel. With only ~800 development rows this is a real overfitting risk the funnel doesn't itself guard against.

### Remaining next steps for real data

1. **More races.** One session isn't a generalizable model. Fetching several real sessions and concatenating them before Task 5 needs a small change to add a race-grouping key (currently assumes one session).
2. **Revisit the 45-feature regression set** — tighten the near-zero-variance/correlation thresholds for many-driver real data, or cap the one-hot dummies.
3. **Replace the other Kaggle-style tables** with real Ergast/Kaggle data too — they don't feed Task 5/6, but Task 4's driver/constructor/season EDA does.
4. **Cache real sessions in CI**, or accept that `fetch_real_session.py` needs network access; the synthetic path stays the default for fast, offline, deterministic testing.

## Design principles

1. One integrated application, not ten lab-exercise folders
2. Clear computational-intelligence boundaries without artificial directory boundaries
3. Single source of truth for domain concepts and data/artifact paths
4. No duplicated datasets or feature-engineering logic
5. No data leakage, ever
6. Chronology-aware validation for time-ordered data
7. Reproducible experiments (seeds, versions, full metadata recorded)
8. Only actual generated artifacts are shown — never a fabricated one
9. The frontend consumes backend-generated results exclusively
10. Synthetic vs. real data is always explicitly labelled, never assumed
11. Every recommendation has traceable evidence back to real computation

## End-to-end flow

What happens from raw CSV to a number on the dashboard. Each stage is marked **real**, **partial** or **stub** from what was actually executed and verified here, not from what was planned.


```text
 ┌─────────────────────────────────────────────────────────────────────┐
 │ data/raw/                                          [REAL DATA]      │
 │   circuits, constructors, drivers, lap_times, pit_stops, races,     │
 │   results, fastf1_laps                                              │
 │   .data_source.json → real_fastf1, 2023 Bahrain GP (R),             │
 │                       1055 laps, 20 drivers                         │
 └────────────────────────────────┬────────────────────────────────────┘
                                  │  scripts/run_eda.py
                                  │  app/intelligence/data/pipeline.py
                                  ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │ STAGE 4 · Cleaning & EDA                                    [REAL]  │
 │   dedupe → dtype coercion → m:ss.mmm → seconds → categorical        │
 │   normalisation → imputation → IQR outlier detection                │
 │   → artifacts/data_engineering/{clean,figures,reports}              │
 │   Full audit trail in reports/cleaning_audit.md                     │
 └────────────────────────────────┬────────────────────────────────────┘
                                  │
                                  ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │ STAGE 5 · Feature engineering                               [REAL]  │
 │   app/intelligence/features/build.py                                │
 │   (scripts/build_features.py; the notebook is a walkthrough)         │
 │   4-stage funnel: near-zero variance → correlation → VIF →          │
 │   importance with fold stability                                    │
 │   → data/processed/f1_features_selected.csv                         │
 │     data/processed/feature_metadata.json   ← the contract           │
 └────────────────────────────────┬────────────────────────────────────┘
                                  │  app/intelligence/features/contract.py
                                  ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │ STAGE 6 · Machine learning                                  [REAL]  │
 │   app/intelligence/ml/pipeline.py                                   │
 │   splits (expanding-window, lap-forward) → preprocessing (fitted    │
 │   INSIDE each fold) → 5 regressors + 4 classifiers → tuning →       │
 │   evaluation → selection → persistence → registry                   │
 │   → artifacts/models/{laptime,pit_decision}/*.joblib                │
 │     artifacts/metrics/*.json, artifacts/reports/*.md                │
 │     artifacts/metadata/model_registry.json                          │
 └────────────────────────────────┬────────────────────────────────────┘
                                  │
                                  ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │ STAGE 7 · Deep learning                                     [REAL]  │
 │   app/intelligence/dl/pipeline.py                                   │
 │   imports ml.splits + ml.evaluation directly, so DL and classical   │
 │   numbers are produced by the SAME code                             │
 │   Keras MLPs (torch backend): linear head / sigmoid head            │
 │   → artifacts/models/deep_learning/*/f1_dnn_model.h5 (private)      │
 │     artifacts/deep_learning/ (reports, curves, comparison)          │
 │     model_registry.json extended (not duplicated)                   │
 └────────────────────────────────┬────────────────────────────────────┘
                                  │
                                  ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │ STAGE 8 · Explainable AI                                    [REAL]  │
 │   app/intelligence/xai/pipeline.py                                  │
 │   explains Task 6's PERSISTED pipelines + Task 7's saved networks    │
 │   permutation importance · SHAP (Tree exact / Kernel sampled) ·      │
 │   LIME · counterfactual scan + DiCE · trust score · fairness         │
 │   → artifacts/xai/xai_metadata.json                                 │
 │     artifacts/xai/{shap,lime,counterfactual,stratification}/, *.md  │
 └────────────────────────────────┬────────────────────────────────────┘
                                  │  app/services/model_cache.py (load once, cache)
                                  ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │ API · FastAPI                                               [REAL]  │
 │   /api/health              /api/ml/{models,metrics,comparison,…}    │
 │   /api/ml/predict/laptime  /api/ml/predict/pit                      │
 │   /api/strategy/predict    ← ML + Expert System + Search combined   │
 │   /api/data/*              /api/tasks/evidence                      │
 │   /artifacts/*             ← static mount, serves figures/reports   │
 └────────────────────────────────┬────────────────────────────────────┘
                                  │  frontend/lib/api.ts (typed fetch, no local data)
                                  ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │ FRONTEND · Next.js 15                                       [REAL]  │
 │   /                 Dashboard              [real]                   │
 │   /strategy         Race Strategy Simulator[real]                   │
 │   /models           Tasks 6 + 7 compared   [real]                   │
 │   /explainability   Task 8                 [real]                   │
 │   /data             Task 4 + evidence index[real]                   │
 └─────────────────────────────────────────────────────────────────────┘
```


### Why Task 7 and 8 sit where they do

**Task 7 does not branch off** — it reads the same Task 5 contract Task 6 reads, and
imports Task 6's `splits.py` and `evaluation.py` rather than copying them. That is
deliberate: the deep-versus-classical comparison is only meaningful if both sides are
scored by the same code on the same holdout. The comparison table's classical rows are
read from `artifacts/metrics/*.json` — Task 6's own committed numbers, the same ones the
Machine Learning dashboard shows.

**Task 8 depends on both and trains nothing.** It loads Task 6's persisted `.joblib`
pipeline through `ModelCache` — the exact model the API serves — and Task 7's saved
`f1_dnn_model.h5` network, then explains the network (Task 6's model is the second opinion). If either is missing it raises
`ExplainerUnavailableError` rather than substituting a stand-in, so an explanation is
always an explanation *of the deployed model*.


### The side branch: symbolic engines

Tasks 1–3 do not sit in the data pipeline. They are built by
`scripts/build_all.py` into `artifacts/`, surfaced read-only through
`/api/tasks/evidence` and the Evidence page, and — importantly — two of them are
**wired into the live strategy recommendation**:

```text
app/services/strategy_service.py
    ├── ML          → predicted lap time, pit probability
    ├── Expert Sys  → triggered_expert_rules  (e.g. R-TYRE-002, R-RISK-002)
    └── Search      → expected_cost_seconds, recommended_action
```

A single `POST /api/strategy/predict` returns all three. That is the one place
where the symbolic and statistical halves of the project actually meet, and it
is real — verified in this audit returning `recommended_action: "PIT_NOW"` with
two triggered rule ids and a search cost.


### Which stages are real, and which are not

| Stage | Status | Evidence |
|---|---|---|
| Raw data ingest | **Real** | `data/raw/.data_source.json` → real FastF1, 2023 Bahrain GP |
| Task 1 Knowledge Representation | **Real** | 61 entities / 29 relationships, OWL 2 ontology regenerated identically |
| Task 2 Expert System | **Real** | 32 rules, static validator passes, 5 worked inference reports |
| Task 3 Search | **Real** | A\* == UCS == 2262.42 s, invariant asserted at build time |
| Task 4 Cleaning & EDA | **Real** | full cleaning audit; cleaned CSVs regenerate byte-identically |
| Task 5 Feature engineering | **Real** | a `build_all.py` stage since 2026-09-27; regenerates the committed contract exactly |
| Task 6 Machine learning | **Real** | 10 models trained in 25 s; metrics reproduce to ~1e-14 |
| Task 7 Deep learning | **Real** | 2 Keras MLPs, one-factor-at-a-time search over the same folds, saved as reload-verified `.h5` |
| Task 8 Explainable AI | **Real** | SHAP + LIME + consistent tyre-age counterfactuals + trust + driver/team/compound stratification, on the Task 7 DNN |
| API | **Real** | all endpoints verified live; values traced to artifacts |
| Frontend | **Real** | every page builds and reads its values from the API |
| Tasks 9–10, Quantum | **Not started** | listed as planned in the README status table |


### Rebuilding the feature contract

`scripts/build_all.py`'s Task 5 stage used to be a presence check that finished in 0.0 s: the
real feature engineering lived only in the notebook, so pointing Task 4 at a new session and
running `build_all.py --force` left Task 6 training on the **previous** race's feature matrix.

The logic now lives in `app/intelligence/features/build.py`, and the stage rebuilds the
contract like any other. It reproduces the committed `f1_features_selected.csv` and
`feature_metadata.json` exactly — every stochastic step is seeded with `random_state=42`.
