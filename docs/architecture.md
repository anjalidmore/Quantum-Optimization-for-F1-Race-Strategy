# Architecture

Deep technical reference for the F1 Race Strategy Intelligence platform. See the [README](../README.md) for a quick overview, screenshots, and setup instructions.

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
├── frontend/                        # Next.js 14 + TypeScript + Tailwind
│   ├── app/
│   │   ├── page.tsx                 # Dashboard
│   │   ├── strategy/page.tsx        # Race Strategy Simulator
│   │   ├── machine-learning/page.tsx
│   │   ├── data-analysis/page.tsx
│   │   └── evidence/page.tsx        # Project Evidence
│   ├── components/
│   └── lib/api.ts                   # typed fetch client (no hard-coded data)
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
| Lap-time regression | `decision_tree` | MAE 1.19s | MAE 0.87s, R² −0.17 |
| Pit-decision classification | `random_forest` | ROC-AUC 0.85 | ROC-AUC 0.98 |

The regression test R² being negative is an honest result, not a bug: a single real race, a compact feature-selected model, and a fuel/tyre state very different from the training laps in the final stint is a genuinely hard extrapolation. Full tables, per-fold metrics, and the full discussion live in `artifacts/reports/*.md` and are served live by `GET /api/ml/comparison` / the Machine Learning dashboard.

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

Built with Next.js 14 (App Router) + TypeScript + Tailwind. Every page renders real artifacts and calls the real backend — nothing is mocked.

- **Dashboard** — workflow diagram, honest Task 1–10 progress, clickable task cards showing each task's real generated reports/figures (or "Artifact not generated yet," never a placeholder)
- **Race Strategy Simulator** — driver/team/compound as real dropdowns sourced from the dataset (`GET /api/data/options`), server + client validation, a model selector (best-performing or manual), quick scenario presets built from the data's real ranges, and a "Top Features" tab for a simplified feature-level demo
- **Machine Learning** — model comparison tables, ROC/PR curves, confusion matrix, feature importance, and a live prediction form, all reading `GET /api/ml/{comparison,artifacts,models,feature-importance}`
- **Data & Analysis** — Task 4's real EDA figures and reports
- **Project Evidence** — every task's artifacts in one place, scanned live from `artifacts/`

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

# 3. Re-run Task 5's feature-engineering notebook against the new clean data.
#    It is fully parameterised (paths + TOTAL_LAPS are derived from the data,
#    not hard-coded), so re-running it end to end is enough:
jupyter nbconvert --to notebook --execute --inplace \
  docs/notebooks/task5_feature_engineering.ipynb

# 4. Retrain Task 6 on the new, real feature matrix
python scripts/build_all.py --force
```

**This has been done in this repository** — the committed `artifacts/`, `data/processed/`, and model registry reflect the **2023 Bahrain Grand Prix (Race)**, 995 modelling rows across 20 drivers, not the synthetic demo.

What changed once real strategic variation entered the data:

- **Regression got harder, honestly.** CV MAE went from 0.27s (synthetic) to ~1.2s (real); the best model changed from `linear_regression` to `decision_tree`. Real lap times have far more structure a 6-feature linear model can't capture.
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
