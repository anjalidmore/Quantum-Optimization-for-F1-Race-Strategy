# FLOW — end-to-end data and code flow

What actually happens, from raw CSV to a number on the dashboard. Every stage
below is marked **real**, **partial** or **stub** based on what was executed and
verified in this repository — not on what was planned.

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
 │ STAGE 5 · Feature engineering                    [REAL, but NOT     │
 │                                                   in the pipeline]  │
 │   docs/notebooks/task5_feature_engineering.ipynb                    │
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
 │   → artifacts/models/dl/*.keras                                     │
 │     artifacts/metrics/dl_{metrics,training_history,vs_classical}    │
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
 │   → artifacts/metadata/xai_results.json                             │
 │     artifacts/reports/xai_*.md, artifacts/figures/xai_*.png         │
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
 │ FRONTEND · Next.js 14                                    [PARTIAL]  │
 │   /                 Dashboard              [real]                   │
 │   /strategy         Race Strategy Simulator[real]                   │
 │   /machine-learning Metrics + live predict [real]                   │
 │   /data-analysis    Task 4 figures/reports [real]                   │
 │   /evidence         Task 1-9 artifact index[real]                   │
 │   dedicated KR / Expert System / Search pages  [NOT BUILT]          │
 └─────────────────────────────────────────────────────────────────────┘
```

## Why Task 7 and 8 sit where they do

**Task 7 does not branch off** — it reads the same Task 5 contract Task 6 reads, and
imports Task 6's `splits.py` and `evaluation.py` rather than copying them. That is
deliberate: the deep-versus-classical comparison is only meaningful if both sides are
scored by the same code on the same holdout. The comparison table's classical rows are
read from `artifacts/metrics/*.json` — Task 6's own committed numbers, the same ones the
Machine Learning dashboard shows.

**Task 8 depends on both and trains nothing.** It loads Task 6's persisted `.joblib`
pipeline through `ModelCache` — the exact model the API serves — and Task 7's saved
`.keras` network, then explains both. If either is missing it raises
`ExplainerUnavailableError` rather than substituting a stand-in, so an explanation is
always an explanation *of the deployed model*.

## The side branch: symbolic engines

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

## Which stages are real, and which are not

| Stage | Status | Evidence |
|---|---|---|
| Raw data ingest | **Real** | `data/raw/.data_source.json` → real FastF1, 2023 Bahrain GP |
| Task 1 Knowledge Representation | **Real** | 61 entities / 29 relationships, OWL 2 ontology regenerated identically |
| Task 2 Expert System | **Real** | 32 rules, static validator passes, 5 worked inference reports |
| Task 3 Search | **Real** | A\* == UCS == 2262.42 s, invariant asserted at build time |
| Task 4 Cleaning & EDA | **Real** | full cleaning audit; cleaned CSVs regenerate byte-identically |
| Task 5 Feature engineering | **Real, but a stub in `build_all.py`** | see below |
| Task 6 Machine learning | **Real** | 10 models trained in 25 s; metrics reproduce to ~1e-14 |
| Task 7 Deep learning | **Real** | 2 Keras MLPs, fully-enumerated grid over the same folds, saved as `.keras` |
| Task 8 Explainable AI | **Real** | SHAP + LIME + counterfactuals + trust + fairness, all on the persisted models |
| API | **Real** | all endpoints verified live; values traced to artifacts |
| Frontend | **Partial** | 7 pages build and render; 3 symbolic-engine pages not built |
| Tasks 9–10, Quantum | **Not started** | listed as planned in the README status table |

### The one thing to be careful about

**`scripts/build_all.py`'s Task 5 stage does not regenerate anything.** Under
`--force` it logs `Task 5 contract present: f1_features_selected.csv,
feature_metadata.json` and finishes in 0.0 s. It is a *presence check*, not a
build step. The real feature engineering lives in
`docs/notebooks/task5_feature_engineering.ipynb` and must be re-run by hand
(see README §13) after changing the underlying data. If you point Task 4 at a
new session and only run `build_all.py --force`, Task 6 will silently retrain on
the **old** feature matrix.

## Reproducibility, measured

`scripts/build_all.py --force` was run from scratch during this audit. Results:

* Every committed artifact regenerated; the build exited 0.
* **Identical:** entity/relationship counts (61/29), rule count (32), all search
  path costs (2262.42 s), all cleaned CSVs, the selected best models
  (`decision_tree` for lap time, `random_forest` for pit decision).
* **Differed only as expected:** embedded "Generated …" timestamps, wall-clock
  timings and peak-memory figures in the search report, and ML metrics at the
  ~1e-14 level (floating-point non-determinism).

Nothing in `artifacts/` is a hand-written constant. That was confirmed
adversarially — see [SHOWCASE.md](SHOWCASE.md#the-no-fabricated-results-check).
