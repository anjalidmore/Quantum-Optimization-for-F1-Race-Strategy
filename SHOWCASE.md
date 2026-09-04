# SHOWCASE — what a reviewer should actually look at

Ordered by how much it should change your opinion of the project. Everything
below was executed and verified, not inferred.

---

## The "no fabricated results" check

Start here, because it determines how much you trust everything else. README §8
claims no value in this project is hand-written. That was tested adversarially:

```
TRACE 1  API /api/ml/metrics  →  artifacts/metrics/regression_metrics.json
   decision_tree   API=0.8673476599610876  DISK=0.8673476599610876  match=True
   random_forest   API=0.9253493148742147  DISK=0.9253493148742147  match=True

TRACE 2  API best_model='decision_tree'  →  model_registry.json ['decision_tree']

TRACE 3  MUTATION TEST
   set the on-disk MAE to 999.123456  →  the API returned 999.123456
   VERDICT: reads from the artifact, not a hard-coded constant
```

The third one is the one that matters. A matching value proves nothing on its
own — a constant could coincide. Changing the artifact and watching the API
follow proves the read path is real.

Corroborating evidence: a full `scripts/build_all.py --force` rebuild reproduced
every committed artifact, with entity counts, rule counts and search costs
**identical** and ML metrics agreeing to ~1e-14.

---

## 1. `POST /api/strategy/predict` — the integration payoff ⭐

One endpoint, three reasoning paradigms, one answer. This is the thing the whole
architecture exists to make possible:

```json
{
  "predicted_lap_time_seconds": 102.85834374999999,
  "probability_pit": 0.5622544303893682,
  "recommended_action": "PIT_NOW",
  "expected_cost_seconds": 3286.8485,
  "triggered_expert_rules": ["R-TYRE-002", "R-RISK-002"],
  "data_source": "real_fastf1"
}
```

The lap time comes from a trained regressor, `triggered_expert_rules` from the
symbolic rule engine, and `expected_cost_seconds` from A\* search over the pit
state space. Most student projects implement these three and leave them in
separate folders. Here they compose.

📄 `app/services/strategy_service.py` · try it at `/docs` once the API is up.

---

## 2. It runs on real F1 data

`data/raw/.data_source.json`:

```json
{ "source": "real_fastf1", "year": 2023, "event": "Bahrain",
  "session": "R", "n_laps": 1055, "n_drivers": 20 }
```

Not synthetic. The 2023 Bahrain Grand Prix race session, 1055 laps, all 20
drivers, fetched via FastF1 (`scripts/fetch_real_session.py`). The provenance
marker propagates all the way to the API responses and to a badge in the UI, so
the dashboard states which dataset it is showing rather than leaving you to
guess.

---

## 3. Task 3 — A\* optimality, asserted at build time

📄 `artifacts/search/reports/comparison_report.md`

| Algorithm | Cost (s) | Optimal | Expanded |
|---|---|---|---|
| BFS | 2270.66 | | 1455 |
| DFS | 2299.93 | | 24 |
| UCS | **2262.42** | ★ | 1589 |
| Greedy | 2270.66 | | 26 |
| A\* | **2262.42** | ★ | 1005 |

A\* matches UCS exactly while expanding 37% fewer nodes. The build *asserts*
this and fails loudly if it ever stops holding. A falsifiable claim beats a
paragraph of prose.

🖼️ `artifacts/search/diagrams/optimal_strategy_path.png`

---

## 4. Leakage prevention, done properly

The single most common failure in student ML projects, handled explicitly here:

* 📄 `data/processed/feature_metadata.json` — `excluded_as_leakage` lists six
  columns with the reason. The sector-time identity (they sum exactly to
  `LapTime`) is demonstrated **numerically** in the notebook before exclusion,
  not merely asserted.
* Splits are **expanding-window, lap-forward**, keeping whole laps in one fold —
  not random K-fold, which would leak future laps into training on a
  time-ordered panel.
* Scaling is fitted **inside** each CV fold. Task 5 deliberately exports
  *unscaled* features and records which columns need scaling, so the modelling
  stage cannot accidentally leak test-fold statistics.

📄 `app/intelligence/ml/splits.py` · `app/intelligence/ml/preprocessing.py`
· 🧪 `tests/test_ml_splits.py`, `tests/test_ml_data_contract.py`

---

## 5. The dashboard

Run `./run.sh`, then:

| Page | What works |
|---|---|
| `/strategy` | Race Strategy Simulator — set race state, get a live recommendation with the rules that fired |
| `/machine-learning` | Model comparison tables, metrics, feature importance, live prediction panel |
| `/data-analysis` | Task 4 figures and reports served straight from `artifacts/` |
| `/evidence` | Per-task artifact index across Tasks 1–9 |
| `/` | Dashboard overview with the dataset-provenance badge |

All five build cleanly (`next build`, 6 routes, 0 errors) and fetch everything
through `frontend/lib/api.ts` — there is no local fixture data in the frontend.

**Best single figure:** 🖼️ `artifacts/data_engineering/figures/dashboard.png`
Also worth opening: `artifacts/figures/roc_curves.png`,
`artifacts/figures/residuals_vs_predictions.png`,
`artifacts/figures/feature_importance.png`.

---

## 6. Task 2 — a knowledge base that validates itself

📄 `artifacts/expert_system/reports/rule_validation_report.md`

A static validator proves the 32-rule base is internally consistent: unique ids,
registered fact keys, no order-dependent contradictions, sane operators and
confidence bounds. Plus five worked scenarios with full HOW/WHY traces —
`inference_01_wet_weather_changeover.md` is the best one to read.

---

## Endpoints worth hitting

```bash
curl -s localhost:8000/api/health              # model count, xgboost status
curl -s localhost:8000/api/ml/metrics          # every metric, from artifacts
curl -s localhost:8000/api/tasks/evidence      # per-task artifact index
curl -s localhost:8000/api/ml/comparison       # model leaderboard
# and the interactive docs:
open http://localhost:8000/docs
```

---

## What this project is honest about

* **Tasks 7–10 and the quantum work do not exist yet.** The name promises
  quantum optimisation; the repository delivers a classical platform built to
  receive it. README §16 says so.
* **Three engine dashboard pages are not built** (Knowledge Representation,
  Expert System, Search have no dedicated page — only the Evidence index).
* **`build_all.py`'s Task 5 stage is a presence check, not a build step** — see
  [FLOW.md](FLOW.md#the-one-thing-to-be-careful-about).
* **The test suite is not hermetic** — it rewrites 10 committed artifact files.
  Reproducibly, but it does dirty a clean clone.

Known gaps are tracked in [TODO.md](TODO.md).
