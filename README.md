<div align="center">

# 🏎️ Quantum Optimization for Formula 1 Race Strategy

**A computational-intelligence platform for Formula 1 race-strategy decision support** — five reasoning
engines (knowledge representation, a rule-based expert system, state-space search, classical machine
learning, and a feature-engineering contract binding them) composed into one FastAPI backend and one
Next.js dashboard, trained on real FastF1 telemetry from the 2023 Bahrain Grand Prix.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Next.js](https://img.shields.io/badge/Next.js-14.2-000000?logo=nextdotjs&logoColor=white)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/tests-120%20passing-success)](#testing)
[![CI](https://img.shields.io/badge/CI-none%20yet-lightgrey)](#contributing)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-Tasks%201--6%20complete%2C%207--10%20planned-blue)](#implementation-status)

</div>

---

## What this is

Formula 1 race strategy is a decision made under time pressure with incomplete information: pit now or
extend the stint, which compound, accept track position loss for fresher tyres. This repository attacks
that single decision with **four different paradigms** and then makes them answer together.

It is the Formula 1 adaptation of a ten-task Computational Intelligence laboratory specification
originally framed around an Intelligent Clinical Decision Support System. That lineage is deliberate and
worth keeping: the same progression — *represent the domain → reason symbolically → search for the optimum
→ learn from data → explain the result* — is what makes the tasks cohere into a system rather than ten
folders sitting next to each other.

The name promises quantum optimisation. **The quantum layer is not built yet.** What exists is the
classical platform designed to receive it, and this README is explicit about that boundary throughout.

### The payoff, in one request

```jsonc
// POST /api/strategy/predict
{
  "predicted_lap_time_seconds": 102.858,      // ← trained regressor
  "probability_pit":            0.562,        // ← trained classifier
  "recommended_action":         "PIT_NOW",    // ← A* search over the pit state space
  "expected_cost_seconds":      3286.85,      // ← search path cost
  "triggered_expert_rules":     ["R-TYRE-002", "R-RISK-002"],  // ← symbolic rule engine
  "data_source":                "real_fastf1"
}
```

Statistical prediction, symbolic reasoning and combinatorial search in one response. That composition is
the point of the project.

---

## Quickstart

> Every command below was executed and verified in a clean clone of this branch on
> macOS 15 (Darwin 25.6.0), Python 3.14.6, Node 24.15.0. Nothing here is copied forward unverified.

**Prerequisites:** Python 3.10+ · Node.js 18+ and npm · ports 8000 and 3000 free
(`run.sh` kills whatever is listening on them).

```bash
git clone <this-repo> && cd CIL
./run.sh
```

`run.sh` creates the virtualenv, installs dependencies, trains the models if `artifacts/` has none,
starts the API on `:8000` and the dashboard on `:3000`, prints three real predictions to your terminal,
and opens the browser. It is idempotent.

<details>
<summary><strong>Manual setup, step by step</strong></summary>

```bash
# 1. Python environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                          # makes `app` importable

# 2. Build every stage (Tasks 1–6). Skips stages whose artifacts exist.
python scripts/build_all.py               # ~30s from scratch; --force to rebuild
                                          # --skip-ml to skip the slow Task 6 stage

# 3. Tests
pytest                                    # 120 tests, ~64s

# 4. Backend  → http://localhost:8000  (interactive docs at /docs)
uvicorn app.api.main:app --reload

# 5. Frontend → http://localhost:3000
cd frontend && npm install
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000 npm run dev
```

XGBoost is optional. If its native OpenMP runtime is missing (common on macOS), Task 6 logs
*"XGBoost unavailable — skipped"* rather than failing or fabricating a result. Fix with
`brew install libomp`.

</details>

> **One gotcha worth knowing:** `pytest` retrains models and **rewrites 10 committed files under
> `artifacts/`**, so a clean clone goes dirty just from running the tests. The values reproduce to ~1e-14 —
> only timestamps and float noise move — so `git checkout -- artifacts/` afterwards is safe. Making the
> suite hermetic is tracked as a known gap.

---

## Architecture

Verified against the filesystem, not aspirational.

```text
├── app/
│   ├── api/                       FastAPI: main.py, schemas.py
│   │   └── routers/               health · ml · strategy · data · tasks
│   ├── core/paths.py              single source of truth for every path
│   ├── services/
│   │   ├── model_cache.py         loads + caches pipelines (no retrain per request)
│   │   ├── feature_approximation.py
│   │   └── strategy_service.py    ← composes ML + Expert System + Search
│   └── intelligence/
│       ├── knowledge_representation/   Task 1  — ontology, graph, validation
│       ├── expert_system/              Task 2  — rules, inference, explanation
│       ├── search/                     Task 3  — BFS/DFS/UCS/Greedy/A*
│       ├── data/                       Task 4  — pipeline, EDA, synthetic generator
│       ├── features/                   Task 5  — the feature contract reader
│       └── ml/                         Task 6  — splits, preprocessing, training,
│                                                 tuning, evaluation, selection,
│                                                 persistence, registry
├── frontend/                      Next.js 14 + TypeScript + Tailwind
│   ├── app/                       / · /strategy · /machine-learning
│   │                              /data-analysis · /evidence
│   ├── components/                Nav, PredictPanel, ArtifactImage, DatasetBadge,
│   │                              TaskEvidence, Modal, Tooltip, strategy/
│   └── lib/api.ts                 typed fetch client — no local fixture data
├── data/
│   ├── raw/                       source CSVs + .data_source.json provenance marker
│   └── processed/                 the Task 5 feature contract
├── artifacts/                     everything scripts/build_all.py generates
│   ├── knowledge_representation/ expert_system/ search/ data_engineering/
│   ├── models/{laptime,pit_decision}/*.joblib      metrics/  figures/  reports/
│   └── metadata/model_registry.json     manifest.json
├── docs/
│   ├── PROJECT_REPORT.md          the full 17-section technical report
│   ├── notebooks/task5_feature_engineering.ipynb
│   └── task{1,2,3,4}_*.md
├── scripts/                       build_all · run_eda · run_search · run_expert_system
│                                  build_knowledge_base · fetch_real_session · demo_predict
├── tests/                         120 tests, one flat suite
└── run.sh
```

**One application, one domain, one shared data/artifact layer, multiple intelligence engines.**

Full end-to-end data flow — and which stages are real versus stubs — is in **[FLOW.md](FLOW.md)**.

---

## Implementation status

Honest as of the audit on 2026-09-05. ✅ means executed and verified in this repository.

| Capability | Status | Evidence |
|---|---|---|
| Knowledge Representation (Task 1) | ✅ Implemented | 61 entities, 29 relationships, OWL 2 ontology |
| Rule-Based Expert System (Task 2) | ✅ Implemented | 32 validated rules, forward + backward chaining |
| State-Space Search (Task 3) | ✅ Implemented | 5 algorithms; A\* == UCS == 2262.42 s, asserted at build |
| Data Engineering & EDA (Task 4) | ✅ Implemented | full cleaning audit; CSVs regenerate byte-identically |
| Feature Engineering (Task 5) | ✅ Implemented | 4-stage funnel + explicit preprocessing contract |
| Classical ML (Task 6) | ✅ Implemented | 10 models trained; see the caveat below |
| Backend API | ✅ Implemented | all endpoints verified live |
| Frontend dashboard | 🚧 Partial | 5 pages build and render; 3 engine pages not built |
| Continuous integration | ❌ None | no `.github/workflows` — tests are run by hand |
| Deep Learning (Task 7) | ⏳ Planned | — |
| Explainable AI (Task 8) | ⏳ Planned | — |
| Responsible AI Evaluation (Task 10) | ⏳ Planned | — |
| **Quantum Optimisation** | ⏳ **Planned — not started** | the classical platform is built to receive it |

### The model-quality caveat, stated up front

Trained on real FastF1 data (2023 Bahrain GP, 1055 laps, 20 drivers):

| | Selected model | CV | Test |
|---|---|---|---|
| Lap-time regression | `decision_tree` | MAE 1.19 s, R² 0.34 | MAE 0.87 s, **R² −0.17** |
| Pit-decision classification | `random_forest` | ROC-AUC 0.85 | ROC-AUC 0.98, **PR-AUC 0.25, F1 0.077** |

Both bold numbers are real results, not bugs, and both matter:

* **Negative test R²** — the final stint of a single real race is a genuinely hard extrapolation from
  earlier laps with different fuel and tyre state. Selection is by cross-validated MAE (where the model is
  positive), so the procedure is sound; the holdout is small and time-ordered.
* **F1 0.077 at 4.8% prevalence** — pit events are 48 of 995 laps. ROC-AUC 0.98 means the model *ranks*
  laps well; PR-AUC 0.25 and F1 0.077 mean that at the default 0.5 threshold it is **not usable as a
  decision rule** (precision ≈ 0.04). No class weighting, resampling or threshold tuning is applied yet.

Quoting ROC-AUC alone here would be the exact mistake the project's own methodology warns against, so it
is not quoted alone. Full per-fold tables: `artifacts/reports/model_selection_report.md`.

---

## No fabricated results

The project claims every displayed number traces to a generated artifact. That was verified
adversarially, not assumed:

```
API /api/ml/metrics  →  artifacts/metrics/regression_metrics.json
   decision_tree   API=0.8673476599610876   DISK=0.8673476599610876   ✓

MUTATION TEST
   set the on-disk MAE to 999.123456  →  the API returned 999.123456
   ⇒ reads from the artifact, not a hard-coded constant
```

A full `scripts/build_all.py --force` rebuild reproduced every committed artifact: entity counts, rule
counts and search costs **identical**, ML metrics agreeing to ~1e-14. Details in
**[SHOWCASE.md](SHOWCASE.md#the-no-fabricated-results-check)**.

---

## Where to look first

**[SHOWCASE.md](SHOWCASE.md)** ranks the artifacts worth a reviewer's time. The short version:

1. `POST /api/strategy/predict` — three paradigms, one answer
2. `artifacts/search/reports/comparison_report.md` — A\* optimality, asserted at build time
3. `data/processed/feature_metadata.json` — leakage exclusion with the sector-time identity proved numerically
4. `artifacts/expert_system/reports/rule_validation_report.md` — a knowledge base that validates itself
5. `artifacts/data_engineering/figures/dashboard.png` — the best single figure

| Document | What it answers |
|---|---|
| **[FLOW.md](FLOW.md)** | How data moves end to end, and which stages are real vs. stubs |
| **[SHOWCASE.md](SHOWCASE.md)** | What to look at first, and what the project is honest about |
| **[docs/PROJECT_REPORT.md](docs/PROJECT_REPORT.md)** | The full 17-section technical report: leakage prevention, data sources, methodology, research direction |
| **`task-mode` branch** | The five practicals as standalone submissions, each with its own README, FLOW and SHOWCASE |
| **`proj-mode` branch** | This project plus `TODO.md`, the working gap-analysis backlog |

> **Why FLOW and SHOWCASE live here on `main`.** They describe *this* application, so they belong beside
> it — a reviewer landing on `main` should not have to switch branches to learn how the system works or
> what to look at. `TODO.md` is the opposite: it is a working backlog, useful to whoever is doing the
> work, noise to someone evaluating the result. It stays on `proj-mode`. The `task-mode` branch has its
> own FLOW and SHOWCASE because it documents different artifacts — the practicals, not the platform.

---

## Testing

```bash
pytest                    # 120 tests
pytest tests/test_ml_splits.py -v          # leakage & split invariants
pytest tests/test_strategy_service.py -v   # the three-engine composition
```

The suite covers feature-contract validation, expanding-window split correctness, leakage exclusion,
model training and persistence, API schemas, and the original Task 1–4 suites. Regression invariants
asserted include: A\* and UCS agree on optimal cost; no algorithm beats UCS; leakage columns never reach
the feature matrix; saved models reload and infer.

**Known limitation:** the suite is not hermetic — the ML tests rewrite committed artifacts (see the
Quickstart note).

---

## Contributing

There is **no CI pipeline** in this repository. Before opening a pull request, run locally:

```bash
pytest                                    # must stay at 120 passing
python scripts/build_all.py --force       # must exit 0
cd frontend && npm run build              # must compile with 0 errors
git checkout -- artifacts/ data/          # discard test/build artifact churn
```

Conventions: commit messages follow `type: subject` (`feat`, `fix`, `docs`, `chore`, `refactor`).
Every path belongs in `app/core/paths.py` — do not hard-code one. Any value shown in the UI must come
from a generated artifact; hard-coded results violate the project's core contract.

Known gaps are tracked in `TODO.md` on the **`proj-mode`** branch.

---

## Licence

[MIT](LICENSE) © 2026 Anjali More.

## Academic context

This project implements the Computational Intelligence workflow of a supplied ten-task Intelligent
Clinical Decision Support System specification, adapted to the Formula 1 race-strategy domain:

```text
Knowledge Representation → Rule-Based Expert System → State-Space Search →
Data Preparation & EDA → Feature Engineering & Selection → Machine Learning →
Deep Learning → Explainable AI → System Integration & Deployment →
Responsible AI Evaluation
```

Tasks 1–6 and the integration layer are implemented. Tasks 7, 8 and 10, and the quantum optimisation
layer the project is named for, are the road ahead — see [FLOW.md](FLOW.md) and `TODO.md` on `proj-mode`.
