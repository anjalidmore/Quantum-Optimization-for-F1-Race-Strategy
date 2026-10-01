# Software Design Document

How the F1 Race Strategy Intelligence system is actually built, at the level of modules,
components, data shapes and control flow. Where [`architecture.md`](architecture.md) gives the
repository layout and the high-level end-to-end diagram, this document goes one level deeper into
the services that sit between the API and the intelligence packages, and into the exact stage
sequence one strategy request goes through.

## Contents

- [1. Architecture overview](#1-architecture-overview)
- [2. Component design](#2-component-design)
- [3. Data design](#3-data-design)
- [4. The strategy pipeline: stage sequence](#4-the-strategy-pipeline-stage-sequence)
- [5. Error handling and honesty patterns](#5-error-handling-and-honesty-patterns)
- [6. Deployment view](#6-deployment-view)

## 1. Architecture overview

The system is four layers, each with one job, wired together by a single shared data/artifact
contract rather than by direct imports between intelligence packages:

```
frontend/            Next.js 15 dashboard — six pages, one typed fetch client, no local data
        │  HTTP (JSON, fetched at request/render time)
        ▼
app/api/              FastAPI routers + Pydantic schemas — the only externally reachable surface
        │
        ▼
app/services/          Orchestration — chains intelligence modules together, caches loaded models,
                        approximates features from a UI snapshot, generates the strategy report
        │
        ▼
app/intelligence/       One package per computational-intelligence task (1–8), each independently
                         testable, each reading/writing only through app/core/paths.py
        │
        ▼
data/, artifacts/       The only persistent state: raw/processed data and every generated model,
                         metric, figure and report. No database.
```

### Module-responsibility table

| Module | Responsibility | Depends on |
|---|---|---|
| `app/core/paths.py` | Single source of truth for every filesystem path (data + artifacts); the `ArtifactPaths` dataclass lets the test suite redirect all writes to a temp root without touching production code | nothing else in the project |
| `app/core/runtime.py` | Import-order guard so XGBoost and the PyTorch-backed Keras coexist without a native OpenMP segfault | `app/core/paths.py` |
| `app/intelligence/knowledge_representation/` | Task 1: entity/attribute/relationship schema, ontology and graph generation | `app/core/paths.py` |
| `app/intelligence/expert_system/` | Task 2: rule base definition, forward/backward-chaining inference engine | `app/core/paths.py` |
| `app/intelligence/search/` | Task 3: race-as-search problem formulation, BFS/DFS/UCS/Greedy/A\* | `app/core/paths.py` |
| `app/intelligence/data/` | Task 4: cleaning, imputation, outlier capping, EDA | `app/core/paths.py` |
| `app/intelligence/features/` | Task 5: feature construction (`build.py`), the feature contract reader (`contract.py`), and display helpers (`display.py`) that every other module treats as the sole feature-name authority | Task 4's cleaned CSV |
| `app/intelligence/ml/` | Task 6: data contract adapter, time-aware splits, preprocessing (fit-inside-fold), regression + classification model families, tuning, evaluation, selection, persistence, registry | `app/intelligence/features/contract.py` |
| `app/intelligence/dl/` | Task 7: Keras network definitions, training loop, persistence (HDF5 + reload verification) | imports `app/intelligence/ml/splits.py` and `.evaluation.py` directly (not copied) |
| `app/intelligence/xai/` | Task 8: permutation importance, SHAP, LIME, counterfactual scan, trust score, narrative generation, fairness stratification, plus the reduced-budget live-explanation path (`live.py`) | Task 6's persisted `.joblib` (through `ModelCache`) and Task 7's saved `.h5` network |
| `app/intelligence/qml/` | Quantum-ML extension: PennyLane circuits, PCA-reduced encoding, parameter-matched classical baselines | `app/intelligence/ml/splits.py` |
| `app/services/model_cache.py` | Loads a Task 6 pipeline once per process and caches it; resolves "the selected-best model" from the registry when no specific model is named | `app/intelligence/ml/persistence.py`, `.registry.py` |
| `app/services/feature_approximation.py` | Maps a race-state *snapshot* (from the Strategy page form) onto a full Task 5 feature row, generically from the live feature contract | `app/intelligence/features/contract.py` |
| `app/services/strategy_service.py` | Orchestrates ML + expert system + search into one recommendation | `model_cache`, `feature_approximation`, `expert_system`, `search` |
| `app/services/strategy_report.py` | Renders one `run_strategy_analysis` result as a self-contained Markdown briefing | `strategy_service`'s output shape only — computes nothing itself |
| `app/api/routers/*.py` | One router per task/concern; every handler either returns real computed/read data or a structured 404/503 | the corresponding `app/services/` or `app/intelligence/` module |
| `frontend/lib/api.ts` | Typed fetch client — the only place the frontend knows the backend's URL shape | `NEXT_PUBLIC_API_BASE_URL` |
| `frontend/app/*/page.tsx` | Six server/client components, each fetching its own data at render/interaction time | `frontend/lib/api.ts` |

## 2. Component design

### 2.1 `app/services/strategy_service.py`

**Responsibility.** The single place where the ML, expert-system and search paradigms actually
meet for a live request. It is a thin orchestrator with no model-fitting or rule-authoring logic of
its own — every real computation is delegated.

**Key functions** (all module-level, no class — there is no per-request state to hold):

- `_run_ml(race_state, laptime_model, pit_model) -> dict` — for each of the two targets, builds a
  feature row via `feature_approximation.build_feature_row`, records which features were
  approximated or out-of-range, fetches a cached pipeline via `model_cache.get_pipeline`, and
  predicts. A `ModelUnavailableError` for one target is appended to `errors` and does not abort the
  other target's prediction.
- `_run_expert_system(race_state) -> dict` — builds a 16-key `inputs` dict (mapping race-state fields
  to the rule base's vocabulary, e.g. `tyre_wear = tyre_age * 3.0` as a percentage, `fuel_margin` as a
  computed ratio) and runs `InferenceEngine.forward_chain`, returning every firing with its matched
  conditions. Four of those keys — `in_pit_window`, `safety_car_probability`, `safety_car_likelihood`,
  `tyre_age_laps` — were added after a reachability audit (`scripts/evaluate_system.py`'s
  `rule_base_reachability_check`) found 14 of the 32 rules referenced an input this function never
  populated; adding these four (all genuinely derivable — see the module-level "Rule-base input
  coverage" comment in `strategy_service.py`) made R-SC-002, R-SC-003, R-PIT-004 and R-STRAT-004
  reachable. `safety_car_probability`/`_likelihood` come from `_historical_safety_car_rate()`, a
  cached empirical SC/VSC-lap rate read from the cleaned training data (2.94% for 2023 Bahrain), not a
  live incident prediction. The remaining 10 unreachable rules need rival-car telemetry, multi-circuit
  data, or a tyre-graining model this project doesn't have, and are documented as limitations rather
  than worked around (§7, below).
- `_run_search(race_state) -> dict` — builds a fresh `RaceProblem` for the *remaining* laps only
  (search does not carry over tyre age already accumulated — stated explicitly in the returned
  `note` field) and runs `astar_search`.
- `_explain(ml) -> dict` — opt-in (only called when `explain=True`); imports
  `app.intelligence.xai.live` locally so a plain prediction call never pays the cost of loading the
  deep-learning stack. Wraps each target's explanation in a `try/except` so an explainer failure
  degrades to `{"available": False, "reason": ...}` rather than failing the whole request.
- `run_strategy_analysis(race_state, laptime_model, pit_model, explain) -> dict` — the public entry
  point (called by `app/api/routers/strategy.py`). Composes the three private runners, decides
  `recommended_action` (the expert system's `pit_decision` conclusion if present, else the ML
  classifier's own class), and assembles the final response dict.

**Current response shape** (`app/services/strategy_service.py::run_strategy_analysis`): `race_state`,
`validation` (the normalised inputs plus `laps_remaining` — a stage-1 card for the frontend, not a
re-validation; the schema and domain-option checks already ran at the API layer), `feature_construction`
(`feature_rows`/`approximated_features`/`out_of_range`/`context_only`, duplicated into `prediction`
below for backward compatibility with existing callers), `prediction` (the Task 6 ML result:
`predicted_lap_time_seconds`, `laptime_model`, `probability_pit`, `pit_model`, plus the
feature-construction fields again, `errors`), `dl_prediction` (the Task 7 network's lap-time and
pit-probability predictions at the tuned decision threshold — see §4), `recommended_action` (kept for
backward compatibility: the expert system's raw `pit_decision` conclusion, or the ML class as a
fallback), `recommendation` (the richer combiner's output: `action` one of `"PIT NOW"` /
`"STAY OUT"` / `"PIT IN N LAPS"`, `confidence`, `source`, `disagreement`, `ml_pit_probability`,
`dl_pit_probability`, `reason` — see §4), `expected_cost_seconds`, `optimal_search_strategy`
(aliased as `search_plan`), `triggered_expert_rules`, `evidence`, `xai_explanation` (`{"available":
false, "reason": ...}` when `explain=false`; otherwise identical to the legacy `explanation` key,
kept alongside it rather than renaming it, so existing tests/report code did not need to change),
`data_source`.

### 2.2 `app/services/model_cache.py`

**Responsibility.** Keep exactly one loaded copy of each trained Task 6 pipeline in memory per
process, and resolve "give me the best model for this task" without every caller having to read the
registry itself.

**Design.** `ModelCache` holds a `dict[(task, model_name), pipeline]`. `get_pipeline(task,
model_name=None)` resolves an omitted `model_name` to the registry's `is_selected_best` entry for
that task, raises `ModelUnavailableError` (a plain `RuntimeError` subclass, not an HTTP exception —
callers translate it) if nothing is registered or the artifact file is missing, and otherwise loads
the `.joblib` pipeline once with `joblib.load` and caches it for the process lifetime. A
module-level `get_model_cache()` is memoized with `functools.lru_cache(maxsize=1)`, so the whole
application shares one cache instance without any explicit dependency injection.

**Why this shape.** Training is a separate, offline operation
(`scripts/build_all.py` / `pipeline.train_all()`); the API is documented and tested as read-only at
request time (`docs/DEPLOYMENT.md` §5: "Workers are safe: the API is read-only at request time...
nothing writes to `artifacts/` while serving"), which is only true because prediction requests never
touch the training code path.

### 2.3 `app/services/feature_approximation.py`

**Responsibility.** Turn a race-state *snapshot* (the handful of fields a UI form or API caller can
realistically supply — driver, team, compound, tyre age, lap, temperature, ...) into a complete
Task 5 feature row, for models that were trained on features requiring multi-lap history a snapshot
cannot supply.

**Design — the important part is that it is generic, not a fixed mapping.** `build_feature_row`
iterates the *live* feature contract's selected feature names for the target and classifies each
one at request time by pattern-matching its name against known Task 5 naming conventions:

- `driver_<code>` / `team_<slug>` — one-hot identity dummies, matched via regex
  (`_DRIVER_RE`, `_TEAM_RE`) against the contract's own `binary_features_no_scaling_needed` list to
  tell "genuine one-hot dummy" apart from "continuous per-driver statistic that happens to share a
  name" (see the module's docstring for the exact historical bug this fixes: a hard-coded
  `driver_sai` special case silently broke on real data because a real one-hot dummy for driver
  Carlos Sainz happens to share that literal name with the synthetic dataset's driver-skill index).
- `compound_<slug>` — one-hot compound dummy.
- `tyrelife_x_<slug>` — tyre-age × compound interaction, zero unless the compound matches.
- `tyre_life`, `race_progress`, `tracktemp_dev_x_tyrelife` — computed directly from race-state
  fields.
- Anything else — filled from the training data's per-driver/per-team/overall median and recorded
  in `approximated`, because it needs history (a rolling gap, a field-pace lag) the snapshot cannot
  supply.

`_flag_out_of_range` then checks every *exactly derived* (non-approximated, non-binary) feature
value against the training column's real min/max and records any that fall outside it — the source
of the strategy response's `out_of_range` field and the dashboard's "Extrapolating beyond training
data" warning.

### 2.4 `app/intelligence/xai/live.py` — the live-explanation path

**Responsibility.** Explain one arbitrary, caller-supplied feature row while the caller waits,
using the *same* trust/SHAP/narrative code the batch Task 8 pipeline uses, at a reduced sampling
budget.

**Design.** `explain_feature_row(target, row)`:

1. Loads (and process-caches, in a plain module-level `_CACHE` dict) a `TargetBundle` via
   `xai.loading.load_target(target)` — the trained DNN, the classical second-opinion model, the
   training data needed as a SHAP background, and the decision threshold.
2. Validates that `row` actually contains every feature the model expects; returns a structured
   `available: False` payload naming the missing features rather than raising.
3. Runs `shap_analysis.kernel_shap` with `LIVE_NSAMPLES = 60` and `LIVE_BACKGROUND_K = 15` — both
   far smaller than the batch pipeline's budget, and both reported back in the response's `method`
   block so a caller can see exactly how this differs from a committed report.
4. Deliberately skips LIME (a 2,000-perturbation surrogate per request is judged not worth the
   latency) and calls `trust.compute(..., shap_top=None, lime_top=None, ...)`, which drops the
   `explanation_stability` term and **renormalises the remaining weights** rather than scoring a
   missing component as zero — stated explicitly in the module docstring as the reason the trust
   score for a live explanation is not bit-for-bit the same formula as a committed Task 8 report's.
5. Generates a plain-English sentence via `narrative.pit_decision_sentence` /
   `narrative.laptime_sentence` and returns SHAP factors, the two models' raw predictions, the
   trust score/band/components, and the narrative together.

This is the module `strategy_service._explain` calls for `explain=True` strategy predictions, and
that `strategy_report.py` always calls (with `explain=True` forced) when rendering a downloadable
report — so the Explainability page's committed-artifact explanations and a fresh strategy report's
explanation are produced by variations of the same code, at different sampling budgets, never by
two independently-written implementations.

## 3. Data design

### 3.1 The feature contract

The single artifact every downstream module (Task 6, 7, 8, and the live services above) treats as
authoritative for "what is a valid feature and what is its type" is
`data/processed/feature_metadata.json`, read through `app/intelligence/features/contract.py`. As
inspected for this document, its top-level structure is:

| Key | Content |
|---|---|
| `task` | `"Phase 2 / Task 5 - Feature Engineering & Feature Selection"` |
| `source_dataset` | `data/processed/fastf1_laps_clean.csv` |
| `dataset_source` | `{source, year, event, session, fetched_at, n_laps, n_drivers}` — real-vs-synthetic provenance |
| `rows` | `995` |
| `random_state` | `42` |
| `identifier_columns` | 5 non-feature id columns kept for traceability |
| `targets` | `target_laptime`, `target_pit_next_lap`, `target_laptime_fuel_corrected` |
| `selected_features` | `{target_laptime: [...45 names...], target_pit_next_lap: [...8 names...], union_exported: [...]}` |
| `feature_provenance` | per-feature dict of where it came from (e.g. one-hot compound/driver dummies) |
| `numeric_features_requiring_scaling` | 14 features |
| `binary_features_no_scaling_needed` | 31 features |
| `preprocessing_contract` | `{scaling: "NOT applied here — fit inside each CV fold", validation: "expanding-window lap-forward, no random K-fold", warmup_rows_dropped: 60, first_usable_lap: 4}` |
| `selection_funnel` | what each of the four selection stages dropped (e.g. stage 1 dropped `is_rainfall` for near-zero variance; stage 2 dropped `tyre_life_sq`, `air_temp`, `track_air_delta` for correlation; stage 3 dropped `race_progress` for VIF; stage 4 is the importance/stability ranking that chose K\*=45) |
| `excluded_as_leakage` | `{columns: [Sector1Time, Sector2Time, Sector3Time, SpeedFL, SpeedST, IsPersonalBest], reason: "Sector times sum exactly to LapTime; speed traps and IsPersonalBest are only knowable during or after the lap being predicted."}` |
| `validation_scores` | cross-validated MAE/R²/AUC recorded at selection time, with a stated caveat |

The feature matrix itself, `data/processed/f1_features_selected.csv`, is 995 rows: one row per
usable driver-lap from the 2023 Bahrain GP (1,055 raw laps minus a 60-row warm-up trim, first usable
lap 4). Lap-time regression trains on 45 selected features (17 race-state features and 28
driver/team one-hot dummies, per the README's Task 7 section); pit-decision classification trains
on 8.

### 3.2 Artifact layout

`app/core/paths.py` defines every path as either a module-level constant or, for the pipelines'
write targets, a property on the `ArtifactPaths` frozen dataclass — the latter exists specifically
so the test suite can pass `ArtifactPaths(root=<tmp dir>)` and redirect every write, which is what
makes `pytest` hermetic (the module docstring records that before this existed, running the test
suite retrained Task 6 and overwrote ten tracked files, so a clean clone went dirty just from
running the documented test command).

Top-level artifact directories: `models/{laptime,pit_decision,deep_learning,qml}` (private — never
mounted over HTTP), `deep_learning/`, `xai/`, `metrics/`, `figures/`, `reports/`, `metadata/`,
`chart_data/` (per-chart JSON so the frontend can draw its own figures instead of showing a static
PNG), plus the Task 1–4 directories (`knowledge_representation/`, `expert_system/`, `search/`,
`data_engineering/`, `feature_engineering/`).

### 3.3 Data flow, stage to artifact

```
data/raw/  →  [Task 4 clean]  →  data/processed/fastf1_laps_clean.csv
                                            │
                                  [Task 5 build.py]
                                            │
                    data/processed/f1_features_selected.csv  (995 × 53: ids + features + 3 targets)
                    data/processed/feature_metadata.json      (the contract)
                                            │
                          ┌─────────────────┼─────────────────┐
                   [Task 6 ml/pipeline]            [Task 7 dl/pipeline]
                          │                                    │
        artifacts/models/{laptime,pit_decision}/*.joblib   artifacts/models/deep_learning/*/f1_dnn_model.h5
        artifacts/metrics/*.json, artifacts/reports/*.md   artifacts/deep_learning/*
                          └─────────────────┬─────────────────┘
                                   [Task 8 xai/pipeline]
                        (loads Task 6's .joblib through ModelCache,
                         Task 7's saved .h5 — trains nothing itself)
                                            │
                                  artifacts/xai/*
                                            │
                                  app/services/model_cache.py (in-process cache)
                                            │
                                     app/api/ (FastAPI)
                                            │
                                frontend/lib/api.ts (typed fetch)
                                            │
                              six dashboard pages, each real
```

## 4. The strategy pipeline: stage sequence

What `POST /api/strategy/predict` and `POST /api/strategy/report` actually execute, as read from
`app/api/routers/strategy.py` and `app/services/strategy_service.py` in this worktree:

1. **Input validation.** `RaceStateRequest` (Pydantic, `app/api/schemas.py`) enforces field-level
   constraints (`current_lap >= 1`, `tyre_age >= 0`, `current_position` 1–24, a cross-field
   validator rejecting `tyre_age > current_lap` or `current_lap > total_laps`). The router then
   calls `_validate_against_known_options`, which checks driver/team/tyre_compound against
   `GET /api/data/options`'s real dataset values (case-insensitive), returning HTTP 422 with the
   valid option list on a mismatch — never silently accepting an unknown value. `_validate_model_choice`
   similarly rejects a named `laptime_model`/`pit_model` that isn't a registered, non-deep model for
   that task.
2. **Feature construction.** `feature_approximation.build_feature_row` maps the validated race state
   onto each target's full Task 5 feature row (§2.3 above), recording approximated and out-of-range
   features.
3. **ML prediction.** `model_cache.get_pipeline` resolves and loads (from cache) the requested or
   selected-best Task 6 pipeline for each target and predicts (`strategy_service._run_ml`).
4. **Expert system.** `InferenceEngine.forward_chain` runs Task 2's rules against a derived-inputs
   dict built from the race state (`strategy_service._run_expert_system`), independently of the ML
   step above — it does not consume the ML prediction as an input.
5. **Search.** `astar_search` plans the remaining laps as a fresh `RaceProblem`
   (`strategy_service._run_search`), also independent of the ML step.
6. **DL prediction.** `strategy_service._run_dl` reuses the ML stage's already-built feature rows
   and calls `xai.live.predict_point` per target — a point-prediction-only path (no SHAP) that
   shares `xai.live`'s module-level cache with the XAI stage below, so a DL prediction never costs a
   second model load if an explanation is also requested. The pit target's `predicted_class` uses
   the same tuned threshold Task 7's own `/api/dl/predict/pit` endpoint uses (read from
   `DL_METRICS_JSON`), never a hard-coded 0.5.
7. **Recommendation assembly.** `strategy_service._combine_recommendation` is the actual combining
   rule: an expert-system `pit_decision` conclusion wins outright if one fired (rules encode domain
   knowledge — safety car, mandatory-stop windows — that the statistical models don't have);
   otherwise the ML and DL pit-probability predicted classes are compared — agreement is used
   directly, disagreement sets `disagreement: true` and falls back to ML (the better holdout
   precision on this dataset's one real stop) while still surfacing both numbers. A "pit" verdict is
   refined into `"PIT NOW"` or `"PIT IN N LAPS"` from the A* plan's first `PIT` action index.
   `recommended_action` (the pre-existing, simpler field) is computed separately and kept for
   backward compatibility with existing callers.
8. **XAI (opt-in).** Only when `explain=True` (always forced true for `/report`): `_explain` calls
   `xai.live.explain_feature_row` per target (§2.4), computing a live SHAP explanation, trust score
   and narrative for *this specific* feature row.
9. **Report rendering (report endpoint only).** `strategy_report.render_markdown` (or `render_html`
   for `?format=html` — the two share one content path: `render_html` converts `render_markdown`'s
   output rather than re-deriving it, so the two formats cannot say different things) takes the
   completed analysis dict and formats it into the briefing described in the
   [user manual](user_manual.md#reading-a-downloaded-strategy-report); it performs no computation of
   its own.

## 5. Error handling and honesty patterns

Two patterns recur across every layer and are worth naming as design decisions rather than
incidental behaviour:

- **404/503, never a fabricated 200.** Every `GET` handler that reads a generated artifact
  (`_read_json` helpers in `ml.py`, `dl.py`; `_results`/`_target` in `xai.py`; `_missing` in
  `reasoning.py`) returns HTTP 404 with the exact command that would generate the missing file if
  it is absent. Every prediction handler that needs a trained model catches
  `ModelUnavailableError`/`ExplainerUnavailableError` and returns HTTP 503 with the same kind of
  actionable message. No handler in the codebase returns a 200 response containing a placeholder,
  zero, or interpolated value in place of a genuinely missing artifact.
- **Approximation and extrapolation are reported, not hidden.** `feature_approximation.py`'s
  `approximated` and `out_of_range` lists, `strategy_service.py`'s `context_only` map (does the
  currently-trained model even use this field?), and `xai/live.py`'s reduced-budget `method` block
  are all examples of the same rule: whenever the system has to fill in, guess, or shortcut
  something, that fact becomes a structured, named field in the response rather than an
  undocumented implementation detail.

## 6. Deployment view

Two supported paths, both documented in full in [`deployment_guide.md`](deployment_guide.md):

**Process-based (primary local path, unchanged by Docker's addition):**

- The backend runs as a plain `uvicorn` ASGI process (`uvicorn app.api.main:app`), optionally with
  multiple workers since it is read-only at request time.
- The frontend runs as a Next.js process (`npm run dev` for development, `npm run build` +
  `npm run start` for production), configured at build time with `NEXT_PUBLIC_API_BASE_URL`.
- `./run.sh` is the supported single-command way to bring both up together on one machine, with
  port and CORS-origin resolution handled by the script itself.

**Container-based (`Dockerfile`, `frontend/Dockerfile`, `docker-compose.yml`, repo root):**

- Both images are code-only — `artifacts/` and `data/` are excluded via `.dockerignore` and instead
  bind-mounted at runtime (`docker-compose.yml`), since they are generated by
  `python scripts/build_all.py` and meant to be regenerated, not baked into an image layer.
- The frontend image bakes `NEXT_PUBLIC_API_BASE_URL` in at `next build` time (a Next.js
  constraint — `NEXT_PUBLIC_*` values are inlined into the client bundle) via a Docker build arg,
  mirroring how `run.sh` passes the same variable to `next dev`.
- `docker-compose.yml`'s backend service has a healthcheck against the extended `GET /api/health`
  (see §4's response shape), and the frontend's `depends_on` waits on that healthcheck rather than
  just the backend's port opening — the backend imports the full ML/DL/XAI/QML stack at process
  start, which is slower than the port becoming reachable.
- `scripts/record_environment.py` writes `artifacts/deployment/environment.txt` (Python/Node
  versions, full `pip freeze`) from whichever environment runs it — intended to be run once per
  build/deployment, not committed as a static claim.

No database, message queue, or external service is part of either deployment path — the only
persistent state is the `data/` and `artifacts/` directory trees on local disk (or the host, via the
bind mounts, in the container case).

## 7. Known design limitations — expert-system rule coverage

10 of the Task 2 rule base's 32 rules are structurally unreachable through the live
`/api/strategy/predict` pipeline, each for a specific, investigated reason (see
`strategy_service.py`'s "Rule-base input coverage" module comment and
`scripts/evaluate_system.py`'s `rule_base_reachability_check`, whose output is
`artifacts/evaluation/domain_validation.json`'s `rule_base_reachability_check` key):

| Rule(s) | Needs | Why it isn't supplied |
|---|---|---|
| R-PIT-002, R-PIT-003, R-ERS-001, R-TAC-001/002/003 | `gap_ahead`, `gap_behind`, `drs_enabled`, `undercut_threat`, `overcut_opportunity` | Rival-car state. `RaceStateRequest` is a single-car snapshot; this project has no rival-car telemetry anywhere. (The feature contract's `gap_roll3_mean`/`gap_expanding` mean "gap to the field median pace" — a different quantity; reusing them here would be a fabrication, not a proxy.) |
| R-STRAT-001 | `circuit`, `grid_position` | This system only ever runs on one circuit (2023 Bahrain GP, fixed by the training data), and `RaceStateRequest` has no starting-grid field. |
| R-STRAT-002, R-STRAT-003 | `overtaking_difficulty`, `pit_loss` | Circuit characteristics in no dataset this system reads; asserting a value would be an opinion presented as a fact. |
| R-DEG-003 | `graining_risk` | No tyre-graining model exists anywhere in this codebase to ground a value on. |

These are not bugs to be silently patched with an invented number — they are places where the rule
base was authored for a richer live-telemetry system than this project actually builds. Closing them
for real would mean adding rival-car position/gap data and multi-circuit characteristics to the data
pipeline, which is out of scope for Task 9/10.
