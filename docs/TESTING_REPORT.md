# Testing report

**Run date:** 2026-09-27 · **Machine:** macOS (Darwin 25.6.0), Apple silicon, Python 3.14.6 ·
**Commands:** `python scripts/build_all.py --force`, then `pytest`, `ruff check .`,
`npm run lint`, `npx tsc --noEmit`, `npm run build`.

Every figure below is copied from that run's output. Nothing here is a summary of an earlier run.

---

## 1. Summary

| Check | Result |
|---|---|
| Clean rebuild from raw data (`build_all.py --force`) | **passed** — all 10 stages, exit 0 |
| Python test suite (`pytest`) | **284 passed, 0 failed, 0 skipped** in 57.67 s |
| Python lint (`ruff check .`) | **All checks passed** |
| Frontend lint (`npm run lint`) | **No ESLint warnings or errors** |
| Frontend types (`npx tsc --noEmit`) | **clean** |
| Frontend production build (`npm run build`) | **passed** — 6 routes |
| Metric drift after the clean rebuild | **none** (see §4) |

## 2. The clean build

`python scripts/build_all.py --force` rebuilt every artifact from the raw session data. Stage
timings as reported by the build:

| Stage | Time |
|---|---|
| Environment validation | 1.1 s |
| Task 1 — Knowledge representation | 0.6 s |
| Task 2 — Expert system | 0.0 s |
| Task 3 — State-space search | 0.4 s |
| Task 4 — Data engineering & EDA | 1.2 s |
| Task 5 — Feature engineering | 9.4 s |
| Task 6 — Machine learning | 25.4 s |
| Task 7 — Deep learning | 723.8 s |
| Task 8 — Explainable AI | 103.2 s |
| Quantum machine learning | 36.6 s |
| Artifact consistency validation | 0.0 s |
| **Total** | **≈ 15 minutes**, of which Task 7's hyperparameter search is 80% |

The build then validated artifact consistency and exited 0.

## 3. The test suite

284 tests across 16 files. Counts are from `pytest --collect-only`:

| Test file | Tests | What it guards |
|---|---:|---|
| `tests/test_xai.py` | 38 | SHAP/LIME/counterfactual correctness, trust-score algebra, stratification, committed Task 8 artifacts |
| `tests/test_api.py` | 34 | every endpoint, race-state validation, CORS configurability, path traversal, the Task 9 reasoning endpoints and report generator |
| `tests/test_dl_xai_api.py` | 31 | Task 7/8 API surface, the live explainer's reduced budget, the lap inspector's SHAP additivity |
| `tests/test_dl_training.py` | 27 | leakage, time-aware splits, class weighting inside the loss, `.h5` reload verification, hyperparameter coverage |
| `tests/test_expert_system.py` | 21 | forward and backward chaining, conflict resolution, rule validation |
| `tests/test_search.py` | 18 | BFS/DFS/UCS/Greedy/A\*, admissibility, the A\*-equals-UCS optimality invariant |
| `tests/test_ml_threshold.py` | 17 | threshold tuning on out-of-fold predictions |
| `tests/test_data_engineering.py` | 17 | cleaning, imputation, outlier capping, data-quality scoring |
| `tests/test_qml.py` | 15 | encoder fitted on training rows only, angle clipping, circuit shapes, seed determinism, kernel validity, weight reload |
| `tests/test_knowledge_representation.py` | 15 | ontology and graph integrity |
| `tests/test_ml_data_contract.py` | 13 | the Task 5 → Task 6 contract, leakage exclusions, lap-forward splits |
| `tests/test_artifact_contract.py` | 10 | every artifact path the API serves exists; no absolute paths in the manifest |
| `tests/test_ml_training.py` | 8 | the Task 6 pipeline end to end, best-model selection and saving |
| `tests/test_strategy_service.py` | 7 | the ML + expert system + search chain |
| `tests/test_ml_splits.py` | 7 | chronological holdout and expanding-window folds |
| `tests/test_openmp_runtime.py` | 6 | the XGBoost/PyTorch OpenMP import-order guard (reproduces the real segfault) |

Raw result line:

```
284 passed in 57.67s
```

Slowest five, from `pytest --durations=5`:

```
24.89s setup    tests/test_ml_training.py::test_pipeline_selects_a_best_model_for_each_task
12.99s call     tests/test_openmp_runtime.py::test_every_dl_entry_point_goes_through_the_guard
3.76s call     tests/test_dl_training.py::test_class_weighting_inside_the_loss_moves_the_predictions
2.15s call     tests/test_openmp_runtime.py::test_guarded_import_order_lets_xgboost_and_keras_coexist
1.81s call     tests/test_openmp_runtime.py::test_kmp_duplicate_lib_ok_does_not_fix_it
```

The first entry is setup, not a single assertion: that test retrains Task 6 into a temporary
directory. The OpenMP tests are slow because each one launches a subprocess to check whether a
specific import order segfaults.

### Test hermeticity

`git status` was clean before and after the suite: the tests write to `tmp_path`, so running them
never modifies a committed artifact. This is asserted rather than assumed — `ArtifactPaths` exists
precisely so every pipeline's output can be redirected.

## 4. Reproducibility — does a rebuild change the results?

The clean `--force` rebuild regenerated every artifact. Comparing the regenerated JSON against the
committed copies, field by field and ignoring timestamps and wall-clock timings:

| Artifact | Result |
|---|---|
| `artifacts/deep_learning/evaluation_report.json` | **identical** |
| `artifacts/metrics/qml_metrics.json` | **identical** |
| `artifacts/xai/xai_metadata.json` | **identical** |
| `data/processed/feature_metadata.json` | **identical** |
| `artifacts/metrics/regression_metrics.json` | identical except 5 `fit_seconds` values |
| `artifacts/metrics/classification_metrics.json` | identical except 5 `fit_seconds` values |

The only differences anywhere are wall-clock measurements (`fit_seconds`), report timestamps, and
PNG bytes (matplotlib embeds a creation date). **No metric moved.** Every stochastic step is
seeded, and the model libraries are pinned to exact versions in `requirements.txt` so a new release
cannot silently shift the numbers.

## 5. Live checks against a running system

Performed against `uvicorn` on a local port with the dashboard pointed at it:

| Check | Expected | Result |
|---|---|---|
| `GET /api/health` | 200 | pass |
| `GET /api/ml/{models,comparison,artifacts,feature-importance}` | 200 | pass |
| `GET /api/dl/{models,metrics,comparison,history,artifacts}` | 200 | pass |
| `GET /api/xai/{summary,shap,lime,counterfactual,trust-score,fairness,explanation,stratification,laps,lap}` | 200 | pass |
| `GET /api/qml/{summary,training,search}` | 200 | pass |
| `GET /api/reasoning/{knowledge,expert-system,search}` | 200 | pass |
| `POST /api/strategy/predict` | 200 with a recommendation | pass |
| `POST /api/strategy/report` | 200, `text/markdown`, attachment | pass |
| `GET /artifacts/figures/qml_circuit.png` | 200 | pass |
| `GET /artifacts/models/qml/vqc_weights.npy` | **404** (weights must not be served) | pass |
| Dashboard pages `/`, `/strategy`, `/reasoning`, `/models`, `/explainability`, `/data` | 200, real values rendered | pass |
| `POST /api/strategy/predict` with an unknown driver | 422 | pass |

## 6. What the tests do not cover

Stated plainly, because a green suite is not the same as a verified system:

- **No test asserts that a prediction is correct**, only that it is produced honestly from the
  committed model. Accuracy is reported in the metrics artifacts, not enforced by a test.
- **The frontend has no automated tests.** Lint, type-check and build pass, and the pages were
  checked by hand against a live API; there are no component or end-to-end tests.
- **The pit classifier's holdout contains one labelled pit event**, so no test can meaningfully
  assert its precision or recall.
- **Quantum results are simulated only.** The tests verify determinism, shapes and kernel validity,
  not anything about real quantum hardware.
- **Single session.** Every test runs against one Grand Prix, so nothing here demonstrates
  generalisation to other races.

## 7. How to reproduce this report

```bash
python scripts/build_all.py --force          # ~15 min
pytest                                        # ~1 min
ruff check .
cd frontend && npm run lint && npx tsc --noEmit && npm run build
```
