# F1 Race Strategy Intelligence

A computational-intelligence project built around one question: **when should a Formula 1 driver pit, and what will the next lap cost?**

We take a real Grand Prix session, clean it, engineer features from it, and then attack the same question three ways — with symbolic reasoning (an ontology, a rule base, a state-space search), with machine learning (classical models and neural networks), and with explainability tools that say *why* a prediction came out the way it did. A FastAPI backend serves the results and a small Next.js dashboard displays them.

**The data is one real race:** the 2023 Bahrain Grand Prix, fetched with [FastF1](https://docs.fastf1.dev/) — 1,055 laps from 20 drivers, of which 995 are usable after the warm-up trim. Every number in this README and on the dashboard comes from a file in `artifacts/`. Nothing is typed in by hand.

![Python](https://img.shields.io/badge/Python-3.14-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-backend-009688)
![Next.js](https://img.shields.io/badge/Next.js-15-black)
![Tests](https://img.shields.io/badge/tests-261%20passing-brightgreen)

| Dashboard | Race Strategy Simulator |
|---|---|
| ![Dashboard](docs/screenshots/dashboard.png) | ![Strategy Simulator](docs/screenshots/strategy-simulator.png) |

(More in [`docs/screenshots/`](docs/screenshots/). The Machine Learning and Data screenshots were taken before those pages were merged into `/models` and `/data`, so their layout is older than the code.)

## What we built

The lab specification is ten tasks. Eight are implemented, and they are one application rather than eight folders:

| Task | What it does | Where it lives |
|---|---|---|
| 1 · Knowledge representation | An F1 ontology and knowledge graph — 61 entities, 29 relationships | `app/intelligence/knowledge_representation/` |
| 2 · Expert system | 32 forward/backward-chaining rules with explanations for each firing | `app/intelligence/expert_system/` |
| 3 · State-space search | BFS / DFS / UCS / Greedy / A\* over pit strategies; A\* and UCS agree at 2262.42 s | `app/intelligence/search/` |
| 4 · Data engineering & EDA | Cleaning with a full audit trail, plus nine analyses and their figures | `app/intelligence/data/` |
| 5 · Feature engineering | 45 candidate features → a 4-stage selection funnel → the feature contract | `app/intelligence/features/build.py` |
| 6 · Machine learning | 10 classical models, time-aware validation, tuned decision thresholds | `app/intelligence/ml/` |
| 7 · Deep learning | Two Keras networks on the same folds and the same holdout | `app/intelligence/dl/` |
| 8 · Explainable AI | SHAP, LIME, counterfactuals, a trust score, per-group performance | `app/intelligence/xai/` |
| 9 · Quantum ML | Three PennyLane models simulated on the same split, with fair classical baselines | `app/intelligence/qml/` |
| 10 | Formal responsible-AI evaluation — not started | — |

Two predictions run through everything: **lap time** (regression) and **does this driver pit at the end of this lap?** (classification).

### The rules we held ourselves to

- **No leakage.** Sector times sum to the lap time, so they are excluded along with speed traps and `IsPersonalBest`. A test fails the build if any of them reaches a training matrix.
- **No random splits.** This is a time-ordered panel, so a shuffled K-fold would train on lap 50 and validate on lap 10. Every model uses expanding-window, lap-forward folds with a chronological holdout.
- **Scalers are fit inside each fold**, never on the whole dataset.
- **No made-up numbers.** Every value on the dashboard is read from a generated artifact. When something cannot be computed, the UI and the reports say "undefined" and why.

## Running it

You need Python 3.12+ and Node 20+.

```bash
./run.sh                    # sets up, builds what's missing, starts both servers, opens the browser
```

Or by hand:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

python scripts/build_all.py          # builds Tasks 1-9 (skips stages that already have artifacts)
python scripts/build_all.py --force  # rebuilds everything from the raw data (~15 min; Task 7 is most of it)

uvicorn app.api.main:app --reload    # backend on :8000
cd frontend && npm install && npm run dev   # dashboard on :3000
```

Single stages, when you only want one:

```bash
python scripts/build_features.py     # Task 5 only
python -c "from app.intelligence.dl import pipeline; pipeline.train_all()"      # Task 7
python -c "from app.intelligence.xai import pipeline; pipeline.run_all()"       # Task 8
python scripts/run_qml.py            # quantum models (~30 s; skip in build_all with --skip-qml)
```

To train on a different race:

```bash
python scripts/fetch_real_session.py --year 2023 --event Monza --session R
python scripts/build_all.py --force
```

**Why Next.js and FastAPI instead of the lab's suggested Streamlit:** we wanted the models served behind a real HTTP API that something other than the UI could call — the same endpoints back the dashboard, the tests and `scripts/demo_predict.py` — and a separate frontend made it obvious when a page was inventing a number instead of fetching one.

## What's in the repo

```
app/
  intelligence/     tasks 1-8, one package per task
  services/         model cache, feature approximation, the strategy service that chains ML + rules + search
  api/              FastAPI routers (ml, dl, xai, strategy, data, tasks)
  core/             paths and runtime setup
frontend/           Next.js dashboard: 5 pages, every value fetched from the API
scripts/            build_all.py and one script per buildable stage
tests/              261 tests
artifacts/          every generated model, metric, figure and report
data/               raw session data and the Task 5 feature contract
docs/               architecture, per-task notes, the Task 5 notebook walkthrough
```

`scripts/build_all.py` is the entry point: it runs each task in order, skips a stage whose artifacts already exist, and refuses to run Tasks 7–8 if Task 6 has no results to compare against.

## Results

Both model families are scored on the **same 180 held-out laps** (laps 47–57), which no model saw during training or selection.

| | Task 6 best classical | Task 7 neural network |
|---|---|---|
| Lap time | `svr` — MAE **0.781 s**, R² 0.302 | **MAE 0.565 s, R² 0.456** (16,257 parameters) |
| Pit decision | `random_forest` — PR-AUC 0.25, caught the one real stop | CV PR-AUC 0.482, but missed that stop (289 parameters) |

### Quantum models (simulated)

Three PennyLane models run on the noiseless `default.qubit` simulator — a variational classifier, a variational
regressor and a fidelity-kernel SVM — on the same laps, the same folds and the same metric code. Fitting 45
features into 4 qubits means PCA-reducing the inputs (on training rows only), so the fair comparison is against
classical models with a similar parameter count on those same reduced inputs.

| Cross-validated | Quantum | Parameter-matched classical | Task 6 (all features) |
|---|---|---|---|
| Pit decision, PR-AUC | **0.304 ± 0.191** (VQC, 14 params) | 0.205 (logistic regression, 5 params) | 0.386 (random forest) |
| Lap time, MAE (s) | 1.746 (VQR, 14 params) | 1.624 (linear regression, 5 params) | **1.381** (SVR) |

The VQC edges out both parameter-matched classical models on the pit decision; the VQR does not beat a linear
model on lap time; neither comes near Task 6 with all features. **This is a noiseless simulation of tiny
circuits on one race and demonstrates nothing about quantum advantage** — and the fold spreads overlap, so it
does not cleanly separate these models either. Quantum computing explained from zero, for this
circuit: [`docs/QML_README.md`](docs/QML_README.md). Results discussion:
[`artifacts/reports/classical_vs_quantum_report.md`](artifacts/reports/classical_vs_quantum_report.md).

The lap-time network is the best model in the project. The pit-decision result is genuinely inconclusive, and the reason matters more than the number: **the test laps contain exactly one labelled pit stop.** Precision, recall and F1 on one event are noise, so we lean on the cross-validated figures and say so everywhere the number appears.

---

# Task 7 — Deep Learning Model Development

### Objective

Build neural networks for the two existing prediction problems and test — rather than assume — whether they beat Task 6's classical models on the same data.

### F1 Dataset

Task 5's feature matrix, `data/processed/f1_features_selected.csv`: 995 driver-laps from the 2023 Bahrain GP, no synthetic data, no missing values. Task 5's leakage exclusions stay excluded.

### Prediction Target

| Target | Task | Distribution |
|---|---|---|
| `target_laptime` | regression, seconds | mean 99.10 s, sd 2.01 s, range 94.00–103.98 s |
| `target_pit_next_lap` | binary | **48 of 995 laps (4.8%)** — heavily imbalanced |

The top of the lap-time range is Task 4's outlier cap: 79 laps (60 of them in- or out-laps) sit at exactly 103.985 s. One real stop is unlabelled — see [Known limitations](#known-limitations).

### Feature Engineering

None added. Task 7 consumes Task 5's selected features unchanged: 45 inputs for lap time (17 race-state features and 28 driver/team one-hots), 8 for the pit decision. `StandardScaler` is fitted on training laps only; the lap-time target is standardised too and every prediction converted back to seconds before any metric.

### DNN Architecture

| Lap time (45 → 1) | Pit decision (8 → 1) |
|---|---|
| Dense 128 → 64 → 32 → 1, ReLU, linear head | Dense 16 → 8 → 1, ReLU, sigmoid head |
| dropout 0.2, L2 1e-3, RMSprop, MSE | dropout 0.3, L2 1e-4, RMSprop, binary cross-entropy |
| **16,257 parameters, all trainable** | **289 parameters, all trainable** |

Keras 3 on the PyTorch backend (TensorFlow has no wheel for Python 3.14).

### Hyperparameters

One factor at a time over the cross-validation folds: architecture, learning rate (1e-3 / 5e-4 / 1e-4), batch size (16 / 32 / 64), dropout (0.2–0.5), optimizer (Adam / RMSprop), L2, loss, and class weighting. Selection used **only** the folds; the test laps were untouched until the end. Every experiment is in `artifacts/deep_learning/hyperparameter_report.csv`.

Class weighting was investigated rather than applied by default — and lost, on validation: unweighted CV PR-AUC 0.421 vs 0.361 weighted.

### Training

EarlyStopping on validation loss (patience 20, best weights restored), dropout and L2 together, since both networks have more parameters than training rows.

### Time-Aware Validation

Four expanding-window lap-forward folds for selection, then a final fit that trains on laps 4–38 and early-stops on laps 39–46 — later laps the network has never seen. Test = laps 47–57, used once.

### Overfitting Prevention

Dropout, L2, early stopping, and a verdict computed from the curves rather than asserted. Lap time: **OVERFITTING** (large train/validation gap; validation loss rose 41% after the restored epoch). Pit decision: **REASONABLE FIT**. Curves: `artifacts/deep_learning/*/loss_curve.png`.

### Evaluation

| Split | Lap time MAE / R² | Pit F1 / ROC-AUC / PR-AUC |
|---|---|---|
| Train | 0.522 / 0.782 | 0.214 / 0.680 / 0.184 |
| Validation | 1.500 / 0.248 | 0.286 / 0.968 / 0.606 |
| **Test** | **0.565 / 0.456** | **0.000 / 0.777 / 0.024** |

Decision threshold 0.1524, tuned on pooled out-of-fold predictions (F1 0 → 0.458 there), not left at 0.5. Confusion matrix on the test laps: `[[179, 0], [1, 0]]` — the model called no pit anywhere, including on the one real stop.

### Comparison with Existing ML Model

On cross-validation the network leads on both targets (MAE 0.953 vs 1.190 for the best classical model; PR-AUC 0.482 vs 0.386). On the test laps it wins clearly for lap time and loses the single pit event to the random forest. Full tables: `artifacts/deep_learning/model_comparison.csv`.

### Results

- **Lap time: the network is the best model in the project**, on both CV and the holdout.
- **Pit decision: inconclusive.** One event cannot separate two models.
- Three defects found and fixed while building this, each now pinned by a test: `fit(class_weight=...)` silently did nothing on this Keras build (weights moved into the loss), the final fit was early-stopping on its own training rows, and `/api/dl/predict/pit` decided at 0.5 instead of the tuned threshold.

### Saved Artifacts

`artifacts/models/deep_learning/{laptime,pit_decision}/f1_dnn_model.h5` plus scalers and `model_spec.json` — reloaded after saving and required to reproduce the trained predictions exactly. Public reports, curves and comparisons live in `artifacts/deep_learning/`. Weights are **not** served over HTTP.

### How to Run

```bash
python -c "from app.intelligence.dl import pipeline; pipeline.train_all()"   # ~10 min on a laptop CPU
```

Results are served at `GET /api/dl/{models,metrics,comparison,history,artifacts}` and shown on the dashboard's **Models** page.

---

# Task 8 — Explainable Artificial Intelligence (XAI)

### Objective

Explain *why* the predictions come out as they do, say how far each explanation can be trusted, and show where the model is weakest — without inventing anything the data does not contain.

### Model Explained

The trained Task 7 network, reloaded from disk and never retrained. Task 6's selected model is a second opinion only, for the trust score's agreement term and an importance comparison.

### Prediction Task

The same two targets on the same 180 test laps. Scenarios are chosen by rule, not by hand: fastest / median / slowest predicted lap, freshest and oldest tyres, the clearest non-pit lap, the lap closest to the decision threshold, and **the one lap where a driver actually pitted**.

### Feature Importance

Permutation importance on the test laps (10 repeats), in `artifacts/xai/feature_importance.png`. Both targets are led by `tyre_life` and its interaction with track temperature. For lap time the network and Task 6's SVR rank the same top three in the same order.

### SHAP Analysis

`KernelExplainer` (model-agnostic; 200 samples over a 25-point k-means background) for all 180 laps — sampled approximations, so ranks and signs are meaningful and magnitudes carry noise. Top features: `field_median_lag1`, `tracktemp_dev_x_tyrelife`, `stint_number` (lap time); `tracktemp_dev_x_tyrelife`, `tyre_life`, `tyrelife_x_soft` (pit). Reports: `artifacts/xai/SHAP_Report.md`.

**Driver and team identity carry 3.8% of lap-time attribution**, against 62.2% if attribution were spread evenly across the 45 inputs — so the model is mostly reading race state, not who is driving. The pit model has no identity inputs at all.

### LIME Analysis

2,000 perturbations per scenario with a local linear surrogate. Mean SHAP–LIME top-3 overlap is 0.30 (lap time) and 0.55 (pit). **LIME ranks `track_status` first on all 180 laps, but that feature is 1 (green flag) on every one of them** — it varies only in training, so LIME is describing a yellow flag that never happened. Its surrogates also fit poorly (local R² 0.19–0.44). SHAP describes what drove the prediction here; LIME describes what *might* move it.

### Counterfactual Analysis

"What tyre age would change this call?", swept over the training range (1–22 laps) in half-lap steps, with the derived features (`tyrelife_x_soft`, `tyrelife_x_medium`, `tracktemp_dev_x_tyrelife`) recomputed so every row is a lap that could exist. Compound and set freshness are held fixed — changing tyre age cannot change the compound.

No tyre age flips any pit call: the closest lap peaks at P = 0.1364 at 16.5 laps, just under the 0.1524 threshold. For lap time the effects are small (< 0.1 s for ±5 laps) except on fresh tyres, where **the model predicts that older tyres are faster** — an out-lap confound, not tyre physics, and exactly the kind of thing this scan exists to surface. Output: `artifacts/xai/counterfactual_analysis.csv`.

### Trust Score

A **project-defined** score, not a validated one:

```
trust = Σ wₖ·componentₖ / Σ wₖ   over the components that apply
  0.35 confidence            distance from the tuned threshold (not used for lap time — no decision point)
  0.25 model_agreement       1 − |DNN − Task 6|
  0.20 explanation_stability SHAP vs LIME top-3 overlap
  0.20 input_validity        share of inputs inside the training 1st–99th percentile
HIGH ≥ 0.75 · MODERATE ≥ 0.50 · LOW ≥ 0.25 · DO NOT ACT < 0.25
```

Means over the test laps: 0.632 (lap time), 0.712 (pit). All 360 rows are in `artifacts/xai/trust_score_report.csv`.

**We checked it against outcomes, and it does not predict error:** Spearman correlation between trust and lap-time error is −0.09 (p = 0.24). Read the components to see *which* signal is weak; do not read the total as a probability of being right.

### F1-Specific Performance Stratification

Error by **driver, team and tyre compound** — the groupings this dataset actually has. There are no demographic attributes and none were invented, and with one circuit there is no circuit stratum. Groups under 30 laps or 5 pit laps are flagged descriptive-only. Output: `artifacts/xai/fairness_assessment.csv`.

The one adequately-sized split is compound: **HARD 0.388 s MAE (99 laps) vs SOFT 0.782 s (81 laps)**, the soft-tyre laps predicted too fast. Per-driver MAE ranges from 0.235 s to 2.316 s, but the worst drivers are the ones whose laps include a pit stop, so that spread is about pit laps rather than drivers.

### Explainability Dashboard

The **Explainability** page reads only from `artifacts/xai/`: global importance, the stratification table, a card per scenario with its plain-English sentence, SHAP and LIME factors, trust components and counterfactual — plus a **lap inspector** for any of the 180 test laps, whose tyre-age counterfactual is computed live by the saved network.

Generated sentence for the missed stop:

> Recommend STAYING OUT - model confidence 6%. Track temperature acting on tyre age (-27.73) is a major factor pushing against stopping; tyre age (25) is a major factor pushing towards a stop; tyre age on the soft compound (25) is a minor factor pushing towards a stop. Trust in this recommendation: MODERATE.

### Results

- The model runs on race state, not identity (3.8% vs 62.2% expected).
- **Why the one real stop was missed:** on ZHO's lap 54 the tyres were 25 laps old — beyond the 1–22 laps seen in training — and the cool-track × tyre-age term (SHAP −0.049) cancelled tyre age (+0.046), leaving P = 0.063, below the 0.076 base rate. Task 6's random forest, which does not extrapolate a trend, gave 0.579.
- One non-physical behaviour found (older tyres predicted faster on out-laps), one unreliable explainer identified (LIME's constant feature), and one honest negative on our own trust score.

### Limitations

One race, 180 test laps, one labelled pit stop. SHAP values are sampled, LIME surrogates fit poorly, the counterfactual varies a single feature within the training range, permutation importance for the pit model is scored on one positive lap, and the trust score is an unvalidated heuristic. Stratification groups are 9–22 laps outside the compound split.

### How to Run

```bash
python -c "from app.intelligence.xai import pipeline; pipeline.run_all()"   # ~2 min
```

Everything lands in `artifacts/xai/` (`shap/`, `lime/`, `counterfactual/`, `importance/`, `stratification/`, four CSVs, six Markdown reports, `xai_metadata.json`).

---

## Testing

```bash
pytest                     # 261 tests
ruff check .               # lint
cd frontend && npm run lint && npm run build
```

The suite writes to a temporary directory, so running it never modifies the committed artifacts. The tests that matter most are the ones that would catch us cheating: no leakage column reaches a training matrix, no fold trains on a later lap, the committed trust scores recompute from their own recorded inputs, and every artifact path the API serves exists on disk.

## Known limitations

- **One session.** Everything is trained on a single Grand Prix. The results are honest for that race and should not be read as a general F1 model.
- **One labelled pit stop in the holdout**, so pit-decision test metrics are illustrative. There is also a **second, unlabelled stop**: FastF1 has no stint data for NOR, Task 4 filled it with the median, so NOR's real stop on lap 47 is labelled "no pit". Both models rank that lap low, so no conclusion changes — but the fix belongs in Task 4.
- **The lap-time network overfits** (train MAE 0.52 vs validation 1.50), which the report states rather than hides.
- **The trust score does not track error** on this holdout.
- **One transitive `postcss` advisory** remains inside Next's own bundle; clearing it needs Next 16, which requires an ESLint 9 flat-config migration.

## What we learned

- **A pipeline can look correct and be silently wrong.** The simulator's driver and team dropdowns had no effect on predictions for a while, because the feature builder recognised only the synthetic dataset's column names. Now every feature is classified against the live contract, and a test asserts that changing a dropdown changes a feature value.
- **A library can ignore you.** `model.fit(class_weight=...)` had no effect on this Keras/PyTorch build. We only found it by measuring the loss by hand, and the fix was to put the weights inside the loss function.
- **The evaluation design explains most "bad" results.** A negative R², an uninterpretable classifier and CV disagreeing with the holdout were not three bugs; they were one consequence of holding out the tail of a single race.
- **Explaining a model finds bugs in it.** The counterfactual scan is what revealed that the network thinks older tyres are faster just after a stop.

## Documentation

| Document | What it answers |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | Module layout, the feature contract, leakage prevention, end-to-end flow |
| [`docs/task7_task8.md`](docs/task7_task8.md) | Tasks 7–8: what we built, and the concepts behind deep learning and XAI |
| [`docs/QML_README.md`](docs/QML_README.md) | Quantum ML from zero: qubits, our circuit, how it trains, our results, viva prep |
| [`docs/DELIVERABLES_CHECKLIST.md`](docs/DELIVERABLES_CHECKLIST.md) | Every lab deliverable, its file path, and whether it is complete |
| [`docs/TESTING_REPORT.md`](docs/TESTING_REPORT.md) | The last clean build and test run, with the actual output |
| [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) | Install, build, run, serve, retrain, troubleshoot |
| [`docs/task1_knowledge_representation.md`](docs/task1_knowledge_representation.md) … [`task4_data_engineering.md`](docs/task4_data_engineering.md) | One note per early task |
| [`docs/notebooks/task5_feature_engineering.ipynb`](docs/notebooks/task5_feature_engineering.ipynb) | Task 5 walkthrough over the functions in `app/intelligence/features/build.py` |
| `artifacts/**/*.md` | The generated reports — every table in them was computed, not written |

## License

MIT — see [LICENSE](LICENSE).
