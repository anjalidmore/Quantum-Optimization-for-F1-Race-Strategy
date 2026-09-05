# Phase 3 · Tasks 7 & 8 — Deep Learning and Explainable AI

Part of **Quantum Optimization for Formula 1 Race Strategy**.

> **Scope:** classical deep learning and post-hoc explainability. No quantum computing.
> This document describes **what this project actually built**. For the underlying
> concepts and tools independently of this codebase, see
> [`STUDY_GUIDE_TASK7_TASK8.md`](STUDY_GUIDE_TASK7_TASK8.md).

Every claim below is traceable to a real file under `artifacts/`. Where a result is poor,
it is written down as it came out.

---

# Task 7 — Deep Learning Model Development

## Why this task was implemented

Task 6 established classical baselines for the two race-state predictions this platform
makes: expected lap time and pit-decision probability. Task 7 asks the obvious follow-up
question and answers it with evidence rather than assumption: **does a deep neural network
improve on those baselines on this data?**

That question matters for three reasons specific to this project:

1. **The project's research direction** treats each task as a rung — represent, reason,
   search, learn, go deeper, explain. Skipping Task 7 would leave the "go deeper" rung
   asserted rather than tested.
2. **Task 9's unified strategy engine** needs to know which model to serve. That choice
   should be made on measured holdout performance, not on which technique sounds more
   advanced.
3. **Task 8 needs a second model family to explain.** A trust score that compares two
   independently-trained families is only possible if a second family exists.

The honest framing carried throughout: *"improved predictive capability" is a hypothesis
to test, not a conclusion to assert.*

## Input

| File | Role |
|---|---|
| `data/processed/f1_features_selected.csv` | The Task 5 feature matrix — **reused, never regenerated** |
| `data/processed/feature_metadata.json` | The Task 5 contract: selected features per target, leakage exclusions, scaling policy, validation strategy |
| `data/processed/data_source.json` | Provenance marker — reported in every Task 7 artifact rather than assumed |
| `artifacts/metrics/regression_metrics.json` | Task 6's committed holdout metrics, used as the classical side of the comparison |
| `artifacts/metrics/classification_metrics.json` | ditto |
| `artifacts/metadata/model_registry.json` | Identifies Task 6's selected-best model per target |

Task 7 **engineers no new features**. It consumes exactly what Task 5 exported, so any
difference between a deep and a classical score is attributable to the model.

## What was done (method)

### In plain language

Two small neural networks were built — one predicting lap time, one predicting the
probability of pitting next lap. Each was trained many times over with different settings
(how wide, how much regularisation, how fast to learn), and each of those settings was
scored on race laps the network had not seen, always **later** laps than it trained on.
The best setting was retrained once on all the development laps, then tested a single
time on a block of final laps that nothing had touched. The classical models from Task 6
were then placed beside it on that identical test block.

### Technically

**Architectures** (`app/intelligence/dl/models.py`)

```
LAP TIME (regression)                   PIT DECISION (classification)
Input(n_features)                       Input(n_features)
Dense(h1, relu, L2=1e-4)                Dense(h1, relu, L2=1e-4)
Dropout(p)                              Dropout(p)
Dense(h2, relu, L2=1e-4)                Dense(h2, relu, L2=1e-4)
Dropout(p)                              Dropout(p)
Dense(1, LINEAR)                        Dense(1, SIGMOID)
loss = MSE, optimizer = Adam            loss = binary crossentropy, Adam
                                        class_weight = balanced
```

`(h1, h2)` and `p` are chosen by the search, from `{(32,16), (64,32)}` for regression and
`{(16,8), (32,16)}` for classification. On the committed run it picked **(32, 16)** with
dropout 0.1 for lap time and **(16, 8)** with dropout 0.2 for the pit decision — so the
search chose the *smaller* option in both cases, which is what the small-data rationale
predicts.

The classifier's search space is smaller still: its target has far fewer positive examples, and
extra capacity there fits noise rather than signal.

**Validation** (`app/intelligence/ml/splits.py` — *imported, not copied*)

The last 20% of laps are reserved as an untouched chronological holdout. The remainder is
cut into expanding-window lap-forward folds: fold *i* trains on every lap up to block
*i−1* and validates on block *i*. Whole laps never straddle a fold boundary. This is the
Task 5 contract's explicit requirement — *"Do not use random K-fold — this is a
time-ordered panel"* — and Task 7 satisfies it by importing Task 6's module rather than
writing its own.

**Preprocessing** (`app/intelligence/dl/training.py`)

Feature standardisation is fitted on the **training rows of each fold only** and applied
to the validation rows, per the contract's *"Fit the scaler inside each CV fold"*. Binary
indicator columns are passed through unscaled.

The regression target is **also** standardised, fitted on training rows only and inverted
before any metric is computed. This is not optional: `target_laptime` has a mean near
91 s and a standard deviation near 0.56 s, and a linear output head initialised near zero
cannot climb to 91 under MSE with L2 decay in any reasonable epoch budget. The first run
of this pipeline produced an MAE of **29 seconds** and an R² of **−5949** for exactly that
reason. It is recorded here rather than quietly fixed.

**Hyperparameter search** (`app/intelligence/dl/tuning.py`)

A **fully enumerated** grid — not sampled — over hidden units, dropout, learning rate and
batch size, scored on the expanding-window folds. Selection is by mean CV MAE for
regression and mean CV ROC-AUC for classification: the same primary metrics Task 6
selects on.

**Overfitting prevention**, all three applied together and all three reported:

| Mechanism | Setting |
|---|---|
| Dropout | on every hidden layer, rate chosen by the search |
| L2 weight decay | `1e-4` on every Dense kernel |
| Early stopping | `val_loss`, patience 20, `restore_best_weights=True` |

The evaluation report states, per target, how many epochs of validation-loss
deterioration early stopping actually discarded — so the mechanism is shown working
rather than merely configured.

## Output

Per target: a trained network producing a lap time in seconds (regression) or a pit
probability in [0, 1] (classification), together with the fitted scalers and the exact
feature order it expects. A network's weights are meaningless without both.

## Deliverables produced

| Deliverable | Path |
|---|---|
| Trained models | `artifacts/models/dl/target_laptime.keras`, `artifacts/models/dl/target_pit_next_lap.keras` |
| Fitted scalers | `artifacts/models/dl/target_*_scaler.joblib`, `target_laptime_target_scaler.joblib` |
| Feature specs | `artifacts/models/dl/target_*_spec.json` |
| Metrics | `artifacts/metrics/dl_metrics.json` |
| Training history | `artifacts/metrics/dl_training_history.json` |
| DL-vs-classical comparison | `artifacts/metrics/dl_vs_classical.json` |
| Loss / metric curves | `artifacts/figures/dl_target_*_training_history.png` |
| Comparison charts | `artifacts/figures/dl_target_*_model_comparison.png` |
| Evaluation report | `artifacts/reports/dl_evaluation_report.md` |
| Hyperparameter report | `artifacts/reports/dl_hyperparameter_report.md` |
| Registry entries | `artifacts/metadata/model_registry.json` (extended, **not** duplicated) |
| API | `GET /api/dl/{models,metrics,comparison,history,artifacts}`, `POST /api/dl/predict/{laptime,pit}` |
| Dashboard | `/deep-learning` |

### A documented divergence from the reference task spec

The reference table names a **`.h5`** deliverable. Under Keras 3 the HDF5 path is legacy:
a model saves to `.h5` but **fails to load back** with
`Could not deserialize 'keras.metrics.mse'` — verified on this installation. The native
`.keras` archive round-trips correctly, so that is what ships. A model that cannot be
reloaded is not a deliverable. Separately, TensorFlow publishes **no wheel for Python
3.14** (this project's interpreter), so Keras 3 runs on the PyTorch backend; the Keras API
in `app/intelligence/dl/` is identical either way.

The divergence is surfaced in three places rather than left to be discovered: the spec
JSON, the registry entry's `format_note`, and the `/api/dl/artifacts` response.

## Results as measured (2023 Bahrain GP, real FastF1 data — 815 training rows, 180 test rows)

### `target_laptime` — the deep network wins, clearly

| Model | Test MAE (s) | Test RMSE (s) | Test R² |
|---|---:|---:|---:|
| **`dnn_mlp`** | **0.5154** | **0.8895** | **+0.4786** |
| svr | 0.7815 | 1.0290 | +0.3023 |
| linear_regression | 0.7901 | 1.1414 | +0.1415 |
| decision_tree *(Task 6's selected best)* | 0.8673 | 1.3308 | **−0.1669** |
| random_forest | 0.9253 | 1.2671 | −0.0579 |
| xgboost | 1.0858 | 1.4412 | −0.3686 |

34% lower MAE than the best classical model. More importantly, **R² is +0.48 where Task
6's selected regressor is −0.17** — the network is the first model in this project to
generalise to the closing laps rather than doing worse than predicting the mean there.
Chosen configuration: hidden `(32, 16)`, dropout 0.1, lr 1e-3, batch 32. 2,017 parameters
against 815 training rows (ratio 2.47).

### `target_pit_next_lap` — the classical model wins, and the network is degenerate

| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| logistic_regression | **0.9832** | — | — | — | — | — |
| random_forest *(Task 6's selected best)* | **0.9832** | — | — | — | — | — |
| xgboost | 0.9777 | — | — | — | — | — |
| **`dnn_mlp`** | 0.9218 | **0.0667** | **0.0000** | **0.0000** | **0.0000** | 0.9944 |
| decision_tree | 0.8492 | — | — | — | — | — |
| svm | 0.8045 | — | — | — | — | — |

Two things must be read together here. The network loses on ROC-AUC (0.9218 vs 0.9832) —
that alone would be an ordinary "classical wins" result. But its **F1, precision and
recall are all exactly 0.0** while its accuracy is 0.9944: at the default 0.5 threshold it
predicts *never pit* for every test lap and rides the class imbalance. A 99.4% accuracy
that never fires is the textbook illustration of why accuracy is the wrong headline for a
rare event, and it is reported here rather than quoted as a success. 289 parameters
against 815 rows (ratio 0.35) — this network is small, and the ceiling is the data, not
the capacity.

**Neither result was tuned toward.** Selection used cross-validated MAE and ROC-AUC
respectively, decided before the holdout was touched.

## Significance — what these deliverables mean

**The loss curves** (`dl_target_*_training_history.png`) are the primary diagnostic. Read
them as: training and validation loss both falling → still learning; both flat →
plateaued; training falling while validation rises → **overfitting despite dropout**. The
marked epoch is the one early stopping restored, so the gap between it and the final epoch
is how much deterioration the countermeasures caught.

**The parameters-to-training-rows ratio** in the evaluation report is the single number
that frames everything else. Above 1, the network has more parameters than examples and
*could* memorise the training set outright; only regularisation stops it. That is also the
honest reason to expect a tree ensemble to be competitive.

**The comparison table** is only meaningful because both sides were produced by the same
code. `app/intelligence/dl/` imports `app.intelligence.ml.splits` and
`app.intelligence.ml.evaluation` directly, and the classical rows are read from Task 6's
own committed `artifacts/metrics/*.json` — the same numbers `/api/ml/metrics` serves.

**The hyperparameter report** lists every combination evaluated, not just the winner. A
grid where the winner barely beats its neighbours means the choice was not very
consequential; a grid with a clear peak means it was.

## How to interpret the results

**What a "good" MAE looks like here.** Lap times in this data vary by well under a second
around their mean. An MAE near or below that spread means the model is tracking real
lap-to-lap variation; an MAE well above it means the model is doing little better than
predicting the average lap.

**What R² near zero or negative means, specifically.** R² ≤ 0 means the model does worse
than always predicting the training mean. It is not a bug and it is not impossible: it is
what extrapolation failure looks like. The holdout here is the closing laps of a single
race, where fuel load and tyre state differ substantially from the laps the model trained
on. Read a negative holdout R² as *"this model does not extrapolate to late-race
conditions"*, not as *"the code is broken"*. Note also that model **selection** uses
cross-validated MAE, where the model is positive — so the procedure is sound even where
the holdout number is poor. Task 6 reports the same phenomenon for its own regressor, and
Task 7 does not pretend to be exempt from it.

**Why ROC-AUC alone is misleading for the pit target.** Pit events are a small minority of
laps. A model can achieve a high ROC-AUC — meaning it *ranks* laps well — while being
useless as a decision rule at the default 0.5 threshold, because precision collapses. Read
PR-AUC and F1 alongside ROC-AUC, always. Where a metric is mathematically undefined on a
split (a holdout containing no pit events at all), the artifacts record `null` with a
reason rather than a plausible-looking number.

**When to trust the DNN over the classical model, and when not to.** Trust the network
where it wins on the **primary CV metric** *and* its holdout number agrees — that pattern
means the advantage is real rather than a lucky test block. Prefer the classical model
when the two disagree, when the network's parameter-to-row ratio is high, or when the
target has few positive examples. And whichever wins, Task 8's trust score is the
per-prediction arbiter: two families agreeing is evidence, one disagreeing means at least
one is wrong and you cannot tell which from the prediction alone.

---

# Task 8 — Explainable AI

## Why this task was implemented

A model that outperforms another but cannot say **why** it decided something is close to
useless on a pit wall. A race engineer has seconds to accept or reject a recommendation,
and needs to know which factors drove it and whether the recommendation is solid enough to
act on.

There is a second, sharper motivation specific to this project. Task 5's real-data feature
selection **retained one-hot driver and team dummies** — a large share of the regression
feature set encodes *who is driving* rather than *what the car is doing*. A model leaning
on those cannot generalise to an unseen driver, silently conflates car performance with
driver pace, and would give two cars in an identical race state different strategy calls
purely because of the name on the car. That risk is real here, and the only way to know
whether it has materialised is to measure it.

Task 8 also closes the project's intellectual arc. The symbolic engines (Tasks 1–3) were
explainable **by construction** — the expert system emits HOW/WHY traces, and A\*'s path is
its own justification. The statistical engines (Tasks 6–7) have to be made explainable
**after the fact**, which is a strictly harder problem. Task 8 is where the two halves are
held to the same standard.

## Input

| Input | Role |
|---|---|
| `artifacts/models/laptime/*.joblib`, `artifacts/models/pit_decision/*.joblib` | Task 6's **persisted, selected-best** pipelines, loaded through `ModelCache` — the exact models the API serves |
| `artifacts/models/dl/*.keras` (+ scalers, + specs) | Task 7's saved networks |
| `data/processed/f1_features_selected.csv` | The same feature matrix, split with the same `chronological_holdout` |
| `data/processed/feature_metadata.json` | Identifies which features are driver/team identity dummies |

Task 8 **trains nothing**. If either model family is missing it raises
`ExplainerUnavailableError` rather than fitting a stand-in, so an explanation is always an
explanation *of the deployed model*.

## What was done (method)

### Seven techniques, each chosen for a stated reason

| Step | Method | Why this one | Module |
|---|---|---|---|
| Global importance | **Permutation importance** | The only method that means the same thing for a tree ensemble and a neural network, so the two columns are comparable. A tree's Gini importance and a network's gradients are not the same quantity. | `importance.py` |
| Attribution (classical) | **SHAP `TreeExplainer`** | *Exact* Shapley values for tree ensembles in polynomial time — free accuracy. | `shap_analysis.py` |
| Attribution (deep) | **SHAP `KernelExplainer`** | Model-agnostic; needs only a `predict` function. `DeepExplainer`'s Keras 3 support targets the TensorFlow backend, which is unavailable on this Python version. | `shap_analysis.py` |
| Local surrogate | **LIME** | An independent second account of the same prediction. Its disagreement with SHAP is itself a signal, and the trust score uses it. | `lime_analysis.py` |
| Counterfactual | **Single-feature bisection scan** | Answers the question an engineer actually asks — *"how many more laps on these tyres before the call changes?"* — with one actionable number. | `counterfactual.py` |
| Counterfactual | **DiCE (random search)** | Diverse whole-row alternatives, for when several routes to a different call exist. | `counterfactual.py` |
| Fairness | **Identity-attribution share** | Directly measures the risk above: identity's share of total attribution versus what an even spread would give. | `fairness.py` |

Explanations are computed on **representative test rows**, chosen and labelled by what was
actually found: for the classifier, the lowest pit probability, the row closest to the 0.5
boundary, and the highest pit probability. Labels are deliberately descriptive rather than
aspirational — on a holdout where the model never predicts above 0.5, calling its
highest-probability row "clear pit now" would misrepresent it. Rows are de-duplicated, so
the same row is never explained twice under two names.

### The trust score (`app/intelligence/xai/trust.py`)

```
trust = 0.40 · confidence
      + 0.30 · model_agreement
      + 0.30 · explanation_stability
```

| Component | Definition | The failure mode it catches |
|---|---|---|
| `confidence` | Classification: `2·\|p − 0.5\|`. Regression: `1 − min(1, \|p_dnn − p_classical\| / σ_target)` | A prediction sitting on the decision boundary is unusable however well explained. |
| `model_agreement` | `1 − \|p_dnn − p_classical\|` | Two families trained on identical folds agreeing is real evidence. One disagreeing means at least one is wrong and you cannot tell which. |
| `explanation_stability` | Jaccard overlap of SHAP's and LIME's top-3 features | If two established methods disagree about *why*, the explanation you would show the engineer is untrustworthy even when the prediction is right. |

The **components** are principled — each captures a distinct, real failure mode. The
**weights** are a documented judgement, not a derivation: confidence carries the largest
share because boundary predictions are useless regardless of explanation quality, and
agreement and stability are weighted equally because a wrong-but-explained and a
right-but-unexplainable prediction are both unsafe. They are exposed as
`app.intelligence.xai.trust.WEIGHTS` precisely so they can be challenged.

### The strategic explanation (`app/intelligence/xai/narrative.py`)

SHAP output is translated into one or two sentences a race engineer can read mid-race —
`"Recommend PITTING — model confidence 82%. Tyre age (18) is the dominant factor pushing
towards a stop; …  Trust in this recommendation: HIGH."` Feature names map to plain
English through an explicit glossary; any feature missing from it falls back to a readable
form of its own name rather than being silently dropped, so a new Task 5 feature can never
vanish from an explanation. A probability of 0.0004 renders as `<1%`, not `0%` — rounding
it away would overstate the model's certainty in exactly the direction a reader should not
be misled.

## Output

Per prediction: a ranked list of named factors with signed contributions, an independent
LIME account of the same prediction, what would have to change to flip the call, a trust
score with its three components, and a plain-English sentence. Per model: a global
importance ranking, a global SHAP ranking, and a fairness measurement.

## Deliverables produced

| Deliverable | Path |
|---|---|
| SHAP report | `artifacts/reports/xai_shap_report.md` |
| SHAP figures | `artifacts/figures/xai_*_shap_summary.png`, `xai_*_shap_waterfall.png` |
| LIME report | `artifacts/reports/xai_lime_report.md` |
| LIME figures | `artifacts/figures/xai_*_lime.png` |
| Counterfactual report | `artifacts/reports/xai_counterfactual_report.md` |
| Counterfactual figures | `artifacts/figures/xai_*_counterfactual.png` |
| Trust score report | `artifacts/reports/xai_trust_score_report.md` |
| Fairness report | `artifacts/reports/xai_fairness_report.md` + `xai_*_fairness.png` |
| **Explainability dashboard** | `artifacts/reports/xai_explainability_dashboard.md` |
| Importance comparison | `artifacts/figures/xai_*_importance_comparison.png` |
| Machine-readable results | `artifacts/metadata/xai_results.json` |
| API | `GET /api/xai/{summary,feature-importance,shap,lime,counterfactual,trust-score,fairness,explanation}` |
| Dashboard | `/explainability` |

## Significance — what these deliverables mean

**The SHAP summary plot** shows which features move predictions most across the whole test
set, and in which direction. A feature high in that ranking is one the model relies on; if
it is a feature that *should not* matter, that is a finding about the model, not about
racing.

**A waterfall plot** decomposes one prediction: the base value is what the model predicts
on average, and each bar is one feature's push away from it. The bars sum to the gap
between base and prediction — that summation is SHAP's local-accuracy guarantee, and it is
what LIME does **not** provide.

**LIME's `local_r2`** is the diagnostic most easily overlooked. It reports how well a
straight line reproduced the real model near that row. A low value means the linear
explanation is a poor stand-in and should be discounted **however confident it looks**.

**The counterfactual result** is actionable when reachable and informative when not. "Not
reachable" — no value of tyre age anywhere in its observed range flips the call with the
rest of the race state fixed — means the decision is driven by feature *combinations*
rather than that feature alone. Reported with the range searched, that is a genuine
finding, not a failed computation.

**The fairness concentration ratio** is the number to read first. Above 1, identity
features are doing more work than their share of the feature count, and the model is
partly predicting *who* rather than *what* — expect it not to transfer to an unseen
driver. Below 1, race-state features dominate, which is the desired outcome. Because a
large share of the regression feature set is identity dummies, the *expected* share under
a uniform spread is itself high — which is exactly why the ratio, not the raw percentage,
is the meaningful quantity.

**The trust score** should be read as a structured summary of three real signals, not as a
calibrated probability. The bands are deliberately conservative: a score in the middle
should send the engineer to the evidence rather than to the recommendation.

## How to interpret the results

**Compare SHAP values by rank and sign, not magnitude, across model families.** The
forest's are exact; the network's are sampled by `KernelExplainer` with a recorded
`nsamples` budget and carry sampling noise. Both reports say so in their own text.

**When SHAP and LIME disagree, believe neither uncritically.** Their disagreement means
the local decision surface is not well approximated by a line. The trust score already
penalises this, and the `explanation_stability` component is where you see it.

**A high ROC-AUC with a low PR-AUC and F1 is a ranking model, not a decision rule.** It
orders laps by pit risk correctly while being unusable at the default threshold. The fix
is threshold tuning on validation folds — not yet applied, and named as such.

**SHAP explains what the model does, not what is true.** A model that learned a spurious
correlation gets a faithful explanation *of that spurious correlation*. That limitation is
precisely why the fairness assessment exists as a separate check rather than being
inferred from the attribution ranking alone.

**Two operational limits, both reported rather than smoothed over.** DiCE's random search
is bounded to a 60-second budget: with pit events at roughly 5% of laps, an early run
sampled for 15 minutes and returned nothing, so it is now queried from the row closest to
the decision boundary and restricted to observed feature ranges. And where the
counterfactual scan finds no reachable flip, the report says so with the range searched.

---

## Reproducing all of this

```bash
python scripts/build_all.py            # builds Tasks 1-8; skips stages whose artifacts exist
python scripts/build_all.py --force    # regenerates everything
python scripts/build_all.py --skip-dl  # Tasks 1-6 only
pytest tests/test_dl_training.py tests/test_xai_explanations.py tests/test_dl_xai_api.py
```

Tasks 7 and 8 depend on Task 6: Task 7 compares against its committed results, and Task 8
explains its persisted pipelines. `build_all.py` refuses to run them if Task 6 was skipped
and has no existing artifacts, rather than producing a comparison against nothing.
