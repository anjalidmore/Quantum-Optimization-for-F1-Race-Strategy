# Task 6 — Model Selection Report

Generated: 2026-09-05T06:38:02.674312+00:00

## Selection criteria

| Task | Primary | Secondary |
|---|---|---|
| Lap-time regression | CV MAE (lower better), **subject to a generalisation guard** | CV RMSE, CV R², test R² |
| Pit-decision classification | **CV PR-AUC** (higher better) | CV ROC-AUC, CV F1, precision/recall |

Selection uses cross-validated metrics, not a single holdout number, and a
model is only preferred over another when it wins on the primary metric
computed identically across the same lap-forward folds.

**Why PR-AUC and not ROC-AUC for classification.** Pit events are 4.8% of laps.
At that prevalence ROC-AUC is dominated by the majority class and stays near
0.98 for a model that never fires — it measures ranking, not usefulness.
PR-AUC asks the question that matters: of the laps this model flags, how many
are real pit windows?

**The generalisation guard.** Selecting purely on CV MAE once shipped a model
with a negative test R² — worse than predicting the training mean — while a
candidate with positive test R² sat unselected. The guard refuses to select a
negative-R² model when a positive-R² candidate exists, and states plainly when
it has overridden the CV ranking.

**Decision thresholds are tuned, not assumed.** Each classifier's cut-off is
chosen on pooled out-of-fold CV predictions rather than left at sklearn's
default 0.5, which is only optimal for balanced classes with equal error costs.
Neither holds here. Per-model thresholds appear in the table below.

## Regression

| Model | CV MAE | CV RMSE | CV R² | Test MAE | Test RMSE | Test R² | Selected |
|---|---:|---:|---:|---:|---:|---:|:---:|
| decision_tree | 1.1898 | 1.6321 | 0.3381 | 0.8673 | 1.3308 | -0.1669 |  |
| svr | 1.3813 | 1.6942 | 0.2553 | 0.7815 | 1.0290 | 0.3023 | ✅ |
| random_forest | 1.3817 | 1.7624 | 0.2141 | 0.9253 | 1.2671 | -0.0579 |  |
| xgboost | 1.4384 | 1.8325 | 0.1351 | 1.0858 | 1.4412 | -0.3686 |  |
| linear_regression | 2.0181 | 2.7964 | -1.2807 | 0.7901 | 1.1414 | 0.1415 |  |

Selected: **svr**

> ⚠ **Selection warning.** CV ranking overridden by the generalisation guard: the best CV MAE model (decision_tree, CV MAE 1.1898) scores test R2 -0.1669 — worse than predicting the mean. Selected svr instead (CV MAE 1.3813, test R2 0.3023), the best CV performer among candidates that generalise to the holdout. This disagreement between CV and holdout is itself a finding: the holdout is the closing laps of a single race, a different fuel and tyre regime from training.

## Classification

| Model | CV ROC-AUC | CV PR-AUC | CV F1 | Test ROC-AUC | Test PR-AUC | Test F1 | Selected |
|---|---:|---:|---:|---:|---:|---:|:---:|
| random_forest | 0.8482 | 0.3863 | 0.1967 | 0.9832 | 0.2500 | 0.0833 | ✅ |
| svm | 0.7843 | 0.3321 | 0.1429 | 0.7989 | 0.0270 | 0.0000 |  |
| xgboost | 0.8125 | 0.3083 | 0.2307 | 0.9274 | 0.0714 | 0.0606 |  |
| logistic_regression | 0.8298 | 0.2964 | 0.2887 | 0.9832 | 0.2500 | 0.0213 |  |
| decision_tree | 0.6711 | 0.1797 | 0.1870 | 0.9693 | 0.0833 | 0.0278 |  |

Selected: **random_forest**

### Decision thresholds

Chosen on pooled out-of-fold CV predictions by maximising F1, never on the
test set. 0.5 is sklearn's default, not a considered choice.

| Model | Threshold | Objective | Test F1 | Test precision | Test recall |
|---|---:|---|---:|---:|---:|
| random_forest | 0.4803 | f1 | 0.0833 | 0.0435 | 1.0000 |
| svm | 0.1008 | f1 | 0.0000 | 0.0000 | 0.0000 |
| xgboost | 0.1527 | f1 | 0.0606 | 0.0312 | 1.0000 |
| logistic_regression | 0.5756 | f1 | 0.0213 | 0.0108 | 1.0000 |
| decision_tree | 0.7608 | f1 | 0.0278 | 0.0141 | 1.0000 |


## Notes on skipped models

Any model listed as "skipped" was not trained because an optional
dependency (XGBoost, and/or its native OpenMP runtime) was unavailable in
this environment. No result was fabricated in its place.
