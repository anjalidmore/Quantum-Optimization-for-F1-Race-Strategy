# Task 6 — Model Selection Report

Generated: 2026-09-05T01:46:15.654215+00:00

## Selection criteria

| Task | Primary | Secondary |
|---|---|---|
| Lap-time regression | CV MAE (lower better) | CV RMSE, CV R², train/test gap |
| Pit-decision classification | CV ROC-AUC (higher better) | CV PR-AUC, CV F1, precision/recall |

Selection uses cross-validated metrics, not a single holdout number, and a
model is only preferred over another when it wins on the primary metric
computed identically across the same lap-forward folds.

## Regression

| Model | CV MAE | CV RMSE | CV R² | Test MAE | Test RMSE | Test R² | Selected |
|---|---:|---:|---:|---:|---:|---:|:---:|
| decision_tree | 1.1898 | 1.6321 | 0.3381 | 0.8673 | 1.3308 | -0.1669 | ✅ |
| svr | 1.3813 | 1.6942 | 0.2553 | 0.7815 | 1.0290 | 0.3023 |  |
| random_forest | 1.3817 | 1.7624 | 0.2141 | 0.9253 | 1.2671 | -0.0579 |  |
| xgboost | 1.4384 | 1.8325 | 0.1351 | 1.0858 | 1.4412 | -0.3686 |  |
| linear_regression | 2.0181 | 2.7964 | -1.2807 | 0.7901 | 1.1414 | 0.1415 |  |

Selected: **decision_tree**

## Classification

| Model | CV ROC-AUC | CV PR-AUC | CV F1 | Test ROC-AUC | Test PR-AUC | Test F1 | Selected |
|---|---:|---:|---:|---:|---:|---:|:---:|
| random_forest | 0.8517 | 0.3809 | 0.2033 | 0.9832 | 0.2500 | 0.0769 | ✅ |
| logistic_regression | 0.8298 | 0.2964 | 0.2887 | 0.9832 | 0.2500 | 0.0182 |  |
| xgboost | 0.8231 | 0.3205 | 0.1389 | 0.9777 | 0.2000 | 0.0000 |  |
| svm | 0.7854 | 0.3316 | 0.3100 | 0.8045 | 0.0278 | 0.0000 |  |
| decision_tree | 0.7073 | 0.1511 | 0.2466 | 0.8492 | 0.0182 | 0.0179 |  |

Selected: **random_forest**

## Notes on skipped models

Any model listed as "skipped" was not trained because an optional
dependency (XGBoost, and/or its native OpenMP runtime) was unavailable in
this environment. No result was fabricated in its place.
