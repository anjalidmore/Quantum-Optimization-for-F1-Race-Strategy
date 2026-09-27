# Task 7 — Hyperparameter Report

_Generated 2026-09-27 17:55 UTC._

Selection used **only** the expanding-window lap-forward folds over the development laps. The chronological test laps were not touched until the final evaluation.

Search method: one factor at a time. Each stage varies one hyperparameter while the others hold the best values so far; every value listed in the specification is tried and every experiment is recorded (the full table is `hyperparameter_report.csv`).

## target_laptime (regression)

Selection metric: **CV MAE** (lower is better), mean of 4 folds. 13 distinct configurations trained.

| Exp | Varied | Layers | LR | Batch | Dropout | Optimizer | L2 | Loss | CV MAE (± sd) | Selected |
|---|---|---|---:|---:|---:|---|---:|---|---:|:---:|
| E01 | architecture (layers / neurons) | [128, 64, 32] | 0.001 | 32 | 0.3 | adam | 0.0001 | mse | 1.1525 (± 0.4858) |  |
| E02 | architecture (layers / neurons) | [64, 32] | 0.001 | 32 | 0.3 | adam | 0.0001 | mse | 1.3703 (± 0.7084) |  |
| E03 | architecture (layers / neurons) | [32, 16] | 0.001 | 32 | 0.3 | adam | 0.0001 | mse | 1.3440 (± 0.4506) |  |
| E04 | learning rate | [128, 64, 32] | 0.0005 | 32 | 0.3 | adam | 0.0001 | mse | 1.1911 (± 0.4821) |  |
| E05 | learning rate | [128, 64, 32] | 0.0001 | 32 | 0.3 | adam | 0.0001 | mse | 1.5916 (± 0.9741) |  |
| E06 | batch size | [128, 64, 32] | 0.001 | 16 | 0.3 | adam | 0.0001 | mse | 1.1505 (± 0.4705) |  |
| E07 | batch size | [128, 64, 32] | 0.001 | 64 | 0.3 | adam | 0.0001 | mse | 1.2455 (± 0.5167) |  |
| E08 | dropout | [128, 64, 32] | 0.001 | 16 | 0.2 | adam | 0.0001 | mse | 1.1393 (± 0.4876) |  |
| E09 | dropout | [128, 64, 32] | 0.001 | 16 | 0.4 | adam | 0.0001 | mse | 1.2050 (± 0.4380) |  |
| E10 | dropout | [128, 64, 32] | 0.001 | 16 | 0.5 | adam | 0.0001 | mse | 1.3307 (± 0.4126) |  |
| E11 | optimizer | [128, 64, 32] | 0.001 | 16 | 0.2 | rmsprop | 0.0001 | mse | 1.0186 (± 0.3992) |  |
| E12 | L2 regularisation | [128, 64, 32] | 0.001 | 16 | 0.2 | rmsprop | 0.001 | mse | 0.9530 (± 0.4221) | ✅ |
| E13 | loss function | [128, 64, 32] | 0.001 | 16 | 0.2 | rmsprop | 0.001 | huber | 0.9771 (± 0.4374) |  |

**Selected:** `{'hidden_units': [128, 64, 32], 'dropout': 0.2, 'learning_rate': 0.001, 'batch_size': 16, 'optimizer': 'rmsprop', 'l2': 0.001, 'loss': 'mse'}`

Read the ± column before reading a winner into small differences: where two configurations differ by less than their fold-to-fold spread, the data cannot separate them.

## target_pit_next_lap (classification)

Selection metric: **CV PR_AUC** (higher is better), mean of 4 folds. 13 distinct configurations trained.

| Exp | Varied | Layers | LR | Batch | Dropout | Optimizer | L2 | Class weighting | CV PR_AUC (± sd) | Selected |
|---|---|---|---:|---:|---:|---|---:|---|---:|:---:|
| E01 | class weighting | [128, 64, 32] | 0.001 | 32 | 0.3 | adam | 0.0001 | none | 0.4214 (± 0.1387) |  |
| E02 | class weighting | [128, 64, 32] | 0.001 | 32 | 0.3 | adam | 0.0001 | balanced | 0.3608 (± 0.1657) |  |
| E03 | architecture (layers / neurons) | [32, 16] | 0.001 | 32 | 0.3 | adam | 0.0001 | none | 0.4373 (± 0.3473) |  |
| E04 | architecture (layers / neurons) | [16, 8] | 0.001 | 32 | 0.3 | adam | 0.0001 | none | 0.4516 (± 0.1324) |  |
| E05 | learning rate | [16, 8] | 0.0005 | 32 | 0.3 | adam | 0.0001 | none | 0.3639 (± 0.1050) |  |
| E06 | learning rate | [16, 8] | 0.0001 | 32 | 0.3 | adam | 0.0001 | none | 0.3334 (± 0.1316) |  |
| E07 | batch size | [16, 8] | 0.001 | 16 | 0.3 | adam | 0.0001 | none | 0.4311 (± 0.1072) |  |
| E08 | batch size | [16, 8] | 0.001 | 64 | 0.3 | adam | 0.0001 | none | 0.3758 (± 0.1137) |  |
| E09 | dropout | [16, 8] | 0.001 | 32 | 0.2 | adam | 0.0001 | none | 0.4356 (± 0.1160) |  |
| E10 | dropout | [16, 8] | 0.001 | 32 | 0.4 | adam | 0.0001 | none | 0.3678 (± 0.1068) |  |
| E11 | dropout | [16, 8] | 0.001 | 32 | 0.5 | adam | 0.0001 | none | 0.4299 (± 0.1191) |  |
| E12 | optimizer | [16, 8] | 0.001 | 32 | 0.3 | rmsprop | 0.0001 | none | 0.4817 (± 0.1152) | ✅ |
| E13 | L2 regularisation | [16, 8] | 0.001 | 32 | 0.3 | rmsprop | 0.001 | none | 0.4731 (± 0.1503) |  |

**Selected:** `{'hidden_units': [16, 8], 'dropout': 0.3, 'learning_rate': 0.001, 'batch_size': 32, 'optimizer': 'rmsprop', 'l2': 0.0001, 'class_weighted': False}`

Read the ± column before reading a winner into small differences: where two configurations differ by less than their fold-to-fold spread, the data cannot separate them.

