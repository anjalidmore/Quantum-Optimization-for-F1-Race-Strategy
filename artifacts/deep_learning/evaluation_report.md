# Task 7 — Deep Learning Evaluation Report

_Generated 2026-09-27 17:55 UTC._

Every network below was evaluated **once** on the chronological holdout — the last 20% of laps — after its hyperparameters, decision threshold and early-stopping epoch had been fixed on earlier laps.

## target_laptime — regression

Input features: 45 · rows: train 675, validation 140, test 180 · test laps 47–57

### Architecture

| Layer | Type | Units | Activation | Dropout |
|---|---|---:|---|---:|
| hidden_1 | Dense | 128 | relu |  |
| dropout_1 | Dropout |  |  | 0.2 |
| hidden_2 | Dense | 64 | relu |  |
| dropout_2 | Dropout |  |  | 0.2 |
| hidden_3 | Dense | 32 | relu |  |
| dropout_3 | Dropout |  |  | 0.2 |
| laptime_seconds | Dense | 1 | linear |  |

Total parameters 16,257 · trainable 16,257 · optimizer RMSprop · training loss `mse`.

Hyperparameters: `{'hidden_units': [128, 64, 32], 'dropout': 0.2, 'learning_rate': 0.001, 'batch_size': 16, 'optimizer': 'rmsprop', 'l2': 0.001, 'loss': 'mse'}`

### Train / validation / test

| Split | MAE (s) | RMSE (s) | R² |
|---|---:|---:|---:|
| Train | 0.5222 | 0.8643 | 0.7818 |
| Validation | 1.5001 | 2.3752 | 0.2482 |
| **Test** | 0.5651 | 0.9083 | 0.4564 |

### Overfitting

Verdict: **OVERFITTING (large train/validation gap at the saved weights)**. 35 epochs run; early stopping restored epoch 10. Validation loss minimum 1.7906 → final 2.5178 (+40.6%); training loss at the best epoch 0.4692 → final 0.2709. Overfitting emerged after the best epoch: **yes**.

### Comparison with Task 6 (same test laps)

| Model | MAE | RMSE | R2 |
|---|---:|---:|---:|
| dnn_mlp **(DNN)** | 0.5651 | 0.9083 | 0.4564 |
| linear_regression | 0.7901 | 1.1414 | 0.1415 |
| decision_tree | 0.8673 | 1.3308 | -0.1669 |
| random_forest | 0.9253 | 1.2671 | -0.0579 |
| svr (Task 6 selected) | 0.7815 | 1.0290 | 0.3023 |
| xgboost | 1.0858 | 1.4412 | -0.3686 |

**The deep network wins on mae** (0.5651), against 5 of Task 6's classical models evaluated on the same chronological holdout.

Primary comparison metric: MAE.

## target_pit_next_lap — classification

Input features: 8 · rows: train 675, validation 140, test 180 · test laps 47–57

### Architecture

| Layer | Type | Units | Activation | Dropout |
|---|---|---:|---|---:|
| hidden_1 | Dense | 16 | relu |  |
| dropout_1 | Dropout |  |  | 0.3 |
| hidden_2 | Dense | 8 | relu |  |
| dropout_2 | Dropout |  |  | 0.3 |
| pit_probability | Dense | 1 | sigmoid |  |

Total parameters 289 · trainable 289 · optimizer RMSprop · training loss `binary_crossentropy`.

Hyperparameters: `{'hidden_units': [16, 8], 'dropout': 0.3, 'learning_rate': 0.001, 'batch_size': 32, 'optimizer': 'rmsprop', 'l2': 0.0001, 'class_weighted': False}`

### Train / validation / test

Decision threshold 0.1524 (tuned on out-of-fold CV predictions, never on the test laps).

| Split | Pit laps | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 41/675 | 0.8370 | 0.1515 | 0.3659 | 0.2143 | 0.6804 | 0.1840 |
| Validation | 6/140 | 0.9643 | 1.0000 | 0.1667 | 0.2857 | 0.9677 | 0.6060 |
| **Test** | 1/180 | 0.9944 | 0.0000 | 0.0000 | 0.0000 | 0.7765 | 0.0244 |

Test confusion matrix (rows actual, columns predicted; 0 = stay out, 1 = pit): `[[179, 0], [1, 0]]`.

> **Caution.** The chronological test laps contain **1 pit event(s)**. Precision, recall, F1 and PR-AUC on so few positives are dominated by chance; the cross-validated figures in `hyperparameter_report.csv` rest on many more pit laps and are the better guide to this model's ranking ability.

### Overfitting

Verdict: **REASONABLE FIT**. 35 epochs run; early stopping restored epoch 10. Validation loss minimum 0.1262 → final 0.1639 (+29.9%); training loss at the best epoch 0.2546 → final 0.1882. Overfitting emerged after the best epoch: **yes**.

### Comparison with Task 6 (same test laps)

| Model | PRECISION | RECALL | F1 | ROC_AUC | PR_AUC |
|---|---:|---:|---:|---:|---:|
| dnn_mlp **(DNN)** | 0.0000 | 0.0000 | 0.0000 | 0.7765 | 0.0244 |
| logistic_regression | 0.0108 | 1.0000 | 0.0213 | 0.9832 | 0.2500 |
| decision_tree | 0.0141 | 1.0000 | 0.0278 | 0.9693 | 0.0833 |
| random_forest (Task 6 selected) | 0.0435 | 1.0000 | 0.0833 | 0.9832 | 0.2500 |
| svm | 0.0000 | 0.0000 | 0.0000 | 0.7989 | 0.0270 |
| xgboost | 0.0312 | 1.0000 | 0.0606 | 0.9274 | 0.0714 |

**Task 6's classical `logistic_regression` wins on pr_auc** (0.2500 vs the deep network's 0.0244). Reported as measured. With 675 training rows from a single race session, a tree ensemble's inductive bias suits this problem better than a network's; deep learning's advantage requires substantially more data than exists here.

Primary comparison metric: PR_AUC.

