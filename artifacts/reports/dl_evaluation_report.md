# Task 7 - Deep Learning Evaluation Report

_Generated 2026-09-05 01:37 UTC._

Deep neural networks for F1 race-state prediction, compared against **Task 6's own
committed holdout results** - the same numbers the Machine Learning dashboard shows,
read from `artifacts/metrics/`. The comparison is like-for-like by construction:
identical feature matrix, identical split code (`app.intelligence.ml.splits`),
identical metric code (`app.intelligence.ml.evaluation`).

## target_laptime

**Task type:** regression  |  **Input features:** 45  |  **Training rows:** 815  |  **Test rows:** 180

**Dataset source:** `real_fastf1` (2023 Bahrain R)

### Network architecture

`laptime_mlp` - 2,017 trainable parameters, optimizer Adam, loss `mse`.

| Layer | Type | Detail |
|---|---|---|
| hidden_1 | Dense | 32 units, relu |
| dropout_1 | Dropout | rate 0.1 |
| hidden_2 | Dense | 16 units, relu |
| dropout_2 | Dropout | rate 0.1 |
| laptime_seconds | Dense | 1 units, linear |

**Parameters-to-training-rows ratio:** 2.47

With 2,017 parameters against 815 training rows (ratio 2.47), this network has more parameters than training examples. That is the expected regime for this dataset and it is why dropout, L2 and early stopping are applied together; it is also the honest reason to expect a tree ensemble to be competitive here.

### Overfitting prevention

| Mechanism | Setting | Effect observed |
|---|---|---|
| Dropout | 0.1 on every hidden layer | see loss curve |
| L2 weight decay | 1e-04 on every Dense kernel | see loss curve |
| Early stopping | patience 20 on `val_loss`, best weights restored | stopped at epoch 200 of 200 run (cap 200) |

Early stopping restored the weights from epoch 200. Validation loss was still improving when the epoch cap was reached, so the model was capacity- or budget-limited rather than overfitting.

### Test-set comparison - DL vs classical

| Model | MAE (s) | RMSE (s) | R2 | MAPE (%) |
|---|---:|---:|---:|---:|
| dnn_mlp **<-- deep network** | 0.5154 | 0.8895 | 0.4786 | 0.52 |
| linear_regression | 0.7901 | 1.1414 | 0.1415 | 0.80 |
| decision_tree | 0.8673 | 1.3308 | -0.1669 | 0.88 |
| random_forest | 0.9253 | 1.2671 | -0.0579 | 0.94 |
| svr | 0.7815 | 1.0290 | 0.3023 | 0.80 |
| xgboost | 1.0858 | 1.4412 | -0.3686 | 1.11 |

### Verdict

**The deep network wins on mae** (0.5154), against 5 of Task 6's classical models evaluated on the same chronological holdout.

---

## target_pit_next_lap

**Task type:** classification  |  **Input features:** 8  |  **Training rows:** 815  |  **Test rows:** 180

**Dataset source:** `real_fastf1` (2023 Bahrain R)

### Network architecture

`pit_decision_mlp` - 289 trainable parameters, optimizer Adam, loss `binary_crossentropy`.

| Layer | Type | Detail |
|---|---|---|
| hidden_1 | Dense | 16 units, relu |
| dropout_1 | Dropout | rate 0.2 |
| hidden_2 | Dense | 8 units, relu |
| dropout_2 | Dropout | rate 0.2 |
| pit_probability | Dense | 1 units, sigmoid |

**Parameters-to-training-rows ratio:** 0.35

With 289 parameters against 815 training rows (ratio 0.35), this network has fewer parameters than training examples, which is the intended small-data design.

### Overfitting prevention

| Mechanism | Setting | Effect observed |
|---|---|---|
| Dropout | 0.2 on every hidden layer | see loss curve |
| L2 weight decay | 1e-04 on every Dense kernel | see loss curve |
| Early stopping | patience 20 on `val_loss`, best weights restored | stopped at epoch 192 of 200 run (cap 200) |

Early stopping restored the weights from epoch 192. Training ran 200 epochs, so 8 epochs of validation-loss deterioration were discarded - the countermeasures did real work.

### Test-set comparison - DL vs classical

| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| dnn_mlp **<-- deep network** | 0.9218 | 0.0667 | 0.0000 | 0.0000 | 0.0000 | 0.9944 |
| logistic_regression | 0.9832 | 0.2500 | 0.0182 | 0.0092 | 1.0000 | 0.4000 |
| decision_tree | 0.8492 | 0.0182 | 0.0179 | 0.0090 | 1.0000 | 0.3889 |
| random_forest | 0.9832 | 0.2500 | 0.0769 | 0.0400 | 1.0000 | 0.8667 |
| svm | 0.8045 | 0.0278 | 0.0000 | 0.0000 | 0.0000 | 0.9944 |
| xgboost | 0.9777 | 0.2000 | 0.0000 | 0.0000 | 0.0000 | 0.9944 |

### Verdict

**Task 6's classical `logistic_regression` wins on roc_auc** (0.9832 vs the deep network's 0.9218). Reported as measured. With 815 training rows from a single race session, a tree ensemble's inductive bias suits this problem better than a network's; deep learning's advantage requires substantially more data than exists here.

---

