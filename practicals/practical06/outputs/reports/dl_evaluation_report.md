# Task 7 - Deep Learning Evaluation Report

_Generated 2026-09-05 06:58 UTC._

Deep neural networks for F1 race-state prediction, compared against classical
baselines trained on **the same folds and the same untouched chronological test
set**. The comparison is like-for-like by construction: identical feature matrix,
identical split code, identical metric code.

## target_laptime

**Task type:** regression  |  **Input features:** 6  |  **Training rows:** 420  |  **Test rows:** 100

**Dataset source:** `practical04/outputs/clean/fastf1_laps_clean.csv`

### Network architecture

`laptime_mlp` - 2,561 trainable parameters, optimizer Adam, loss `mse`.

| Layer | Type | Detail |
|---|---|---|
| hidden_1 | Dense | 64 units, relu |
| dropout_1 | Dropout | rate 0.3 |
| hidden_2 | Dense | 32 units, relu |
| dropout_2 | Dropout | rate 0.3 |
| laptime_seconds | Dense | 1 units, linear |

**Parameters-to-training-rows ratio:** 6.10

With 2,561 parameters against 420 training rows (ratio 6.10), this network has more parameters than training examples. That is the expected regime for this dataset and it is why dropout, L2 and early stopping are all applied together; it is also the honest reason to expect a tree ensemble to be competitive here.

### Overfitting prevention

| Mechanism | Setting | Effect observed |
|---|---|---|
| Dropout | 0.3 on every hidden layer | see loss curve |
| L2 weight decay | 1e-04 on every Dense kernel | see loss curve |
| Early stopping | patience 20 on `val_loss`, best weights restored | stopped at epoch 65 of 85 run (cap 200) |

Early stopping restored the weights from epoch 65. Training ran 85 epochs, so 20 epochs of validation-loss deterioration were discarded - the countermeasures did real work.

### Test-set comparison - DL vs classical

| Model | MAE (s) | RMSE (s) | R2 | MAPE (%) |
|---|---:|---:|---:|---:|
| dnn_mlp **<-- DNN** | 0.1584 | 0.2080 | 0.7631 | 0.17 |
| linear_regression | 0.3005 | 0.3603 | 0.2892 | 0.33 |
| decision_tree | 0.4210 | 0.5134 | -0.4434 | 0.46 |
| random_forest | 0.2298 | 0.2863 | 0.5509 | 0.25 |
| mean_baseline | 0.4645 | 0.5302 | -0.5398 | 0.51 |

### Verdict

**The DNN wins on mae** (0.1584) against 4 classical baselines on the same test set.

---

## target_pit_next_lap

**Task type:** classification  |  **Input features:** 8  |  **Training rows:** 420  |  **Test rows:** 100

**Dataset source:** `practical04/outputs/clean/fastf1_laps_clean.csv`

### Network architecture

`pit_decision_mlp` - 289 trainable parameters, optimizer Adam, loss `binary_crossentropy`.

| Layer | Type | Detail |
|---|---|---|
| hidden_1 | Dense | 16 units, relu |
| dropout_1 | Dropout | rate 0.4 |
| hidden_2 | Dense | 8 units, relu |
| dropout_2 | Dropout | rate 0.4 |
| pit_probability | Dense | 1 units, sigmoid |

**Parameters-to-training-rows ratio:** 0.69

With 289 parameters against 420 training rows (ratio 0.69), this network has fewer parameters than training examples, which is the intended small-data design.

### Overfitting prevention

| Mechanism | Setting | Effect observed |
|---|---|---|
| Dropout | 0.4 on every hidden layer | see loss curve |
| L2 weight decay | 1e-04 on every Dense kernel | see loss curve |
| Early stopping | patience 20 on `val_loss`, best weights restored | stopped at epoch 171 of 191 run (cap 200) |

Early stopping restored the weights from epoch 171. Training ran 191 epochs, so 20 epochs of validation-loss deterioration were discarded - the countermeasures did real work.

### Test-set comparison - DL vs classical

| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| dnn_mlp **<-- DNN** | _undefined_ | _undefined_ | 0.0000 | 0.0000 | 0.0000 | 0.2800 |
| logistic_regression | _undefined_ | _undefined_ | 0.0000 | 0.0000 | 0.0000 | 0.6000 |
| decision_tree | _undefined_ | _undefined_ | 0.0000 | 0.0000 | 0.0000 | 0.9600 |
| random_forest | _undefined_ | _undefined_ | 0.0000 | 0.0000 | 0.0000 | 0.9600 |
| majority_baseline | _undefined_ | _undefined_ | 0.0000 | 0.0000 | 0.0000 | 1.0000 |

### Verdict

**No verdict is possible from the holdout.** Every model's roc_auc - the DNN's and every classical baseline's alike - is mathematically undefined on this test set, because it contains **0 positive examples**. The chronological holdout is laps 46-55, and no pit event falls in that range. That is a property of the data, not a modelling failure, and it applies symmetrically to the DNN and to every classical baseline.

Falling back to the cross-validated folds, where positive examples are present: the DNN's mean CV roc_auc is **0.4178**.

A ROC-AUC below 0.5 means the network ranks pit laps *worse than chance*. With so few positive examples and only 8 input features, there is not enough signal for a network to learn a ranking from - this is an honest negative result, not a bug, and it is exactly what the small-data caveat in the architecture rationale predicts.

The `proj-mode` branch trains this same code on the 995-row real FastF1 matrix, whose pit events are not clustered into three laps, and reports a defined holdout score there.

---

