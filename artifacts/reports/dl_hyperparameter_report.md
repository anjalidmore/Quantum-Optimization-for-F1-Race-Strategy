# Task 7 - Hyperparameter Report

_Generated 2026-09-05 01:37 UTC._

Every combination below was **fully enumerated**, not sampled, and each was
scored on the same expanding-window lap-forward folds Task 6 uses. A random
K-fold search would let the network validate on laps it had already seen the
future of; the Task 5 contract forbids it explicitly.

## target_laptime

**Task:** regression  |  **Features:** 45  |  **Combinations evaluated:** 8  |  **Folds per combination:** 4  |  **Total training runs:** 32

### Search space

| Hyperparameter | Values explored |
|---|---|
| Hidden units | [[32, 16], [64, 32]] |
| Dropout | [0.1, 0.3] |
| Learning rate | [0.001, 0.0003] |
| Batch size | [32] |
| Max epochs | 200 (early stopping, patience 20) |

**Selection metric:** mae (lower is better), averaged across folds.

### All trials

| Hidden units | Dropout | LR | Batch | CV mae (mean) | CV mae (std) | Mean best epoch |
|---|---|---|---|---|---|---|
| [32, 16] | 0.1 | 0.001 | 32 | 1.3007 | 0.4870 | 59.8 |
| [32, 16] | 0.1 | 0.0003 | 32 | 1.6203 | 0.9245 | 96.2 |
| [32, 16] | 0.3 | 0.001 | 32 | 1.3440 | 0.4506 | 55.8 |
| [32, 16] | 0.3 | 0.0003 | 32 | 1.6285 | 0.7480 | 116.8 |
| [64, 32] | 0.1 | 0.001 | 32 | 1.3924 | 0.7776 | 28.5 |
| [64, 32] | 0.1 | 0.0003 | 32 | 1.3588 | 0.7201 | 76.8 |
| [64, 32] | 0.3 | 0.001 | 32 | 1.3703 | 0.7084 | 48.2 |
| [64, 32] | 0.3 | 0.0003 | 32 | 1.3828 | 0.6438 | 101.2 |

### Chosen configuration

```json
{
  "hidden_units": [
    32,
    16
  ],
  "dropout": 0.1,
  "learning_rate": 0.001,
  "batch_size": 32
}
```

**Why:** best mean CV mae across 4 expanding-window folds. Ties are impossible here because the grid is fully enumerated and scored deterministically under a fixed seed.

---

## target_pit_next_lap

**Task:** classification  |  **Features:** 8  |  **Combinations evaluated:** 8  |  **Folds per combination:** 4  |  **Total training runs:** 32

### Search space

| Hyperparameter | Values explored |
|---|---|
| Hidden units | [[16, 8], [32, 16]] |
| Dropout | [0.2, 0.4] |
| Learning rate | [0.001, 0.0003] |
| Batch size | [32] |
| Max epochs | 200 (early stopping, patience 20) |

**Selection metric:** roc_auc (higher is better), averaged across folds.

### All trials

| Hidden units | Dropout | LR | Batch | CV roc_auc (mean) | CV roc_auc (std) | Mean best epoch |
|---|---|---|---|---|---|---|
| [16, 8] | 0.2 | 0.001 | 32 | 0.8903 | 0.0581 | 63.5 |
| [16, 8] | 0.2 | 0.0003 | 32 | 0.8821 | 0.0599 | 116.0 |
| [16, 8] | 0.4 | 0.001 | 32 | 0.8899 | 0.0499 | 73.0 |
| [16, 8] | 0.4 | 0.0003 | 32 | 0.8816 | 0.0568 | 123.5 |
| [32, 16] | 0.2 | 0.001 | 32 | 0.8244 | 0.1485 | 24.0 |
| [32, 16] | 0.2 | 0.0003 | 32 | 0.8182 | 0.1570 | 75.0 |
| [32, 16] | 0.4 | 0.001 | 32 | 0.8136 | 0.1522 | 26.8 |
| [32, 16] | 0.4 | 0.0003 | 32 | 0.8133 | 0.1644 | 87.0 |

### Chosen configuration

```json
{
  "hidden_units": [
    16,
    8
  ],
  "dropout": 0.2,
  "learning_rate": 0.001,
  "batch_size": 32
}
```

**Why:** best mean CV roc_auc across 4 expanding-window folds. Ties are impossible here because the grid is fully enumerated and scored deterministically under a fixed seed.

---

