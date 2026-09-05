# Task 7 - Hyperparameter Report

_Generated 2026-09-05 06:58 UTC._

Every combination below was **fully enumerated**, not sampled, and each was
scored on the same expanding-window lap-forward folds Task 6 uses. A random
K-fold search would let the network validate on laps it had already seen the
future of; the Task 5 contract forbids it explicitly.

## target_laptime

**Task:** regression  |  **Features:** 6  |  **Combinations evaluated:** 8  |  **Folds per combination:** 4  |  **Total training runs:** 32

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
| [32, 16] | 0.1 | 0.001 | 32 | 0.2489 | 0.0339 | 34.8 |
| [32, 16] | 0.1 | 0.0003 | 32 | 0.2594 | 0.0411 | 77.0 |
| [32, 16] | 0.3 | 0.001 | 32 | 0.2202 | 0.0282 | 42.5 |
| [32, 16] | 0.3 | 0.0003 | 32 | 0.2273 | 0.0243 | 97.2 |
| [64, 32] | 0.1 | 0.001 | 32 | 0.2035 | 0.0102 | 40.0 |
| [64, 32] | 0.1 | 0.0003 | 32 | 0.2145 | 0.0192 | 59.5 |
| [64, 32] | 0.3 | 0.001 | 32 | 0.1959 | 0.0170 | 43.8 |
| [64, 32] | 0.3 | 0.0003 | 32 | 0.2074 | 0.0085 | 71.8 |

### Chosen configuration

```json
{
  "hidden_units": [
    64,
    32
  ],
  "dropout": 0.3,
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
| [16, 8] | 0.2 | 0.001 | 32 | 0.3573 | 0.1080 | 38.0 |
| [16, 8] | 0.2 | 0.0003 | 32 | 0.3755 | 0.1121 | 123.2 |
| [16, 8] | 0.4 | 0.001 | 32 | 0.4178 | 0.1781 | 57.8 |
| [16, 8] | 0.4 | 0.0003 | 32 | 0.3803 | 0.1203 | 131.2 |
| [32, 16] | 0.2 | 0.001 | 32 | 0.3245 | 0.2608 | 23.0 |
| [32, 16] | 0.2 | 0.0003 | 32 | 0.3304 | 0.2653 | 81.0 |
| [32, 16] | 0.4 | 0.001 | 32 | 0.3366 | 0.2716 | 25.8 |
| [32, 16] | 0.4 | 0.0003 | 32 | 0.3420 | 0.2795 | 90.2 |

### Chosen configuration

```json
{
  "hidden_units": [
    16,
    8
  ],
  "dropout": 0.4,
  "learning_rate": 0.001,
  "batch_size": 32
}
```

**Why:** best mean CV roc_auc across 4 expanding-window folds. Ties are impossible here because the grid is fully enumerated and scored deterministically under a fixed seed.

---

