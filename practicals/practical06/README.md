# Phase 3 · Task 7 — Deep Learning Model Development

Part of **Quantum Optimization for Formula 1 Race Strategy**.

> **Scope of this module:** classical deep learning. No quantum computing. It trains
> deep neural networks on the feature matrix Task 5 (practical05) exported, and asks a
> single honest question: **does a DNN actually beat the classical baselines on this
> data?** The answer here is *yes for one target and no for the other*, and both
> results are reported as measured.

---

## 1. What this module does

Two Keras multilayer perceptrons, one per race-state target:

| Target | Task | Output head | Loss |
|---|---|---|---|
| `target_laptime` | regression — expected lap time in seconds | linear | MSE |
| `target_pit_next_lap` | binary classification — pit on the next lap? | sigmoid | binary cross-entropy |

Each is tuned by a fully-enumerated grid search on **expanding-window lap-forward
folds**, retrained on the full development set with early stopping, then evaluated
**once** on a chronological holdout that nothing touched until that moment.

| Deliverable | Artifact produced |
|---|---|
| Trained models | `outputs/models/target_*.keras` (+ fitted scalers, + feature spec) |
| Training history | `outputs/history/target_*_history.json` (per-epoch loss/val_loss/metric) |
| Loss & metric curves | `outputs/figures/target_*_training_history.png` |
| Model comparison charts | `outputs/figures/target_*_model_comparison.png` |
| Evaluation report | `outputs/reports/dl_evaluation_report.md` |
| Hyperparameter report | `outputs/reports/hyperparameter_report.md` |
| Model registry | `outputs/metadata/dl_model_registry.json` |

---

## 2. Results (measured, not asserted)

Run on practical05's **synthetic** 520-row matrix — 420 training rows, 100 test rows.

### `target_laptime` — the DNN wins

| Model | MAE (s) | RMSE (s) | R² |
|---|---:|---:|---:|
| **`dnn_mlp`** | **0.1584** | **0.2080** | **0.7631** |
| random_forest | 0.2298 | 0.2863 | 0.5509 |
| linear_regression | 0.3005 | 0.3603 | 0.2892 |
| decision_tree | 0.4210 | 0.5134 | −0.4434 |
| mean_baseline | 0.4645 | 0.5302 | −0.5398 |

31% lower MAE than the best classical model on the same test set. Chosen
configuration: 2 hidden layers (64, 32), dropout 0.3, lr 1e-3, batch 32; early
stopping restored epoch 65 of 85 run.

### `target_pit_next_lap` — no verdict from the holdout, and the DNN is poor in CV

**The holdout contains 0 positive examples.** Every pit event in this synthetic
session occurs at lap 18, 27 or 36; the chronological holdout is laps 46–55. ROC-AUC
and PR-AUC are therefore mathematically undefined for the DNN *and* for every
classical baseline alike. That is a property of the data, reported rather than
patched around.

Falling back to the cross-validation folds, where positives exist: the DNN's mean CV
ROC-AUC is **0.4178** — *worse than chance*. With 16 positive examples in the whole
matrix and 8 input features there is not enough signal for a network to learn a
ranking. This is an honest negative result and exactly what the small-data caveat in
`src/f1dl/models.py` predicts.

> On the `proj-mode` branch the same code runs against the 995-row **real** FastF1
> matrix, whose pit events are not clustered into three laps, and produces a defined
> holdout score.

---

## 3. Why the architecture is small

520 rows. After the holdout and folds, a training fold is a few hundred rows. The
lap-time network already has **2,561 parameters against 420 training rows — a ratio of
6.1**. Anything deeper memorises laps instead of learning race dynamics, so three
regularisers are applied together and each is reported:

* **Dropout** on every hidden layer (0.3 chosen by the search)
* **L2 weight decay** (1e-4) on every Dense kernel
* **Early stopping** on `val_loss`, patience 20, best weights restored

The evaluation report states, per target, how many epochs of validation-loss
deterioration early stopping actually discarded — so the mechanism is shown working,
not merely configured.

### One implementation detail that matters

`target_laptime` has mean ≈ 91 s but standard deviation ≈ 0.56 s. A linear output head
initialised near zero, trained under MSE with L2 decay, **cannot** climb to 91 within
any sane epoch budget — the first run of this practical produced an MAE of 29 seconds
and an R² of −5949 for exactly that reason. The target is therefore standardised using
the **training rows only** and predictions are inverse-transformed before any metric is
computed, so every number above is in real seconds. Tree models are scale-invariant and
need none of this; the network needs it to be compared *fairly* rather than handicapped
by a detail unrelated to its modelling ability. See `src/f1dl/training.py`.

---

## 4. Validation strategy — and why not K-fold

The Task 5 contract is explicit:

> "Expanding-window lap-forward split; keep whole laps in one fold. Do not use random
> K-fold — this is a time-ordered panel."

`src/f1dl/splits.py` is copied **verbatim** from Task 6's `app/intelligence/ml/splits.py`,
and `src/f1dl/evaluation.py` from its `evaluation.py`. The DL and classical numbers in
the comparison table are therefore produced by the *same code*, not merely by
same-named metrics — so any difference is attributable to the model.

A random split would let the network validate on lap 20 having already seen lap 40.
No deployed race-strategy model is ever in that position.

---

## 5. Prerequisites & running

**Run practical05 first** — this practical reads its outputs.

```bash
cd practicals/practical06
pip install -r requirements.txt
python practical06.py                 # full run, ~4 minutes
python practical06.py --quick         # fast smoke run (NOT the committed results)
python practical06.py --output-dir /tmp/somewhere
pytest -q                             # 21 tests
```

### Framework note — a documented divergence from the task spec

The reference task table names **TensorFlow/Keras** and an **`.h5`** deliverable.
TensorFlow publishes **no wheel for Python 3.14** (this project's interpreter) —
`pip index versions tensorflow` returns no distribution at all. Keras 3 is
backend-agnostic, so it runs here on the **PyTorch backend**; the Keras API in
`src/f1dl/` is identical either way, and setting `KERAS_BACKEND=tensorflow` on a Python
version where TF exists runs the same code unchanged.

Models ship as **`.keras`**, not `.h5`: under Keras 3 a model saves to HDF5 but fails to
load back (`Could not deserialize 'keras.metrics.mse'`), verified on this installation.
A model that cannot be reloaded is not a deliverable.

---

## 6. Layout

```
practical06/
├── practical06.py              # end-to-end driver
├── README.md
├── requirements.txt
├── pytest.ini
├── src/f1dl/
│   ├── contract.py             # Task 5 contract gate (leakage checks)
│   ├── splits.py               # copied verbatim from Task 6
│   ├── evaluation.py           # copied verbatim from Task 6
│   ├── models.py               # the two Keras architectures + rationale
│   ├── training.py             # fold fitting, fold-local scaling, early stopping
│   ├── tuning.py               # fully-enumerated grid over the folds
│   ├── baselines.py            # classical models on the identical split
│   ├── persistence.py          # .keras save/load + scalers + feature spec
│   ├── visualize.py            # loss curves, comparison charts
│   └── reports.py              # Markdown deliverable generators
├── tests/test_deep_learning.py # 21 tests
└── outputs/                    # generated (git-ignored)
```
