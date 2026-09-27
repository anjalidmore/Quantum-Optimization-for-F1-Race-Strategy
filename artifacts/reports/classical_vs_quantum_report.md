# Classical vs Quantum — Machine Learning on the Same F1 Data

_Generated 2026-09-27 16:11 UTC._

**Simulator:** default.qubit (noiseless state-vector simulation; no quantum hardware)  ·  **PennyLane** 0.45.1  ·  **seed** 42  ·  **total run time** 26.5 s

> **Read this first.** These are small variational circuits simulated exactly on a classical
> computer, trained on one Grand Prix. There is no quantum hardware in this project and nothing
> here demonstrates a quantum advantage — a noiseless simulation of 4 qubits is something a laptop
> does easily, which is the whole reason it is possible to run this experiment at all.
> The comparison is worth making because it is *fair*: same laps, same splits, same metric code.

## How the comparison was kept fair

| | |
|---|---|
| Data | the Task 5 feature matrix, loaded through Task 6's own data contract |
| Split | Task 6's chronological holdout — 815 development laps, 180 test laps |
| Cross-validation | Task 6's 4 expanding-window lap-forward folds |
| Selection | hyperparameters chosen on the folds only; the test laps were used once |
| Search space | layers ∈ [1, 2, 3], learning rate ∈ [0.05, 0.1] |
| Metrics | `app.intelligence.ml.evaluation` — the same functions Tasks 6 and 7 call |
| Thresholds | tuned on pooled out-of-fold predictions, never left at 0.5 |

### Getting 45 features into 4 qubits

- **Pit decision:** PCA to 4 components, retaining 81.8% of the variance.
- **Lap time:** PCA to 4 components, retaining 28.7% of the variance.

Angle encoding needs a bounded range, so features are standardised and mapped to [0, π]. Both the
standardiser and the PCA are fitted on training rows only — inside each fold during
cross-validation, and on the development laps for the final model.

This reduction is a real cost, and it is why the table below includes **parameter-matched classical
models trained on those same reduced inputs**. Task 6's model sees every feature; the matched models
see exactly what the circuits see.

## Results — same test laps, same metrics

### Pit decision (`target_pit_next_lap`)

Test laps contain **1 pit event(s) in 180 laps**. Precision, recall and F1 on so few positives are dominated by chance; the cross-validated columns carry far more events and are the better guide.

| Model | Family | Params | Train (s) | CV PR-AUC (mean ± sd) | CV ROC-AUC | Test PR-AUC | Test ROC-AUC | Test F1 | Test precision | Test recall | Confusion matrix |
|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---|
| vqc | quantum | 14 | 0.44 | 0.3037 ± 0.1912 | 0.7699 ± 0.1491 | 0.0080 | 0.3073 | 0.0000 | 0.0000 | 0.0000 | `[[108, 71], [1, 0]]` |
| quantum_kernel_svm | quantum | 413 | 3.72 | 0.2127 ± 0.1940 | 0.6543 ± 0.1889 | 0.0122 | 0.5475 | 0.0000 | 0.0000 | 0.0000 | `[[162, 17], [1, 0]]` |
| logistic_regression (reduced inputs) | classical (parameter-matched) | 5 | 0.00 | 0.2047 ± 0.1333 | 0.7162 ± 0.1405 | 1.0000 | 1.0000 | 0.0132 | 0.0066 | 1.0000 | `[[29, 150], [0, 1]]` |
| tiny_mlp (reduced inputs) | classical (parameter-matched) | 13 | 0.02 | 0.1529 ± 0.0883 | 0.7066 ± 0.1344 | 0.0137 | 0.7989 | 0.0118 | 0.0060 | 1.0000 | `[[12, 167], [0, 1]]` |
| random_forest — Task 6, all selected features | classical (full) | — | not measured here | 0.3863 ± 0.1249 | 0.8482 ± 0.0560 | 0.2500 | 0.9832 | 0.0833 | 0.0435 | 1.0000 | `[[157, 22], [0, 1]]` |

A test PR-AUC of 1.0 in that table is not a triumph: with a single positive lap, PR-AUC is 1.0 whenever that one lap happens to receive the highest score, and near 0.01 whenever it does not. The same one lap drives every test precision, recall and F1 value here. Use the CV columns.

`quantum_kernel_svm`'s parameter count is its SVM dual coefficients plus the intercept — the quantum part of that model is the kernel, which has no trained parameters at all. It is not comparable to the variational circuits' angle counts.

### Lap time (`target_laptime`)

| Model | Family | Params | Train (s) | CV MAE (mean ± sd) | CV R² | Test MAE (s) | Test RMSE (s) | Test R² |
|---|---|---:|---:|---|---|---:|---:|---:|
| vqr | quantum | 14 | 0.44 | 1.7459 ± 0.3706 | -0.1188 ± 0.2759 | 1.1044 | 1.4866 | -0.4562 |
| linear_regression (reduced inputs) | classical (parameter-matched) | 5 | 0.00 | 1.6243 ± 0.3357 | -0.1015 ± 0.3086 | 0.9459 | 1.3355 | -0.1753 |
| tiny_mlp (reduced inputs) | classical (parameter-matched) | 13 | 0.00 | 1.7697 ± 0.2388 | -0.1636 ± 0.2548 | 1.6813 | 1.8963 | -1.3695 |
| svr — Task 6, all selected features | classical (full) | — | not measured here | 1.3813 ± 0.5584 | 0.2553 ± 0.3918 | 0.7815 | 1.0290 | 0.3023 |

## The three quantum models

**`vqc`** — AngleEmbedding + StronglyEntanglingLayers, PauliZ expectation through a sigmoid

- 4 qubits, 1 entangling layer(s), learning rate 0.1, 80 epochs (Adam)
- 14 trainable parameters, final fit in 0.44 s
- decision threshold 0.6519 tuned on 635 out-of-fold predictions (36 pit laps): F1 0.1552 at 0.5 → 0.3146 at the tuned value

**`quantum_kernel_svm`** — fidelity kernel |<phi(b)|phi(a)>|^2 (2 feature-map reps) + SVC(kernel='precomputed')

- 4 qubits, 2 feature-map repetitions, C=1.0
- 413 trainable parameters, final fit in 3.72 s
- decision threshold 0.8202 tuned on 635 out-of-fold predictions (36 pit laps): F1 0.1429 at 0.5 → 0.2449 at the tuned value

**`vqr`** — AngleEmbedding + StronglyEntanglingLayers, PauliZ expectation scaled to a standardised lap time

- 4 qubits, 1 entangling layer(s), learning rate 0.1, 80 epochs (Adam)
- 14 trainable parameters, final fit in 0.44 s

## Discussion

**Pit decision.** The best quantum model on cross-validation is `vqc` (CV PR-AUC 0.3037), which is ahead of the parameter-matched classical model `logistic_regression (reduced inputs)` (0.2047) on the identical reduced inputs. Task 6's `random_forest`, which sees every selected feature, reaches 0.3863 on the same folds.

On the test laps every classifier is judged on 1 pit event(s), so the test columns should not decide anything. That limitation is the dataset's, not the models'.

**Lap time.** `vqr` reaches CV MAE 1.7459 s against 1.6243 s for `linear_regression (reduced inputs)` on the same 4 PCA components — worse than the matched classical model by 0.122 s. Test MAE is 1.1044 s for the circuit and 0.7815 s for Task 6's `svr` on all features.

A circuit whose output is a single Pauli-Z expectation is bounded in [-1, 1] and has a few dozen parameters, so it is closer in capacity to a linear model than to Task 6's ensembles or Task 7's network. Reading it as "quantum is worse than classical" would be the wrong lesson; reading it as "a 26-parameter model with 4 compressed inputs cannot match a 45-feature ensemble" is the right one.

### What this experiment cannot tell you

- **Nothing about quantum advantage.** `default.qubit` simulates the circuit exactly. Any speed
  comparison in the table favours whichever model has fewer parameters to fit, not whichever is
  running on better hardware — there is no quantum hardware here.
- **Nothing about scaling.** 4 qubits and a 16-dimensional state space is small enough to simulate
  in milliseconds. The interesting regime starts where simulation stops being possible, and this
  dataset never gets there.
- **Nothing generalisable about F1.** One race, one circuit, 995 laps, one labelled pit stop in the
  holdout. Every caveat that applies to Tasks 6-8 applies here too.
- **Noise is absent.** Real devices decohere. A model that trains cleanly in simulation may not
  survive contact with hardware, and this project makes no claim that it would.

### Reproducing this

```bash
python scripts/run_qml.py            # writes every artifact below
python scripts/build_all.py          # includes the quantum stage (skip it with --skip-qml)
```

Artifacts: `artifacts/models/qml/` (weights and config), `artifacts/metrics/qml_metrics.json`,
`artifacts/figures/qml_*.png`, and this report.

