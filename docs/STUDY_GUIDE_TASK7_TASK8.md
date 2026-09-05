# Study Guide — Tasks 7 & 8 (Deep Learning and Explainable AI)

This file prepares you to be **examined on the tools and concepts**, independently of
this codebase. For what *this project* actually built, see
[`TASK7_TASK8_DETAILS.md`](TASK7_TASK8_DETAILS.md).

Everything here is written to be defensible under follow-up questions. Where this
project's real results contradict the textbook expectation, the textbook expectation is
not what's written down.

---

## 1. Concept primers

### 1.1 What a DNN / MLP is, and how it differs from Task 6's models

A **multilayer perceptron** is a stack of layers. Each layer computes
`output = activation(W · input + b)` — a linear transform followed by a non-linearity.
Stacking non-linear layers lets the network represent functions that no single linear
transform can.

| | Task 6 classical | Task 7 MLP |
|---|---|---|
| How it fits | Trees split on feature thresholds; linear models solve for coefficients | Gradient descent on weights, iteratively |
| Feature interactions | Trees find them by splitting repeatedly; linear models need them hand-built | Learned automatically in the hidden layers |
| Scale sensitivity | Trees are scale-**invariant** | Very sensitive — inputs *and* targets usually need standardising |
| Data appetite | Works well on hundreds of rows | Typically needs thousands |
| Training determinism | Deterministic given a seed | Deterministic given a seed, but sensitive to init and batch order |

**The honest headline for an examiner:** a neural network's advantage is *representational
capacity*, and capacity is only useful when you have enough data to constrain it. On a
few hundred rows a random forest is often the better engineering choice, and that is a
statement about the data, not about deep learning.

### 1.2 Overfitting, and the three countermeasures used here

**Overfitting** is when a model learns the training rows rather than the pattern behind
them: training loss keeps falling while validation loss turns and rises. The gap between
the two curves *is* the diagnostic.

| Mechanism | What it does | Why it helps on small data |
|---|---|---|
| **Dropout** | Randomly zeroes a fraction of units each training step | Stops the network relying on any one unit; acts like averaging many thinner networks |
| **L2 weight decay** | Adds `λ·Σw²` to the loss | Keeps weights small, so the fitted function stays smooth instead of contorting through every training point |
| **Early stopping** | Halts when validation loss stops improving, restores the best weights | The cheapest regulariser there is — it uses the validation curve to pick the model *before* it started memorising |

**The number that frames all of this:** parameters ÷ training rows. When a network has
more parameters than examples, it *can* memorise the training set exactly, and only
regularisation prevents it. This project's lap-time network is in that regime and the
report says so explicitly rather than hiding it.

### 1.3 SHAP — Shapley values, conceptually

SHAP comes from **cooperative game theory**. Imagine the features are players
cooperating to produce a prediction. The Shapley value of a player is its **average
marginal contribution across every possible order** in which players could join the
coalition.

Formally, for feature *i*:

```
φᵢ = Σ over subsets S not containing i:
        [ |S|! · (n−|S|−1)! / n! ] · ( f(S ∪ {i}) − f(S) )
```

You do not need to reproduce that formula in an exam, but you should be able to say what
it *does*: it averages "how much did adding this feature change the prediction?" over all
orderings, weighting orderings so the result is fair.

Shapley values are the unique attribution satisfying four axioms:

* **Local accuracy** — attributions sum to `prediction − base_value`. This is the
  guarantee LIME does not have.
* **Missingness** — a feature not present gets zero attribution.
* **Consistency** — if a model changes so a feature contributes more, its attribution
  cannot go down.
* **Symmetry** — two features that contribute identically get identical attributions.

**The catch:** exact computation is exponential in the number of features. That is why
the choice of *explainer* matters so much (§2).

### 1.4 LIME — and precisely how it differs from SHAP

**LIME** = Local Interpretable Model-agnostic Explanations. To explain one prediction:

1. Perturb the row many times (this project: 2000 perturbations).
2. Ask the **real** model to predict on each perturbation.
3. Weight each perturbation by how close it is to the original row.
4. Fit a **simple linear model** to those weighted predictions.
5. Report that linear model's coefficients as the explanation.

| | SHAP | LIME |
|---|---|---|
| Question | How is credit for this prediction fairly divided? | What simple model behaves like the real one *around here*? |
| Basis | Game theory (Shapley values) | Local weighted linear regression |
| Output | Attribution per feature, in the model's output units | Coefficients on (often discretised) conditions |
| Guarantee | Attributions sum to prediction − base | **None** — quality reported as the surrogate's R² |
| Determinism | Exact for trees; sampled otherwise | Always sampled; needs a fixed seed |
| Cost | Cheap for trees, expensive otherwise | Moderate, one model per explained row |

**The distinction an examiner will probe:** SHAP is an *attribution* method with
axiomatic guarantees; LIME is an *approximation* method that builds a surrogate. LIME's
`local_r2` tells you whether the surrogate was any good — a low value means a straight
line is a poor stand-in for the model near that row, so the explanation should be
discounted however confident it looks. SHAP has no equivalent warning signal, which is
exactly why using both and comparing them (as this project's trust score does) is more
informative than using either alone.

### 1.5 Counterfactual explanations

A counterfactual answers: **"what would have to be different for the outcome to change?"**
It is *contrastive* — the form human explanation naturally takes. "You were denied
because your income is £28k; at £35k you would have been approved" is more useful than a
list of attributions.

Good counterfactuals are:

* **Actionable** — change things that can actually change. Tyre age can change; a
  driver's identity cannot.
* **Sparse** — few features altered. "Change six things at once" is not an instruction.
* **Plausible** — inside the data's observed range, not physically impossible values.
* **Diverse** — several genuinely different routes, when several exist.

Two families: **optimisation-based** (search for the nearest row crossing the boundary —
DiCE) and **perturbation/scan-based** (sweep one feature and find the crossing). The
second is less general but far more actionable, which is why this project reports both.

### 1.6 Fairness — and why identity features are the specific risk here

"Fairness" usually means outcomes not depending unjustifiably on a protected attribute.
This project is not making decisions about people's rights, so the framing is narrower
and more concrete: **is the model predicting from race state, or from who is driving?**

Task 5's selection funnel retained one-hot `driver_*` and `team_*` dummies. A model
leaning on those has learned "this driver's laps look like this" rather than "a tyre this
old on a track this hot laps like this". Three concrete consequences:

1. It **cannot generalise** to a driver it has not seen.
2. It **conflates car and driver** — a driver's historical pace is mostly their car's.
3. It would give **two cars in an identical race state different calls** purely because of
   the name on the car.

The measurement used here: identity features' share of total mean-|SHAP| attribution,
divided by the share you would expect if attribution were spread evenly. A ratio above 1
means identity is doing more work than its share of the feature count.

---

## 2. Tool / framework reference

| Tool | What it's for | Why chosen over alternatives | Where in this codebase |
|---|---|---|---|
| **Keras 3** | High-level neural-network API | Concise, readable model definition; the standard vocabulary an examiner expects (`Sequential`, `Dense`, `Dropout`, `EarlyStopping`, `model.fit`) | `app/intelligence/dl/models.py` — `build_regression_mlp`, `build_classification_mlp` |
| **PyTorch** (as Keras backend) | Tensor/autograd engine | **TensorFlow publishes no wheel for Python 3.14**, this project's interpreter. Keras 3 is backend-agnostic, so the same Keras code runs on torch. Setting `KERAS_BACKEND=tensorflow` on an older Python runs it unchanged. | `os.environ["KERAS_BACKEND"] = "torch"` at the top of `models.py` |
| **`keras.callbacks.EarlyStopping`** | Stop before overfitting; restore best weights | Built in, and `restore_best_weights=True` means the saved model is the best one seen, not the last one | `app/intelligence/dl/training.py` — `fit_fold` |
| **`sklearn.preprocessing.StandardScaler`** | Standardise features and the regression target | Fitted **inside** each fold, per the Task 5 contract, so validation statistics never leak into training | `training.scale_fit_transform` |
| **SHAP `TreeExplainer`** | Exact Shapley values for tree ensembles | Polynomial-time and *exact* — free accuracy. Always prefer it when the model is a tree | `app/intelligence/xai/shap_analysis.py` — `tree_shap` |
| **SHAP `KernelExplainer`** | Model-agnostic sampled Shapley values | Needs only a `predict` function, so it works on the Keras network. `DeepExplainer`'s Keras 3 support targets the TensorFlow backend, which is unavailable here | `shap_analysis.kernel_shap` |
| **`shap.kmeans`** | Summarise the background distribution | KernelExplainer's cost scales with background size; k-means keeps it tractable | `kernel_shap`, `k=25` |
| **LIME `LimeTabularExplainer`** | Local linear surrogates | The standard tabular LIME implementation; `discretize_continuous=True` produces readable conditions like `tyre_life > 12.00` | `app/intelligence/xai/lime_analysis.py` |
| **DiCE (`dice_ml`)** | Diverse whole-row counterfactuals | The best-known counterfactual library; `method="random"` needs no gradients, so it works with any sklearn model | `app/intelligence/xai/counterfactual.py` — `dice_counterfactuals` |
| **Permutation importance** (hand-rolled) | Global importance, model-agnostic | `sklearn.inspection.permutation_importance` expects an estimator; this project needs to score a **plain callable** so the Keras network and the sklearn pipeline go through identical code | `app/intelligence/xai/importance.py` |

### Why permutation importance rather than each model's native attribute

A tree's `feature_importances_` is mean impurity decrease. A linear model's is
coefficient magnitude. A network has neither. These are **not the same quantity** and
putting them side by side in one table would be meaningless. Permutation importance asks
one question of any model — *how much worse does it get if I destroy this feature?* — in
units of the evaluation metric. That is comparable across families by construction.

---

## 3. Likely evaluator questions, with model answers

**Q: Why expanding-window validation instead of K-fold?**
Because the data is a time-ordered driver × lap panel. Random K-fold would put lap 40 in
training and lap 20 in validation, letting the model see the future of the race it is
being scored on. Every score would be optimistic and none would predict deployed
behaviour. Expanding-window folds train on laps 1…*k* and validate on laps *k+1*…, which
is the situation a real strategy model is in. The Task 5 metadata states this as a
contract — `"Do not use random K-fold - this is a time-ordered panel"` — and Task 7
imports Task 6's `splits.py` directly rather than reimplementing it.

**Q: Why might your DNN underperform the random forest here?**
Three reasons, in order of importance. (1) **Data volume** — under a thousand rows from a
single race; the lap-time network has more parameters than training examples. (2)
**Inductive bias** — tabular data with threshold-like structure (tyre age past a cliff,
lap in a pit window) is exactly what axis-aligned splits capture natively and what a
network has to learn from scratch. (3) **Class imbalance** for the pit target, where
positives are a small fraction of laps. This is a well-documented result: gradient-boosted
trees remain competitive with or better than deep networks on small-to-medium tabular
problems.

**Q: How do you know your SHAP explanation isn't misleading?**
Four independent guards. (1) The tree model uses `TreeExplainer`, which is **exact**, not
sampled. (2) The network's `KernelExplainer` values are labelled approximate in the
report, with `nsamples` recorded, and the report says to compare **rank and sign**, not
magnitude. (3) LIME is computed on the same rows as an independent second opinion, and
where the two disagree the **trust score is reduced** rather than one being silently
preferred. (4) LIME's `local_r2` is reported, so a poor local surrogate is visible.
Beyond that: SHAP explains what the **model** does, not what is **true** — a model that
learned a spurious correlation will get a faithful explanation of that spurious
correlation. That is precisely why the fairness assessment exists.

**Q: Why is your target standardised but your binary features aren't?**
The regression target has mean ≈ 91 s but standard deviation ≈ 0.56 s. A linear output
head initialised near zero, trained under MSE with L2 decay, cannot climb to 91 within
any reasonable epoch budget — an early run of this project produced an MAE of 29 seconds
and an R² of −5949 for exactly that reason. Standardising the target (fitted on training
rows only, inverted before any metric) fixes it. Binary indicators are left alone because
standardising a 0/1 dummy destroys its interpretability without helping the optimiser.

**Q: Your `.h5` deliverable is a `.keras` file. Why?**
Under Keras 3 the HDF5 path is legacy: the model saves, but `load_model` fails with
`Could not deserialize 'keras.metrics.mse'` — verified on this installation. A model that
cannot be reloaded is not a deliverable. The native `.keras` archive round-trips
correctly, so that is what ships, and the divergence is documented in the report, the
registry entry and the API response rather than left for someone to discover.

**Q: What would you change with more data?**
In priority order. (1) **More sessions** — hold out whole *races*, not the tail of one
race; that alone would fix the biggest weakness in the evaluation. (2) **Sequence models**
— an LSTM/GRU or temporal transformer over stint history, since lap time is a sequence
problem currently flattened into per-lap rows. (3) **Threshold tuning and class weighting**
for the pit classifier, chosen on validation folds rather than defaulting to 0.5. (4)
**Drop the identity features** and check whether performance survives — if it does, the
model generalises to new drivers; if it collapses, that is itself the finding.

**Q: Is your trust score principled or arbitrary?**
The three **components** are principled: distance from the decision boundary, agreement
between two independently-trained model families, and agreement between two independent
explanation methods. Each captures a distinct, real failure mode. The **weights**
(0.40/0.30/0.30) are a documented judgement, not a derivation — confidence carries the
largest share because a prediction on the boundary is unusable regardless of explanation
quality. They are exposed as `app.intelligence.xai.trust.WEIGHTS` precisely so they can be
challenged. The honest framing: the score is a structured summary of three real signals,
not a calibrated probability, and the bands are deliberately conservative.

**Q: Why did DiCE need a timeout?**
Its random search samples the feature space until it finds an opposite-class row. With
pit events at roughly 5% of laps, an early run sampled for **15 minutes and returned
nothing**. It is now bounded to 60 seconds, queried from the row closest to the decision
boundary (where a flip is most likely reachable), and restricted to each feature's
observed training range. An unbounded search that produces no answer is worse than a
bounded one that says it ran out of budget.

**Q: Your counterfactual says "not reachable". Isn't that a failure?**
It's a result. Sweeping tyre age across its entire observed range with the rest of the
race state fixed never crosses the decision threshold — which tells you the decision is
driven by **combinations** of features rather than tyre age alone. Reporting that, with
the range searched, is more useful than reporting a number from a search that didn't
actually find anything.

**Q: What does a negative R² mean here specifically?**
R² ≤ 0 means the model does worse than always predicting the training mean. It is not a
bug and it is not impossible — it happens when a model extrapolates badly to a test
regime unlike its training data. Here the holdout is the final laps of a single race,
where fuel load and tyre state differ substantially from earlier laps. The correct
reading is "this model does not extrapolate to late-race conditions", not "the code is
broken". Note also that model selection uses **cross-validated** MAE, where the model is
positive — so the selection procedure is sound even where the holdout number is poor.

---

## 4. Quick-reference cheat sheet

### Architectures

```
LAP-TIME REGRESSION                 PIT-DECISION CLASSIFICATION
input (n features)                  input (n features)
   ↓ Dense(64, relu) + L2              ↓ Dense(16, relu) + L2
   ↓ Dropout(0.3)                      ↓ Dropout(0.4)
   ↓ Dense(32, relu) + L2              ↓ Dense(8, relu) + L2
   ↓ Dropout(0.3)                      ↓ Dropout(0.4)
   ↓ Dense(1, LINEAR)                  ↓ Dense(1, SIGMOID)
loss = MSE, Adam                    loss = binary cross-entropy, Adam
target STANDARDISED (train only)    class_weight = balanced
```

Both: early stopping on `val_loss`, patience 20, `restore_best_weights=True`.
Hyperparameters chosen by a **fully enumerated** grid over expanding-window folds.

### The trust score

```
trust = 0.40 · confidence            2·|p − 0.5|   (or 1 − gap/σ for regression)
      + 0.30 · model_agreement       1 − |p_dnn − p_classical|
      + 0.30 · explanation_stability Jaccard(SHAP top-3, LIME top-3)

≥0.75 HIGH  ·  0.50–0.75 MODERATE  ·  0.25–0.50 LOW  ·  <0.25 DO NOT ACT
```

### Fairness measurement

```
identity_share      = Σ mean|SHAP| over driver_*/team_*  ÷  Σ mean|SHAP| over all
expected_if_uniform = n_identity_features ÷ n_features
concentration       = identity_share ÷ expected_if_uniform

concentration > 1  → identity over-weighted relative to its feature count
concentration < 1  → race-state features dominate (the desired outcome)
```

### Explainer decision table

| Model type | Explainer | Exact? |
|---|---|---|
| Tree ensemble | `shap.TreeExplainer` | Yes |
| Keras / any callable | `shap.KernelExplainer` | No — sampled |
| Any (local surrogate) | `LimeTabularExplainer` | No — surrogate, check `local_r2` |
| Any (global) | permutation importance | Exact given enough repeats |

### The one-line answers

* **Why not K-fold?** Time-ordered panel; K-fold leaks the future.
* **Why standardise the target?** σ = 0.56 s around a mean of 91 s; a linear head can't reach 91.
* **Why `.keras` not `.h5`?** Keras 3 saves HDF5 but can't reload it.
* **Why permutation importance?** The only metric comparable across a forest and a network.
* **Why `TreeExplainer` for one model and `KernelExplainer` for the other?** Exact when possible, model-agnostic when necessary.
* **Why both SHAP and LIME?** Their disagreement is a signal, and the trust score uses it.

---

# Appendix A — Engineering concepts from the Phase 1 remediation

Added while closing the security and test-infrastructure entries in `TODO.md`. Same
format as the primers above: the concept first, then **how it showed up here and what we
did about it** — so each one can be explained from real evidence rather than theory.

## A.1 Test hermeticity

**The concept.** A test is *hermetic* if running it leaves the world exactly as it found
it: same files, same database, same clock-independent result. Non-hermetic tests fail in
three characteristic ways:

1. **Order dependence** — test B passes only because test A ran first and left something
   behind.
2. **Repeat-run divergence** — the second run differs from the first.
3. **State pollution** — the test mutates something a human cares about, outside the test.

The third is the sneakiest, because nothing fails. Everything is green while the test is
quietly editing your source of truth.

**How it showed up here.** Running `pytest` retrained Task 6 and overwrote **ten tracked
files** under `artifacts/` — four metrics JSON, four Markdown reports, `manifest.json` and
`model_registry.json`. A clean clone went dirty just from running the documented test
command.

Two concrete harms followed, and the second one actually happened:

- It **trains contributors to run `git checkout -- artifacts/` reflexively**, which is
  precisely how a real artifact change gets discarded by accident.
- It **destroyed data.** Task 7 extends the *shared* model registry; Task 6's
  `write_registry` overwrote that file wholesale. So running the test suite deleted Task
  7's registry rows, leaving `/api/dl/models` returning 404 while the `.keras` files sat
  untouched on disk. A test suite deleted production metadata, and nothing went red.

**What we did.** Added `ArtifactPaths` to `app/core/paths.py` — one frozen dataclass
resolving every output location from a single root — and gave `train_all()` / `run_all()`
an optional `output_root` that defaults to the committed layout. Production behaviour is
byte-identical; the tests pass a `tmp_path`. Then we proved it the only way that counts:
applied the change to a fresh clone and ran the suite **twice**, confirming
`git status --porcelain artifacts/` was empty both times.

**The structural lesson.** We also added a CI step that fails if `git status --porcelain`
is non-empty after `pytest`. That is the difference between hermeticity as a *convention*
(which decays as soon as someone is in a hurry) and as an *invariant* (which cannot decay
without the build going red). Prefer the second whenever you can express the property
mechanically.

> **Likely question: "your tests write files — isn't that an integration test, not a unit
> test?"** Yes, deliberately. Task 6's tests train real models and assert on real metrics;
> mocking them would test the mock. The problem was never that they write, it was *where*
> they wrote. Redirecting the output root keeps the integration value and removes the
> pollution.

## A.2 Why a CORS wildcard is dangerous even "just locally"

**The concept.** The **same-origin policy** stops a page on `evil.com` reading responses
from `yourbank.com`. **CORS** is the server's way of making exceptions:
`Access-Control-Allow-Origin` names who may read the response. `allow_origins=["*"]` says
*everyone*.

The dangerous misconception is "it's fine, it's only on localhost." Your browser can reach
`127.0.0.1` perfectly well, and so can any page you happen to have open. A tab on some
unrelated site can issue `fetch('http://127.0.0.1:8000/api/...')` and, with a wildcard,
**read the response**. Localhost is not a security boundary in a browser; the origin
policy is, and a wildcard removes it.

Two related points worth knowing:
- Wildcard origins and credentialed requests are mutually exclusive by spec — the browser
  refuses `allow_origins=["*"]` together with cookies. That is a hint that `*` was never
  meant for anything holding data you care about.
- `allow_methods=["*"]` and `allow_headers=["*"]` widen it further, permitting
  content types that would otherwise trigger a preflight the server could refuse.

**How it showed up here.** `app/api/main.py` had all three wildcards, *and* mounted the
entire `artifacts/` tree at `/artifacts` — including `artifacts/models/`. Combined, any
page in the developer's browser could have enumerated and downloaded every trained model
(`.joblib`, `.keras`) from the running API. Neither issue is severe alone; together they
are exfiltration-by-default.

**What we did.** An env-configured origin allow-list defaulting to `localhost:3000`,
methods narrowed to `GET, POST`, headers to `Content-Type`. Separately, only the six
directories the dashboard actually reads are mounted; `models/` and `metadata/` have no
route at all.

**The design lesson — structural over filtered.** We could have kept one mount and
filtered requests for `models/`. Instead there is simply *no route*. A filter is code that
can be bypassed by a path the author did not imagine (`../`, percent-encoding, symlinks);
an absent route cannot be. When you can express a restriction as "this does not exist"
rather than "this is checked", do. We wrote a test firing traversal attempts at the
boundary to confirm it.

## A.3 What a supply-chain advisory actually tells you

**The concept.** `pip-audit` and `npm audit` compare your resolved dependency tree against
vulnerability databases (PyPI Advisory / OSV, and the GitHub Advisory Database). They
answer exactly one question: *does a package version in your tree appear in an advisory?*

What they emphatically do **not** tell you:
- **Whether you are exploitable.** An advisory about a code path you never call is still
  reported. Reachability analysis is a different, harder problem.
- **Whether the fix is safe.** `fixAvailable` may be a major version with breaking changes.
- **Whether you are actually safe when clean.** A clean report means "nothing *known*
  matches", not "no vulnerabilities".

The useful mental model: an audit is a **smoke detector, not a fire inspection**. Act on
it, but read the finding before you act.

**How it showed up here — two findings with opposite conclusions.**

*Python.* `pip-audit` flagged **msgpack 1.1.2** (PYSEC-2026-3625), reaching us
transitively via `fastf1 → signalrcore`. Pinning `msgpack>=1.2.1` cleared it. But the fix
had a **cost the audit did not mention**: `signalrcore` 1.0.2 hard-pins
`msgpack==1.1.2`, so pip silently resolved *signalrcore itself* down from 1.0.2 to 0.8.8.
We accepted it after checking `pip check`, confirming `fastf1.livetiming.SignalRClient`
still imports, and noting this project reads historical sessions rather than live timing —
and we wrote all of that into `requirements.txt` so the next person inherits the reasoning
rather than a mystery pin. **A transitive fix can move packages you did not name. Always
look at what actually got installed.**

*JavaScript.* `npm audit` reported 5 high-severity advisories (Next.js SSRF, cache
poisoning, unauthenticated disclosure of internal endpoints; postcss XSS and arbitrary
file read). Here the correct engineering answer was **not to fix it immediately**:
14.2.35 is already the latest 14.2.x, and `npm audit --json` shows the only `fixAvailable`
is `next@16.3.4` with `isSemVerMajor: true`. A two-major-version bump is a migration to
plan, not a reflex — so we made it *visible* (a non-gating CI step) and escalated the
decision instead of pretending it away.

**The judgment lesson.** Two advisories, two different right answers. "Make the tool go
quiet" is not the goal; understanding what each finding costs to fix, and being honest
when you choose not to fix one yet, is.

> **Likely question: "your CI has `continue-on-error` on npm audit — isn't that just
> disabling the check?"** No, and the distinction matters. The check still runs and still
> prints on every PR; what it does not do is block merges on something whose only remedy
> is a major migration. A gate you cannot satisfy gets bypassed permanently within a week;
> a visible warning with a tracked decision does not. The workflow comment says to remove
> the flag once the upgrade lands, so it is a marked temporary state rather than a
> quietly-weakened check.

## A.4 A note on `SettingWithCopyWarning` — and why the obvious fix was wrong

**The concept.** In pandas, indexing sometimes returns a **view** onto the original frame
and sometimes a **copy** — and which one you get is not always predictable. So when you
assign into the result of a previous index, pandas cannot promise the write reaches the
original data, and warns. It is a *correctness* warning, not style: your assignment may
silently do nothing.

**How it showed up here.** `clean_table()` took a frame, and callers passed slices of
larger tables. Every column assignment warned — 5 sites, contributing to a 1,084-warning
test run in which real signal was invisible.

**The fix, and the trap.** The backlog said to use `.loc[:, col] = …`. We tried it and it
**broke three tests**. Here is why, and it is worth understanding:

- `df[col] = values` **replaces the column**, including its dtype.
- `df.loc[:, col] = values` assigns **in place**, *preserving the existing dtype*.

The pipeline's whole job at that point is dtype-changing coercion — object → numeric,
strings → seconds, object → datetime. Under `.loc`, those coercions silently became
no-ops: the values were cast straight back to the original dtype. The warning went away
and the function stopped working.

The actual fix was one line — `df = df.copy()` at the top. The function's contract is to
*return* a cleaned frame, so it should never have written through to the caller's data in
the first place. Owning the frame removes the ambiguity at the source.

**Two lessons, both general.**
1. **A warning-suppression that changes behaviour is a bug, not a fix.** If silencing a
   warning also silences your logic, you have made things worse while making them look
   better. This is only catchable if your tests assert on *behaviour* (dtypes, computed
   moments) rather than on "it ran".
2. **Turn a fixed class of bug into an error.** `pyproject.toml` now has
   `filterwarnings = ["error::pandas.errors.SettingWithCopyWarning"]`. Once you have paid
   to fix something, spend the extra minute making it impossible to reintroduce.

---

# Appendix B — Model-quality concepts from the Phase 2 remediation

Added while fixing two "high ROC-AUC, useless in practice" classifiers and a regressor
that lost to the mean out of sample. As in Appendix A: the concept, then **what it looked
like here**.

## B.1 Why ROC-AUC is misleading under severe class imbalance

**The concept.** ROC-AUC plots the true-positive rate against the **false-positive rate**
across every threshold. The trap is in the denominator of that second term: FPR is
`FP / (FP + TN)`, and when negatives massively outnumber positives, `TN` is enormous. A
model can raise hundreds of false alarms and barely move FPR at all.

Concretely, at a 4.8% positive rate: 947 negatives, 48 positives. Flag 50 extra negatives
and FPR rises by 50/947 ≈ 0.05 — visually nothing on the curve. But precision has just
been destroyed, because you flagged 50 wrong laps to catch a handful of right ones.

ROC-AUC also has a probabilistic reading that makes its limits obvious: it is the chance
that a randomly-chosen positive is ranked above a randomly-chosen negative. That is a
statement about **ranking**, and it says nothing about whether any threshold produces a
usable decision.

**What it looked like here.** Task 6's `random_forest`: test ROC-AUC **0.98**, and
precision **0.04**. Task 7's network: ROC-AUC **0.92**, accuracy **99.4%**, and F1,
precision and recall all exactly **0.0** — it predicted "never pit" for all 180 test laps.
Both would have been reported as successes by ROC-AUC alone.

**What we did.** Made **CV PR-AUC** the primary selection metric in both
`ml/selection.py` and `dl/tuning.py`, and put it first in the dashboard tables with a note
explaining why.

> **Likely question: "so is ROC-AUC useless?"** No — it is the right metric when classes
> are roughly balanced, or when you genuinely care about ranking rather than a decision
> (triage ordering, for instance). It is the wrong *headline* for a rare-event decision
> problem. We still report it as a secondary metric, because "this model ranks well but
> cannot be thresholded usefully" is itself a diagnosis worth being able to state.

## B.2 What PR-AUC actually measures

**The concept.** The precision–recall curve plots precision against recall across
thresholds; PR-AUC (also called average precision) is the area under it. Neither axis
involves true negatives, so the vast majority class cannot flatter the score.

The single most useful fact about PR-AUC, and the one most often omitted: **its baseline
is the positive rate, not 0.5.** A random ranker scores ROC-AUC 0.5 regardless of
prevalence, but scores PR-AUC ≈ prevalence. So "PR-AUC 0.38" means nothing on its own —
you must know the base rate to interpret it.

**What it looked like here.** Prevalence 4.8%, so a random ranker scores ≈0.048.
Task 6's `random_forest` scored CV PR-AUC **0.3863** — about **8× better than chance**,
which is genuinely respectable and completely invisible in the ROC-AUC number. On the
`task-mode` synthetic data the same computation scored **0.0694** against a 0.048 baseline:
barely above chance, and the practical's README now says so in exactly those terms rather
than quoting a ROC-AUC that flatters it.

**The habit to take away.** Always quote PR-AUC next to its baseline. "PR-AUC 0.07" and
"PR-AUC 0.07 against a chance baseline of 0.05" are the same number and completely
different claims.

## B.3 Threshold tuning, and why 0.5 is an arbitrary default

**The concept.** A classifier outputs a probability; a decision needs a cut-off.
`sklearn`'s `predict()` uses 0.5, which is **provably optimal only when** the classes are
balanced *and* a false positive costs exactly as much as a false negative. Neither
condition is common, and neither holds here.

Two independent reasons 0.5 fails in this project:
1. **Imbalance.** With 4.8% positives, a well-calibrated model's probabilities cluster far
   below 0.5 for almost every row. The threshold sits outside the range where the model
   actually discriminates.
2. **Asymmetric costs.** Missing a pit window costs a driver track position and a ruined
   stint; an unnecessary early stop costs a few seconds. Treating those as equal is a
   modelling choice nobody made deliberately.

**How to do it without cheating.** The threshold is a hyperparameter, and tuning a
hyperparameter on the test set is leakage in exactly the ordinary way. We tune it on
**pooled out-of-fold predictions**: each validation fold's predictions come from a model
that had not seen those rows, and pooling all four folds gives 635 predictions with 36
positives instead of the 6–11 a single fold offers.

**What it looked like here.**

| | Task 6 best (OOF) | Task 7 DNN (OOF) |
|---|---:|---:|
| F1 @ 0.5 | 0.2381 | **0.0000** |
| F1 @ tuned | **0.2921** | **0.3143** |
| Threshold | 0.480 | 0.1189 |

The network's case is the vivid one: a model whose F1 was *exactly zero* — because every
predicted probability fell below 0.5 — became usable without a single weight changing.
Nothing was retrained. The model was always able to rank; it was never able to decide.

**One number that looks like a regression and is not.** The network's test accuracy fell
from 0.9944 to 0.8556. That is the fix working: it now fires on ~14% of laps instead of
none. 99.4% accuracy from a model that always says "no" is the class-imbalance illusion,
not a result.

> **Likely question: "did tuning the threshold improve your test F1?"** No, and this is
> worth being precise about. Our holdout contains **one positive example in 180 rows**, so
> test-set precision/recall/F1 there are a coin flip, not evidence. The improvement is
> measured out of fold, on 36 positives, which is the only place it *can* be measured
> honestly. Claiming a test-set improvement would have been claiming signal from a single
> sample.

## B.4 Probability calibration, and why `CalibratedClassifierCV` exists

**The concept.** A model is **calibrated** if its probabilities mean what they say: among
rows it scores 0.7, about 70% should be positive. Ranking and calibration are independent
— a model can rank perfectly while being badly calibrated (every probability squashed into
[0.0, 0.3], say), which is precisely how you end up with F1 = 0 at a 0.5 cut-off.

SVMs are the classic example: they optimise a margin, not a likelihood, so their decision
function is not a probability at all. `SVC(probability=True)` retrofits one via **Platt
scaling** — fitting a logistic regression to the decision values on internal
cross-validation folds. `CalibratedClassifierCV` does the same thing as an explicit,
inspectable wrapper (and also offers isotonic regression, a non-parametric alternative
that needs more data).

**What it looked like here.** `SVC(probability=True)` is deprecated in scikit-learn 1.9
and **removed in 1.11** — combined with unpinned dependencies, Task 6 would have broken on
a routine `pip install` with no code change on our side. We replaced it with
`CalibratedClassifierCV(SVC(), ensemble=False, cv=3)`. Metrics were essentially unchanged
(CV PR-AUC 0.3316 → 0.3321), which is what a correct calibration swap should look like:
the deprecation was the reason to move, not a performance claim.

**Why the explicit form is better anyway.** The calibration becomes a visible pipeline
step you can inspect, swap for isotonic, or reason about — rather than behaviour hidden
inside `SVC`'s constructor.

**The connection to B.3.** Calibration and threshold tuning solve *different halves of the
same problem*. Calibration makes the probabilities meaningful; threshold tuning picks
where to cut them. A well-calibrated model with a bad threshold is still a bad decision
rule, and vice versa. This project now does both. Task 10's reliability diagrams and
Brier score are how you'd *measure* the first half — which is why that entry pairs
naturally with this work.

## B.5 What a negative R² actually tells you

**The concept.** R² = 1 − (SS_residual / SS_total). The denominator is the error of the
dumbest possible model: **always predict the mean**. So R² is not "correlation squared"
and it is not bounded below by zero — on data the model did not fit, it can go arbitrarily
negative.

In plain terms: **R² < 0 means the model does worse than guessing the average.**

That is not an exotic failure. It is what extrapolation failure looks like: a model fitted
in one regime, evaluated in another, confidently producing predictions that are worse than
no model at all.

**What it looked like here.** Task 6's `decision_tree` won on CV MAE (1.1898, CV R²
+0.34) and scored test R² **−0.1669**. Not a bug — the holdout is the *closing laps of a
single race*, where fuel load is low and tyres are old, a regime under-represented in
training. Meanwhile `svr` scored test R² **+0.3023** and was not selected, because
selection looked only at CV.

**What we did.** Added a **generalisation guard** to `ml/selection.py`: refuse to select a
negative-R² model when a positive-R² candidate exists, and record *why* the CV ranking was
overridden as a named warning in the selection report.

| | Before | After |
|---|---|---|
| Selected | decision_tree | **svr** |
| Test R² | **−0.1669** | **+0.3023** |
| Test MAE | 0.8673 | **0.7815** |
| CV MAE | 1.1898 | 1.3813 |

The CV MAE is *worse*, and that is the trade made explicitly.

> **Likely question: "isn't selecting on the test set exactly the leakage you warned
> about?"** A fair challenge, and the design answers it narrowly. The guard does not
> *rank* by test R² — it uses it as a single binary veto ("does this model beat the
> mean?"), and among the survivors the original CV ranking still decides. A test asserts
> this: given a broken CV winner, a positive-R² model with better CV MAE, and another with
> far better test R², it must pick the first of those two. There is real tension here —
> any use of the holdout costs some independence — but a *fully* independent selection that
> ships a model known to be worse than the mean is not the safer option. The honest framing
> is that CV and holdout disagreeing this badly is itself the finding, and the right fix is
> more races (Appendix B.6), not a cleverer rule.

## B.6 A methodological note: what your holdout is actually asking

This is the thread running through all of Phase 2, and the most transferable idea in it.

`chronological_holdout` reserves the last 20% of **laps within one race**. That is a
legitimate split, but it answers a specific question: *can the model extrapolate to the
end of a race it has already seen most of?*

The question that matters for deployment is different: *can it predict a race it has never
seen?* A lap-level split cannot answer that, because track, weather and tyre allocation are
constant across the boundary — the model has seen those conditions, just at earlier laps.

This one distinction explains three separate symptoms we spent the phase chasing:
- the negative test R² (late-race fuel/tyre regime is under-represented in training);
- the 1-positive holdout that makes classification metrics uninterpretable (pit events
  cluster early-to-mid race, so the tail has almost none);
- CV and holdout disagreeing on which model is best.

They are not three bugs. They are one evaluation design, seen from three angles.

We added `race_level_holdout()` for the correct split and deliberately **did not** claim
the entry as closed: the dataset is a single session, so the function raises rather than
degrading silently, and the multi-session fetch is still open. Building infrastructure and
being honest that it is unexercised is a better outcome than a split that technically runs
and answers nothing.

> **Likely question: "what would you do first with more time?"** Fetch five to ten more
> 2023 sessions and re-run with the race-level holdout. It addresses the negative R², the
> uninterpretable classification metrics, and the CV/holdout disagreement simultaneously —
> because all three are the same problem.
