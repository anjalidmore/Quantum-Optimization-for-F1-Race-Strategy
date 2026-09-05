# FLOW — how the practicals build on each other

Read them in numerical order. That order is not arbitrary: each practical
consumes a concept, a vocabulary or a file established by an earlier one.

```text
practical01  Knowledge Representation
     │        defines the DOMAIN VOCABULARY — 61 entities, 29 relationships,
     │        an OWL 2 ontology and a knowledge graph
     │
     │  vocabulary (fact keys, entity/attribute names)
     ▼
practical02  Rule-Based Expert System
     │        reasons over that vocabulary — 32 production rules, forward and
     │        backward chaining, HOW/WHY explanation traces
     │
     │  domain cost model (tyre degradation, fuel burn, pit loss)
     ▼
practical03  State-Space Search
     │        re-casts the same strategy decision as optimisation instead of
     │        inference — shortest path over (lap, compound, age, stops, fuel)
     │
     │  (independent of 01–03 at the code level;
     │   shares only the domain schemas)
     ▼
practical04  Data Engineering & EDA
     │        turns raw Kaggle/FastF1-shaped CSVs into a clean, audited,
     │        analysed per-lap dataset
     │
     │  FILE DEPENDENCY ↓
     │  practical04/outputs/clean/fastf1_laps_clean.csv
     ▼
practical05  Feature Engineering & Selection
              engineers causal features and prunes them down to a modelling
              matrix + an explicit preprocessing contract

              FILE OUTPUT ↓
              practical05/outputs/f1_features_selected.csv
              practical05/outputs/feature_metadata.json
                          │
                          ├──────────────► Task 6 (Machine Learning), `main` branch
                          │
                          ▼
practical06  Deep Learning
     │        Keras MLPs (one per target) trained on the SAME folds, compared
     │        against classical baselines on the SAME untouched test set
     │
     │  MODEL DEPENDENCY ↓
     │  practical06/outputs/models/target_*.keras (+ scalers, + feature spec)
     ▼
practical07  Explainable AI
              SHAP + LIME + counterfactuals + trust score + fairness, over
              both the deep network and its classical counterpart
```

## Two different kinds of dependency

It matters which is which, because only one of them will break if you run
things out of order.

**Conceptual (01 → 02 → 03).** Practical 02 reasons about tyres, weather and
track status using the vocabulary practical 01 formalised, and practical 03
reuses the same domain cost model. But there is **no import and no file
handoff** between them. Each has its own `src/` package and its own
`requirements.txt`, and each runs standalone. You can run practical 03 without
ever having run practical 01.

**Concrete (04 → 05 → 06 → 07).** These are real. Practical 05 reads
`practical04/outputs/clean/fastf1_laps_clean.csv` from disk, and imports
practical 04's `f1data.schemas` via `sys.path` so that column semantics are
defined once rather than twice. Practical 05's notebook locates practical 04 by
walking up the directory tree looking for a `practical04/` folder, so both must
stay siblings under `practicals/`. If the cleaned CSV is missing, the notebook
shells out to `practical04.py` to regenerate it before continuing.

## The intellectual arc

The five practicals deliberately attack the *same problem* — when should this
car pit, and onto what tyre — with four different paradigms:

1. **Represent it** (01). Before you can reason, you need a formal vocabulary.
   Everything is generated from one declarative source of truth
   (`src/f1kr/schema.py`) so documentation cannot drift from code.
2. **Reason about it symbolically** (02). Encode expert knowledge as explicit
   rules. The payoff is *explainability* — every recommendation carries an audit
   trail. The cost is that the knowledge must be hand-authored.
3. **Search for the optimum** (03). Drop the hand-authored knowledge and define
   a cost model instead, then let A\* find the provably optimal pit strategy.
   The payoff is optimality; the cost is that the model must be correct.
4. **Learn it from data** (04 → 05). Stop hand-specifying anything. Clean real
   race data, engineer causal features, and let Task 6 fit the model. The payoff
   is that it learns what nobody encoded; the cost is leakage, drift and
   opacity — which is exactly why practicals 04 and 05 spend so much effort on
   audit trails, leakage exclusion and an explicit preprocessing contract.

Practical 05's four-stage selection funnel is the hinge between the symbolic
half and the statistical half: near-zero variance → correlation → VIF →
importance with fold stability, with the final feature count chosen by
cross-validated error under a parsimony rule.

5. **Go deeper** (06). Replace the tree ensemble with a neural network and test whether
   the extra capacity buys anything. On this data it does for lap time (MAE 0.1584 vs
   the forest's 0.2298) and does not for the pit decision (CV PR-AUC 0.0694 against a
   0.048 chance baseline, on 16 positive examples). The payoff is representational power;
   the cost is that it needs data this project does not have.
6. **Open the box** (07). A network that outperforms a forest but cannot say why is
   worse than useless on a pit wall. Practical 07 attributes every prediction to named
   race-state factors, cross-checks two independent explanation methods against each
   other, and scores how much the recommendation should be trusted. This closes the arc:
   the symbolic engines were explainable *by construction*; the statistical ones have to
   be made explainable *after the fact*, and that is a strictly harder problem.

## What is real vs. what is synthetic

Being precise about this matters more than it flatters:

* **Real:** all pipeline code, cleaning logic, inference, search algorithms,
  feature engineering and selection. The loader in `f1data/schemas.py` encodes
  the *actual* column names of the Kaggle (Ergast-derived) and FastF1 datasets,
  so it runs unchanged on a genuine download.
* **Synthetic by default:** the input *data*. Practical 04 generates realistic,
  internally-consistent sample data matching those exact schemas — including
  deliberately injected data-quality defects — so every practical runs offline
  with no multi-gigabyte download. Point `practical04.py --data-dir` at real
  CSVs and the whole chain works unchanged.
* **Consequence to keep honest:** the synthetic session has a near-deterministic
  pit schedule, which inflates practical 05's pit-decision classification AUC.
  The notebook says so in its own metadata, and so does this document.
* **A second consequence, surfaced by practical 06:** every pit event in the synthetic
  matrix occurs at lap 18, 27 or 36. The chronological holdout is laps 46–55, so it
  contains **zero** positive examples — ROC-AUC and PR-AUC are mathematically undefined
  there for the deep network *and* every classical baseline alike. Practical 06 reports
  this rather than substituting a number, and falls back to the CV folds for its verdict.
* **A third, surfaced by practical 07:** practical 05's synthetic contract keeps only
  **1** driver/team identity feature per target, so the fairness assessment finds
  identity carrying 2–4% of attribution — well below an even spread. The real-data
  contract on `proj-mode` keeps **28 of 45**, which is where that assessment matters.

## A note on reproducibility

Re-running the full chain on Python 3.14.6 with scikit-learn 1.9.0 reproduced
practical 04's cleaned CSV and practical 05's exported feature matrix
**byte-identically**. Two things did drift:

* the recorded cross-validation scores in `feature_metadata.json`
  (MAE 0.2413 → 0.2417, R² 0.5361 → 0.5343, AUC 0.9934 → 1.0), and
* the tie-broken ordering of two features in the classification list.

The feature *matrix* is stable; only the scores computed over it moved. The
cause is that every `requirements.txt` in this branch uses `>=` constraints, so
a fresh install picks up a newer scikit-learn than the one the committed
metadata was produced with. Pinning exact versions would remove this drift.
