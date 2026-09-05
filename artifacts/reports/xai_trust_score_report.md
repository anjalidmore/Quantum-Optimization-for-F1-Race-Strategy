# Task 8 - Trust Score Report

_Generated 2026-09-05 06:50 UTC._

## The formula

```
trust = 0.40 * confidence
      + 0.30 * model_agreement
      + 0.30 * explanation_stability
```

| Component | Definition | Why it is in the score |
|---|---|---|
| `confidence` | Classification: `2 * abs(p - 0.5)`. Regression: `1 - min(1, gap / target_std)`. | A prediction sitting on the decision boundary is unusable however well explained. |
| `model_agreement` | `1 - abs(p_dnn - p_classical)` | Two model families trained on identical folds agreeing is real evidence. One disagreeing means at least one is wrong and you cannot tell which. |
| `explanation_stability` | Jaccard overlap of SHAP's and LIME's top-3 features | If two established explanation methods disagree about *why*, the explanation you would show the engineer is not trustworthy even if the prediction is right. |

**Weights:** {'confidence': 0.4, 'model_agreement': 0.3, 'explanation_stability': 0.3}. These are a judgement, not a derivation - confidence carries
the largest share because boundary predictions are useless regardless of explanation
quality, while agreement and stability are weighted equally (a wrong-but-explained
and a right-but-unexplainable prediction are both unsafe to act on). They are exposed
as `f1xai.trust.WEIGHTS` so they can be challenged and changed.

## What the bands mean to a race engineer

| Score | Band | Action |
|---|---|---|
| >= 0.75 | **HIGH** | Both families agree, prediction far from the boundary, SHAP and LIME tell the same story. Safe to act on. |
| 0.50 - 0.75 | **MODERATE** | One input among several. Read the SHAP factors before acting. |
| 0.25 - 0.50 | **LOW** | A prompt to look at the evidence, not a recommendation. |
| < 0.25 | **DO NOT ACT** | Carries no more information than a coin flip. |

## Worked examples

### target_laptime

| Case | Test row | Confidence | Agreement | Stability | **Trust** | Band |
|---|---:|---:|---:|---:|---:|---|
| fastest predicted lap | 89 | 0.101 | 0.101 | 0.500 | **0.221** | DO NOT ACT |
| median predicted lap | 92 | 0.851 | 0.851 | 0.500 | **0.745** | MODERATE |
| slowest predicted lap | 178 | 0.967 | 0.967 | 0.500 | **0.827** | HIGH |

**Across all 3 explained rows:** mean 0.598, range 0.221-0.827. Bands: {'HIGH': 1, 'MODERATE': 1, 'LOW': 0, 'DO NOT ACT': 1}.

---

### target_pit_next_lap

| Case | Test row | Confidence | Agreement | Stability | **Trust** | Band |
|---|---:|---:|---:|---:|---:|---|
| lowest pit probability | 43 | 1.000 | 0.977 | 0.500 | **0.843** | HIGH |
| closest to decision boundary | 34 | 0.577 | 0.868 | 0.500 | **0.641** | MODERATE |

**Across all 2 explained rows:** mean 0.742, range 0.641-0.843. Bands: {'HIGH': 1, 'MODERATE': 1, 'LOW': 0, 'DO NOT ACT': 0}.

---

