# Task 8 - Trust Score Report

_Generated 2026-10-18 17:17 UTC._

A project-defined score, not a validated or standardised measure: the weights are a
judgement, and it has not been checked against a race engineer's own assessment.

## The formula

```
trust = ( 0.35 * confidence
        + 0.25 * model_agreement
        + 0.20 * explanation_stability
        + 0.20 * input_validity )
        / (sum of the weights of the components that apply)
```

| Component | Weight | Definition | Why it is in the score |
|---|---:|---|---|
| `confidence` | 0.35 | Classification only: distance from the model's **tuned** decision threshold *t* (never 0.5), scaled to 0 at *t* and 1 at certainty — `(p - t) / (1 - t)` above *t*, `(t - p) / t` below it. Not defined for lap time, which has no decision boundary. | A prediction sitting on the decision point is unusable however well explained. |
| `model_agreement` | 0.25 | Classification: `1 - abs(p_dnn - p_classical)`. Regression: `1 - min(1, abs(dnn - classical) / target_std)`. | Two model families trained on identical folds agreeing is real evidence; disagreement means at least one is wrong. |
| `explanation_stability` | 0.20 | Jaccard overlap of SHAP's and LIME's top-3 features. | If two explanation methods disagree about *why*, the explanation shown to the engineer is not trustworthy even if the prediction is right. |
| `input_validity` | 0.20 | Share of the row's inputs inside the training data's 1st–99th percentile. | Below 1 means the model is being asked about race states it did not learn from. |

**A component that does not apply is left out and the other weights are renormalised** — never scored as zero. For lap time `confidence` never applies, so its score is
`(0.25·model_agreement + 0.20·explanation_stability + 0.20·input_validity) / 0.65`. The live API skips LIME, so `explanation_stability` is left out there too.

**Weights:** `{'confidence': 0.35, 'model_agreement': 0.25, 'explanation_stability': 0.2, 'input_validity': 0.2}` — `app.intelligence.xai.trust.WEIGHTS`, exposed so they can be challenged and changed.

## What the bands mean to a race engineer

| Score | Band | Meaning |
|---|---|---|
| >= 0.75 | **HIGH** | The two model families agree, the prediction is clear of the decision point, the explanations concur and the inputs are familiar. The strongest support this project-defined score can give - still not a guarantee. |
| 0.50 - 0.75 | **MODERATE** | Usable as one input among several. Read the SHAP factors first; at least one component is weak. |
| 0.25 - 0.50 | **LOW** | A prompt to look at the evidence, not a recommendation. The models disagree, the explanations do, or the inputs are unusual. |
| < 0.25 | **DO NOT ACT** | The prediction sits on the decision point and/or the model families contradict each other. |

## Worked examples

### target_laptime

Regression: `confidence` does not apply, so every score below is renormalised over the other three components.

| Case | Test row | Confidence | Agreement | Stability | Input validity | **Trust** | Band |
|---|---:|---:|---:|---:|---:|---:|---|
| fastest predicted lap | 43 | n/a | 0.809 | 0.500 | 1.000 | **0.773** | HIGH |
| median predicted lap | 6 | n/a | 0.761 | 0.500 | 0.978 | **0.747** | MODERATE |
| slowest predicted lap | 178 | n/a | 0.327 | 0.200 | 1.000 | **0.495** | LOW |
| freshest tyres | 86 | n/a | 0.499 | 0.500 | 0.978 | **0.647** | MODERATE |
| oldest tyres | 42 | n/a | 0.856 | 0.200 | 0.956 | **0.685** | MODERATE |

**Across all 5 explained rows:** mean 0.669, range 0.495-0.773. Bands: {'HIGH': 1, 'MODERATE': 3, 'LOW': 1, 'DO NOT ACT': 0}.

---

### target_pit_next_lap

Confidence is measured from this model's tuned decision threshold, **0.1524**.

| Case | Test row | Confidence | Agreement | Stability | Input validity | **Trust** | Band |
|---|---:|---:|---:|---:|---:|---:|---|
| lowest pit probability | 89 | 0.984 | 0.906 | 1.000 | 1.000 | **0.971** | HIGH |
| closest to decision boundary | 171 | 0.156 | 0.725 | 0.200 | 0.750 | **0.426** | LOW |
| actual pit lap | 177 | 0.587 | 0.484 | 0.500 | 0.625 | **0.551** | MODERATE |

**Across all 3 explained rows:** mean 0.649, range 0.426-0.971. Bands: {'HIGH': 1, 'MODERATE': 1, 'LOW': 1, 'DO NOT ACT': 0}.

---

