"""
app.intelligence.xai.trust
==========================

A defined, computable trust score for a single recommendation.

The problem it solves: a model always returns *a* number. A race engineer
needs to know when that number deserves to be acted on. Three independent
things can go wrong, and the score measures all three:

    trust = 0.40 * confidence
          + 0.30 * model_agreement
          + 0.30 * explanation_stability

**confidence** - how far the prediction is from the decision boundary.
    Classification: ``2 * |p - 0.5|``, so p=0.5 scores 0 and p=0.99 scores
    0.98. Regression: how small this row's expected error is relative to the
    spread of the target, ``1 - min(1, |residual_proxy| / target_std)``, using
    the disagreement between the two models as the residual proxy (the true
    error is unknown at prediction time - that is the whole point).

**model_agreement** - do the deep network and the classical model say the
    same thing? ``1 - |p_dnn - p_classical|`` for classification, and a
    normalised absolute difference for regression. Two model families trained
    on identical folds agreeing is genuine evidence; one disagreeing with the
    other means at least one is wrong and you cannot tell which.

**explanation_stability** - do SHAP and LIME name the same drivers of this
    prediction? Jaccard overlap of their top-3 feature sets. If two
    established explanation methods disagree about *why* the model decided
    something, the explanation you would show the engineer is not trustworthy
    even when the prediction happens to be right.

**Why these weights.** Confidence gets the largest share because a prediction
sitting on the decision boundary is unusable regardless of how well it is
explained. Agreement and stability are weighted equally: a wrong-but-explained
prediction and a right-but-unexplainable one are both unsafe to act on. The
weights are a judgement, not a derivation, and they are exposed as
``WEIGHTS`` so they can be challenged and changed.

The bands in ``interpret`` are deliberately conservative: this is a decision
aid, and a score in the middle should send the engineer to the evidence rather
than to the recommendation.
"""
from __future__ import annotations

import numpy as np

WEIGHTS = {"confidence": 0.40, "model_agreement": 0.30, "explanation_stability": 0.30}


def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def compute(
    *,
    task: str,
    dnn_prediction: float,
    classical_prediction: float,
    shap_top: list[str],
    lime_top: list[str],
    target_std: float | None = None,
) -> dict:
    if task == "classification":
        confidence = float(2.0 * abs(dnn_prediction - 0.5))
        agreement = float(1.0 - min(1.0, abs(dnn_prediction - classical_prediction)))
    else:
        if not target_std or target_std <= 0:
            raise ValueError("target_std is required (and must be > 0) for regression trust")
        gap = abs(dnn_prediction - classical_prediction)
        agreement = float(1.0 - min(1.0, gap / target_std))
        confidence = agreement  # the models' disagreement is the only error proxy available

    stability = float(jaccard(shap_top, lime_top))

    components = {
        "confidence": round(confidence, 4),
        "model_agreement": round(agreement, 4),
        "explanation_stability": round(stability, 4),
    }
    score = sum(WEIGHTS[k] * v for k, v in components.items())

    return {
        "trust_score": round(float(score), 4),
        "components": components,
        "weights": dict(WEIGHTS),
        "band": interpret(score),
        "inputs": {
            "dnn_prediction": float(dnn_prediction),
            "classical_prediction": float(classical_prediction),
            "shap_top3": list(shap_top),
            "lime_top3": list(lime_top),
            "target_std": float(target_std) if target_std else None,
        },
    }


def interpret(score: float) -> dict:
    if score >= 0.75:
        return {"label": "HIGH",
                "meaning": "Both model families agree, the prediction is far from the boundary, "
                           "and SHAP and LIME tell the same story. Safe to act on."}
    if score >= 0.50:
        return {"label": "MODERATE",
                "meaning": "Usable as one input among several. Read the SHAP factors before acting; "
                           "one of the three components is weak."}
    if score >= 0.25:
        return {"label": "LOW",
                "meaning": "Treat as a prompt to look at the evidence, not as a recommendation. "
                           "The models disagree, or the explanations do."}
    return {"label": "DO NOT ACT",
            "meaning": "The prediction sits on the decision boundary and/or the two model families "
                       "contradict each other. This carries no more information than a coin flip."}


def summarise(scores: list[dict]) -> dict:
    vals = [s["trust_score"] for s in scores]
    return {
        "n": len(vals),
        "mean": round(float(np.mean(vals)), 4) if vals else None,
        "min": round(float(np.min(vals)), 4) if vals else None,
        "max": round(float(np.max(vals)), 4) if vals else None,
        "bands": {b: sum(1 for s in scores if s["band"]["label"] == b)
                  for b in ("HIGH", "MODERATE", "LOW", "DO NOT ACT")},
    }
