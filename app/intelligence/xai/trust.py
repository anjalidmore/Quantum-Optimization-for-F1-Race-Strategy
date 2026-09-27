"""
app.intelligence.xai.trust
==========================

A defined, computable trust score for one prediction. **Project-defined: it is
not a validated or standardised measure**, its weights are a judgement rather
than a fit, and it has never been checked against a race engineer's own
assessment. It is a structured way to flag which predictions deserve a second
look.

    trust = Σ weight_k × component_k  /  Σ weight_k   (over available components)

**confidence** (0.35) — distance of the prediction from the decision point.
    Classification: measured from the model's *tuned* decision threshold t,
    scaled to 0 at t and 1 at certainty: ``(p - t)/(1 - t)`` above it,
    ``(t - p)/t`` below it. (It used to be measured from 0.5, which is not where
    this model decides.) Regression has no decision boundary and no
    independent confidence signal, so the component is left out for lap time
    and the others are renormalised. (It used to copy model_agreement, which
    silently gave agreement 60% of the weight under two names.)
**model_agreement** (0.25) — do the Task 7 network and Task 6's selected model
    say the same thing? ``1 - |p_dnn - p_classical|`` (classification), or the
    absolute difference scaled by the target's standard deviation (regression).
**explanation_stability** (0.20) — do SHAP and LIME name the same top-3 drivers
    of this prediction? Jaccard overlap of the two sets.
**input_validity** (0.20) — share of this row's inputs that lie inside the
    1st–99th percentile of the training data. Below 1 means the model is being
    asked about race states outside what it learned from.

A component whose input is unavailable (LIME is skipped in the live API path)
is left out and the remaining weights are renormalised, and the result says
which components were used — a missing component is never scored as zero.
"""
from __future__ import annotations

import numpy as np

WEIGHTS = {"confidence": 0.35, "model_agreement": 0.25,
           "explanation_stability": 0.20, "input_validity": 0.20}


def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def input_validity(row: np.ndarray, X_train: np.ndarray) -> float:
    """Share of features inside the training data's 1st–99th percentile band."""
    lo, hi = np.percentile(X_train, 1, axis=0), np.percentile(X_train, 99, axis=0)
    row = np.asarray(row, dtype=float).ravel()
    return float(np.mean((row >= lo) & (row <= hi)))


def confidence_from_threshold(p: float, threshold: float) -> float:
    if p >= threshold:
        return float((p - threshold) / (1.0 - threshold)) if threshold < 1 else 0.0
    return float((threshold - p) / threshold) if threshold > 0 else 0.0


def compute(
    *,
    task: str,
    dnn_prediction: float,
    classical_prediction: float,
    shap_top: list[str] | None = None,
    lime_top: list[str] | None = None,
    target_std: float | None = None,
    threshold: float = 0.5,
    input_validity_share: float | None = None,
) -> dict:
    if task == "classification":
        confidence = confidence_from_threshold(dnn_prediction, threshold)
        agreement = float(1.0 - min(1.0, abs(dnn_prediction - classical_prediction)))
    else:
        if not target_std or target_std <= 0:
            raise ValueError("target_std is required (and must be > 0) for regression trust")
        agreement = float(1.0 - min(1.0, abs(dnn_prediction - classical_prediction) / target_std))
        confidence = None  # no decision boundary; copying agreement here would count it twice

    components = {"model_agreement": agreement}
    if confidence is not None:
        components = {"confidence": confidence, **components}
    if shap_top is not None and lime_top is not None:
        components["explanation_stability"] = float(jaccard(shap_top, lime_top))
    if input_validity_share is not None:
        components["input_validity"] = float(input_validity_share)

    used = {k: WEIGHTS[k] for k in components}
    score = sum(used[k] * components[k] for k in components) / sum(used.values())

    return {
        "trust_score": round(float(score), 4),
        "components": {k: round(v, 4) for k, v in components.items()},
        "weights": dict(WEIGHTS),
        "components_used": list(components),
        "renormalised": set(components) != set(WEIGHTS),
        "band": interpret(score),
        "inputs": {
            "dnn_prediction": float(dnn_prediction),
            "classical_prediction": float(classical_prediction),
            "decision_threshold": float(threshold) if task == "classification" else None,
            "shap_top3": list(shap_top) if shap_top is not None else None,
            "lime_top3": list(lime_top) if lime_top is not None else None,
            "target_std": float(target_std) if target_std else None,
            "input_validity_share": (float(input_validity_share)
                                     if input_validity_share is not None else None),
        },
    }


def interpret(score: float) -> dict:
    if score >= 0.75:
        return {"label": "HIGH",
                "meaning": "The two model families agree, the prediction is clear of the decision "
                           "point, the explanations concur and the inputs are familiar. The strongest "
                           "support this project-defined score can give - still not a guarantee."}
    if score >= 0.50:
        return {"label": "MODERATE",
                "meaning": "Usable as one input among several. Read the SHAP factors first; at least "
                           "one component is weak."}
    if score >= 0.25:
        return {"label": "LOW",
                "meaning": "A prompt to look at the evidence, not a recommendation. The models "
                           "disagree, the explanations do, or the inputs are unusual."}
    return {"label": "DO NOT ACT",
            "meaning": "The prediction sits on the decision point and/or the model families "
                       "contradict each other."}


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
