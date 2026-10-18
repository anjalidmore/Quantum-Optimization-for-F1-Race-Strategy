"""
app.intelligence.xai.live
=========================

**Why this exists:** a reduced-budget version of the batch SHAP in
``pipeline.py``, for explaining one arbitrary race state while a caller waits.

``pipeline.py`` explains a fixed set of representative test rows and writes the
committed artifacts. This module explains the row
``feature_approximation.build_feature_row`` just built from a live request, so
``POST /api/strategy/predict?explain=true`` explains *that* recommendation
rather than a stored example resembling it. Task 9's strategy-report generator
needs the same per-race-state path.

Two differences from the batch path, both latency, both declared in the
response rather than hidden: ``KernelExplainer`` runs with a smaller
``nsamples`` and background, and LIME is skipped — a 2,000-perturbation
surrogate per request is not worth the wait, so the trust score renormalises
over its remaining components instead of quietly redefining itself.
"""
from __future__ import annotations

import logging

import numpy as np

from app.intelligence.xai import narrative, shap_analysis, trust
from app.intelligence.xai.loading import ExplainerUnavailableError, load_target

log = logging.getLogger(__name__)

# Small enough to keep a strategy call interactive; recorded in the response.
LIVE_NSAMPLES = 60
LIVE_BACKGROUND_K = 15

_CACHE: dict[str, object] = {}


def _target_bundle(target: str):
    """Loading a target reads two models off disk; cache it per process so the
    second explained request in a session is fast."""
    if target not in _CACHE:
        _CACHE[target] = load_target(target)
    return _CACHE[target]


def predict_point(target: str, row: dict[str, float]) -> dict:
    """Task 7 DNN point prediction only — no SHAP, no LIME.

    Used by the strategy pipeline's DL stage, which needs a fast per-request
    prediction on every call (unlike ``explain_feature_row``, which a caller
    opts into). Shares ``_target_bundle``'s cache, so a later ``explain=true``
    call on the same target pays no extra model-load cost.
    """
    try:
        t = _target_bundle(target)
    except ExplainerUnavailableError as exc:
        return {"available": False, "reason": str(exc)}

    missing = [f for f in t.features if f not in row]
    if missing:
        return {
            "available": False,
            "reason": f"feature row is missing {len(missing)} feature(s) the model expects: {missing[:5]}",
        }

    X = np.array([[float(row[f]) for f in t.features]], dtype="float32")
    try:
        p_dnn = float(np.asarray(t.dnn_predict(X)).ravel()[0])
    except Exception as exc:  # pragma: no cover - inference failure guard
        log.warning("live DL point-prediction failed for %s: %s", target, exc)
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}

    out = {"available": True, "target": target, "model": "dnn_mlp", "deep_prediction": p_dnn}
    if t.task == "classification":
        threshold = t.decision_threshold or 0.5
        out["decision_threshold"] = threshold
        out["predicted_class"] = int(p_dnn >= threshold)
    return out


def explain_feature_row(target: str, row: dict[str, float]) -> dict:
    """Explain one caller-supplied feature row.

    ``row`` is the mapping ``feature_approximation.build_feature_row``
    produced. Returns SHAP factors, a trust score and a race-engineer sentence,
    or a structured ``available: False`` payload naming what is missing — never
    a fabricated explanation.
    """
    try:
        t = _target_bundle(target)
    except ExplainerUnavailableError as exc:
        return {"available": False, "reason": str(exc)}

    missing = [f for f in t.features if f not in row]
    if missing:
        return {
            "available": False,
            "reason": f"feature row is missing {len(missing)} feature(s) the model expects: {missing[:5]}",
        }

    X = np.array([[float(row[f]) for f in t.features]], dtype="float32")

    try:
        p_dnn = float(np.asarray(t.dnn_predict(X)).ravel()[0])
        p_cls = float(np.asarray(t.classical_predict(X)).ravel()[0])
        shap_res = shap_analysis.kernel_shap(
            t.dnn_predict, t.X_train, X, t.features,
            nsamples=LIVE_NSAMPLES, k=LIVE_BACKGROUND_K,
        )
    except Exception as exc:  # pragma: no cover - explainer failure guard
        log.warning("live explanation failed for %s: %s", target, exc)
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}

    factors = shap_analysis.explain_row(shap_res, 0, top_n=min(6, len(t.features)))

    # LIME is skipped here (see the module docstring), so explanation_stability
    # has no input; trust.compute leaves it out and renormalises the remaining
    # weights rather than scoring a missing component as zero.
    target_std = float(np.std(t.y_train)) if t.task == "regression" else None
    full = trust.compute(
        task=t.task, dnn_prediction=p_dnn, classical_prediction=p_cls,
        shap_top=None, lime_top=None, target_std=target_std,
        threshold=t.decision_threshold or 0.5,
        input_validity_share=trust.input_validity(X[0], t.X_train),
    )
    score = full["trust_score"]
    band = full["band"]

    values = {f: float(row[f]) for f in t.features}
    sentence = (
        narrative.pit_decision_sentence(p_dnn, factors, values, band["label"])
        if t.task == "classification"
        else narrative.laptime_sentence(
            p_dnn, factors, values, band["label"], base_value=shap_res["base_value"])
    )

    return {
        "available": True,
        "target": target,
        "deep_prediction": p_dnn,
        "classical_prediction": p_cls,
        "classical_model": t.classical_name,
        "shap_factors": factors,
        "narrative": sentence,
        "trust_score": round(float(score), 4),
        "trust_band": band,
        "trust_components": full["components"],
        "method": {
            "explainer": shap_res["explainer"],
            "exact": shap_res["exact"],
            "nsamples": LIVE_NSAMPLES,
            "background_k": shap_res["background_k"],
            "note": (
                "Computed live at a reduced sampling budget so the request stays "
                "interactive; the committed reports use a larger budget. LIME is not run "
                "here, so the trust score uses confidence, model agreement and input "
                "validity, renormalised — it omits the explanation-stability term the Task 8 "
                "reports include."
            ),
        },
    }
