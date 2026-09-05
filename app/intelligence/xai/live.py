"""
app.intelligence.xai.live
=========================

On-demand explanation of a **single, arbitrary** feature row — as opposed to
``pipeline.py``, which explains a fixed set of representative test rows and
writes the committed artifacts.

This is what ``POST /api/strategy/predict?explain=true`` calls. The row it
explains is the one ``feature_approximation.build_feature_row`` just built from
the race state the caller sent, so the explanation is of *this* recommendation
rather than of a stored example that merely resembles it.

Two deliberate differences from the batch path, both about latency:

* ``KernelExplainer`` runs with a smaller ``nsamples`` budget and a smaller
  k-means background. A strategy call happens while someone is waiting; the
  committed reports do not. The budget actually used is returned in the
  response, so a reader can see the explanation is coarser than the report's.
* LIME is not run. Its only role in the trust score is the
  ``explanation_stability`` term, and fitting a 2000-perturbation surrogate per
  request is not worth ~1s of pit-wall latency. The trust score is therefore
  computed from its other two components, **renormalised**, and the response
  says so explicitly rather than silently reporting a differently-defined score
  under the same name.
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
    shap_top3 = [f["feature"] for f in factors[:3]]

    # LIME is skipped here (see the module docstring), so explanation_stability
    # has no input. Renormalise the remaining weights rather than scoring a
    # missing component as zero, which would depress every live trust score by
    # a constant 0.30 and make the bands mean something different.
    target_std = float(np.std(t.y_train)) if t.task == "regression" else None
    full = trust.compute(
        task=t.task, dnn_prediction=p_dnn, classical_prediction=p_cls,
        shap_top=shap_top3, lime_top=shap_top3, target_std=target_std,
    )
    w = trust.WEIGHTS
    denom = w["confidence"] + w["model_agreement"]
    score = (
        w["confidence"] * full["components"]["confidence"]
        + w["model_agreement"] * full["components"]["model_agreement"]
    ) / denom
    band = trust.interpret(score)

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
        "trust_components": {
            "confidence": full["components"]["confidence"],
            "model_agreement": full["components"]["model_agreement"],
        },
        "method": {
            "explainer": shap_res["explainer"],
            "exact": shap_res["exact"],
            "nsamples": LIVE_NSAMPLES,
            "background_k": shap_res["background_k"],
            "note": (
                "Computed live at a reduced sampling budget so the request stays "
                "interactive; the committed reports use a larger budget. LIME is not run "
                "here, so the trust score uses confidence and model agreement only, "
                "renormalised — it is not directly comparable to the three-component "
                "score in the Task 8 reports."
            ),
        },
    }
