"""
app.api.routers.xai
===================

Task 8 — Explainable AI endpoints.

Every response is read from the committed Task 8 artifact
(``artifacts/metadata/xai_results.json``), which was produced by
``app.intelligence.xai.pipeline`` — the same code path the build script uses.
An explanation shown in the dashboard is therefore the same explanation the
report contains, not a separately-computed lookalike.

If Task 8 has not been run these endpoints return 404 with the command to run,
rather than an empty structure that would read as "no important features".
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.intelligence.xai import pipeline as xai_pipeline
from app.intelligence.xai import trust as trust_mod
from app.intelligence.xai.loading import ExplainerUnavailableError

router = APIRouter(prefix="/api/xai", tags=["explainable-ai"])

_TARGETS = ("target_laptime", "target_pit_next_lap")


def _results() -> dict:
    try:
        return xai_pipeline.load_results()
    except ExplainerUnavailableError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _target(target: str) -> dict:
    if target not in _TARGETS:
        raise HTTPException(
            status_code=422,
            detail=f"target must be one of {list(_TARGETS)}",
        )
    data = _results()
    entry = data.get("targets", {}).get(target)
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail=f"No Task 8 results for {target!r}. Run scripts/build_all.py.",
        )
    return entry


@router.get("/summary")
def get_summary():
    """One-glance status: which targets are explained, which model each
    explanation covers, and the headline fairness and trust numbers."""
    data = _results()
    return {
        "generated_at": data.get("generated_at"),
        "dataset_source": data.get("dataset_source"),
        "targets": {
            t: {
                "task": r["task"],
                "classical_model_explained": r["classical_name"],
                "n_features": len(r["features"]),
                "n_identity_features": len(r["identity_features"]),
                "identity_attribution_share": r["fairness"]["identity_attribution_share"],
                "concentration_ratio": r["fairness"]["concentration_ratio"],
                "trust": r["trust_summary"],
                "explained_rows": list(r["examples"].keys()),
            }
            for t, r in data.get("targets", {}).items()
        },
    }


@router.get("/feature-importance")
def get_feature_importance(target: str = Query(..., description="target_laptime | target_pit_next_lap")):
    """Permutation importance for both model families.

    Permutation importance is used rather than a model's native attribute
    because it is the only method that means the same thing for a tree
    ensemble and a neural network, so the two columns are comparable.
    """
    r = _target(target)
    return {
        "target": target,
        "method": "permutation importance (model-agnostic)",
        "deep_network": r["importance"]["dnn"],
        "classical": {"model": r["classical_name"], "importance": r["importance"]["classical"]},
        "comparison": r["importance_comparison"],
        "figure": r["figures"].get("importance"),
    }


@router.get("/shap")
def get_shap(target: str = Query(...), row: str | None = Query(None, description="explained-row label")):
    """Global SHAP ranking for both families, plus per-row attributions."""
    r = _target(target)
    payload = {
        "target": target,
        "deep_network": r["shap"]["dnn"],
        "classical": {**r["shap"]["classical"], "model": r["classical_name"]},
        "figure": r["figures"].get("shap_summary_dnn"),
        "available_rows": list(r["examples"].keys()),
    }
    if row is not None:
        ex = r["examples"].get(row)
        if ex is None:
            raise HTTPException(
                status_code=404,
                detail=f"No explained row {row!r} for {target}. Available: {list(r['examples'])}",
            )
        payload["row"] = {
            "label": row,
            "row_index": ex["row_index"],
            "lap": ex["lap"],
            "prediction": ex["dnn_prediction"],
            "attributions": ex["shap_dnn"],
            "top3": ex["shap_top3"],
            "figure": ex["figures"].get("shap_waterfall"),
        }
    return payload


@router.get("/lime")
def get_lime(target: str = Query(...), row: str | None = Query(None)):
    """LIME local surrogates for the same rows SHAP explains.

    ``local_r2`` is the diagnostic that matters: a low value means a straight
    line is a poor stand-in for the model near that row, so the explanation
    should be discounted however confident it looks.
    """
    r = _target(target)
    rows = {
        label: {
            "row_index": ex["row_index"],
            "lap": ex["lap"],
            "local_r2": ex["lime"]["local_r2"],
            "num_samples": ex["lime"]["num_samples"],
            "contributions": ex["lime"]["contributions"],
            "top3": ex["lime_top3"],
            "shap_top3": ex["shap_top3"],
            "agreement_jaccard": ex["trust"]["components"]["explanation_stability"],
            "figure": ex["figures"].get("lime"),
        }
        for label, ex in r["examples"].items()
    }
    if row is not None:
        if row not in rows:
            raise HTTPException(status_code=404, detail=f"No explained row {row!r}. Available: {list(rows)}")
        return {"target": target, "method": "LIME local linear surrogate", "row": {"label": row, **rows[row]}}
    return {"target": target, "method": "LIME local linear surrogate", "rows": rows}


@router.get("/counterfactual")
def get_counterfactual(target: str = Query(...), row: str | None = Query(None)):
    """What would have to change for the recommendation to flip.

    Two methods: a single-feature bisection scan (actionable — "how many more
    laps on these tyres?") and DiCE whole-row alternatives (diverse, less
    actionable). Where no counterfactual exists inside the observed range that
    is reported as ``reachable: false`` with the range searched.
    """
    r = _target(target)
    rows = {
        label: {
            "row_index": ex["row_index"],
            "lap": ex["lap"],
            "scan": ex["counterfactual"],
            "sentence": ex["counterfactual_sentence"],
            "figure": ex["figures"].get("counterfactual"),
        }
        for label, ex in r["examples"].items()
    }
    if row is not None:
        if row not in rows:
            raise HTTPException(status_code=404, detail=f"No explained row {row!r}. Available: {list(rows)}")
        return {"target": target, "row": {"label": row, **rows[row]}, "dice": r.get("dice")}
    return {"target": target, "rows": rows, "dice": r.get("dice")}


@router.get("/trust-score")
def get_trust_score(target: str = Query(...), row: str | None = Query(None)):
    """The trust score for each explained prediction, with its components and
    the formula that produced it."""
    r = _target(target)
    rows = {
        label: {
            "row_index": ex["row_index"],
            "lap": ex["lap"],
            **ex["trust"],
            "narrative": ex["narrative"],
        }
        for label, ex in r["examples"].items()
    }
    payload = {
        "target": target,
        "formula": "0.40*confidence + 0.30*model_agreement + 0.30*explanation_stability",
        "weights": dict(trust_mod.WEIGHTS),
        "bands": {
            "HIGH": ">= 0.75 - safe to act on",
            "MODERATE": "0.50-0.75 - one input among several; read the factors first",
            "LOW": "0.25-0.50 - a prompt to look at the evidence, not a recommendation",
            "DO NOT ACT": "< 0.25 - no more information than a coin flip",
        },
        "summary": r["trust_summary"],
    }
    if row is not None:
        if row not in rows:
            raise HTTPException(status_code=404, detail=f"No explained row {row!r}. Available: {list(rows)}")
        payload["row"] = {"label": row, **rows[row]}
    else:
        payload["rows"] = rows
    return payload


@router.get("/fairness")
def get_fairness(target: str | None = Query(None)):
    """Is the model predicting from race state, or from who is driving?

    Measured as identity features' share of total mean-|SHAP| attribution,
    against the share an even spread across all features would give.
    """
    data = _results()
    targets = data.get("targets", {})
    if target is not None:
        r = _target(target)
        return {"target": target, **r["fairness"], "figure": r["figures"].get("fairness")}
    return {
        t: {**r["fairness"], "figure": r["figures"].get("fairness")}
        for t, r in targets.items()
    }


@router.get("/explanation")
def get_explanation(target: str = Query(...), row: str | None = Query(None)):
    """The race-engineer view: a plain-English sentence per prediction, with
    its trust band and top factors. This is what the dashboard shows."""
    r = _target(target)
    rows = {
        label: {
            "row_index": ex["row_index"],
            "lap": ex["lap"],
            "prediction": ex["dnn_prediction"],
            "classical_prediction": ex["classical_prediction"],
            "narrative": ex["narrative"],
            "counterfactual_sentence": ex["counterfactual_sentence"],
            "trust_score": ex["trust"]["trust_score"],
            "trust_band": ex["trust"]["band"],
            "top_factors": ex["shap_dnn"][:3],
        }
        for label, ex in r["examples"].items()
    }
    if row is not None:
        if row not in rows:
            raise HTTPException(status_code=404, detail=f"No explained row {row!r}. Available: {list(rows)}")
        return {"target": target, "row": {"label": row, **rows[row]}}
    return {"target": target, "classical_model_explained": r["classical_name"], "rows": rows}
