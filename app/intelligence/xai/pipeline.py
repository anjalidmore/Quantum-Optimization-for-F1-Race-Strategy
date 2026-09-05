"""
app.intelligence.xai.pipeline
=============================

Task 8 orchestrator. Produces the committed explainability artifacts, and
exposes ``explain_target`` so the API can compute the same explanations
on demand from the same code path.

Nothing here trains a model. It explains Task 6's persisted pipelines and
Task 7's saved networks; if either is missing it raises
``ExplainerUnavailableError`` rather than substituting a stand-in.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app.core.paths import (
    ML_FIGURES_DIR,
    ML_REPORTS_DIR,
    PROCESSED_DATA_SOURCE_JSON,
    XAI_RESULTS_JSON,
    ensure_dirs,
)
from app.intelligence.xai import (
    counterfactual, fairness, importance, lime_analysis, loading,
    narrative, reports, shap_analysis, trust, visualize,
)

log = logging.getLogger(__name__)

TARGETS = ("target_laptime", "target_pit_next_lap")

# The one feature a race engineer can actually act on. The counterfactual scan
# moves this and holds the rest of the race state fixed, because "pit a lap
# later" is an instruction and "change six features at once" is not.
SCAN_FEATURE = "tyre_life"


def _data_source() -> dict:
    if PROCESSED_DATA_SOURCE_JSON.exists():
        return json.loads(PROCESSED_DATA_SOURCE_JSON.read_text())
    return {"source": "unknown", "reason": f"no marker at {PROCESSED_DATA_SOURCE_JSON}"}


def explain_target(target: str, quick: bool = False, with_figures: bool = True) -> dict:
    """Compute the full Task 8 analysis for one target."""
    t = loading.load_target(target)
    log.info("Task 8: explaining %s - %d features (%d identity), %d test rows",
             target, len(t.features), t.n_identity, len(t.X_test))

    n_repeats = 3 if quick else 10
    nsamples = 60 if quick else shap_analysis.KERNEL_NSAMPLES
    lime_samples = 400 if quick else lime_analysis.NUM_SAMPLES

    # --- global importance, model-agnostic so both families are comparable ---
    imp_dnn = importance.permutation_importance(
        t.dnn_predict, t.X_test, t.y_test, t.features, t.task, n_repeats=n_repeats)
    imp_cls = importance.permutation_importance(
        t.classical_predict, t.X_test, t.y_test, t.features, t.task, n_repeats=n_repeats)
    imp_cmp = importance.compare_importance(imp_dnn, imp_cls)

    # --- SHAP ---------------------------------------------------------------
    shap_cls = _classical_shap(t)
    shap_dnn = shap_analysis.kernel_shap(
        t.dnn_predict, t.X_train, t.X_test, t.features, nsamples=nsamples)
    rank_dnn = shap_analysis.global_ranking(shap_dnn)
    rank_cls = shap_analysis.global_ranking(shap_cls) if shap_cls else []

    # --- fairness ------------------------------------------------------------
    fair = fairness.assess(rank_dnn, t.identity_features)
    log.info("  fairness: identity carries %.1f%% of attribution (uniform would be %.1f%%)",
             fair["identity_attribution_share"] * 100, fair["expected_share_if_uniform"] * 100)

    # --- per-prediction explanations ------------------------------------------
    rows = loading.pick_representative_rows(t)
    lime_exp = lime_analysis.build_explainer(t.X_train, t.features, t.task)
    target_std = float(np.std(t.y_train)) if t.task == "regression" else None

    scan_idx = t.features.index(SCAN_FEATURE) if SCAN_FEATURE in t.features else 0
    scan_name = t.features[scan_idx]
    lo, hi = float(np.min(t.X_train[:, scan_idx])), float(np.max(t.X_train[:, scan_idx]))
    threshold = 0.5 if t.task == "classification" else float(np.median(t.y_train))

    examples: dict = {}
    scores: list[dict] = []

    for label, idx in rows.items():
        row = t.X_test[idx]
        p_dnn = float(np.asarray(t.dnn_predict(row[None, :])).ravel()[0])
        p_cls = float(np.asarray(t.classical_predict(row[None, :])).ravel()[0])

        shap_row = shap_analysis.explain_row(shap_dnn, idx, top_n=min(6, len(t.features)))
        shap_top3 = shap_analysis.stability(shap_dnn, idx, top_k=3)

        lm = lime_analysis.explain_row(
            lime_exp, row, t.dnn_predict, t.task,
            num_features=min(6, len(t.features)), num_samples=lime_samples)
        lime_top3 = lime_analysis.top_features(lm, t.features, top_k=3)

        ts = trust.compute(
            task=t.task, dnn_prediction=p_dnn, classical_prediction=p_cls,
            shap_top=shap_top3, lime_top=lime_top3, target_std=target_std)
        scores.append(ts)

        cf = counterfactual.perturbation_scan(
            t.dnn_predict, row, scan_idx, scan_name, lo, hi, threshold,
            steps=25 if quick else 60)

        fvals = {f: float(v) for f, v in zip(t.features, row)}
        sentence = (
            narrative.pit_decision_sentence(p_dnn, shap_row, fvals, ts["band"]["label"])
            if t.task == "classification" else
            narrative.laptime_sentence(p_dnn, shap_row, fvals, ts["band"]["label"],
                                       base_value=shap_dnn["base_value"])
        )

        figs = {}
        if with_figures:
            figs = {
                "shap_waterfall": visualize.shap_waterfall(
                    shap_row, shap_dnn["base_value"], p_dnn,
                    f"{target} - {label.replace('_', ' ')} (deep network)",
                    ML_FIGURES_DIR / f"xai_{target}_{label}_shap_waterfall.png").name,
                "lime": visualize.lime_plot(
                    lm, f"{target} - {label.replace('_', ' ')} (LIME local surrogate)",
                    ML_FIGURES_DIR / f"xai_{target}_{label}_lime.png").name,
                "counterfactual": visualize.counterfactual_curve(
                    cf, f"{target} - {label.replace('_', ' ')}: sweeping {scan_name}",
                    ML_FIGURES_DIR / f"xai_{target}_{label}_counterfactual.png").name,
            }

        examples[label] = {
            "row_index": int(idx),
            "lap": int(t.test_lap_numbers[idx]),
            "dnn_prediction": p_dnn,
            "classical_prediction": p_cls,
            "feature_values": fvals,
            "shap_dnn": shap_row,
            "shap_top3": shap_top3,
            "lime": {k: v for k, v in lm.items() if k != "explanation"},
            "lime_top3": lime_top3,
            "trust": ts,
            "counterfactual": cf,
            "counterfactual_sentence": narrative.counterfactual_sentence(cf),
            "narrative": sentence,
            "figures": figs,
        }

    # --- DiCE (classification only) --------------------------------------------
    dice = None
    if t.task == "classification":
        q = loading.boundary_row_index(t)
        dice = counterfactual.dice_counterfactuals(
            t.classical_estimator, t.classical_X_train_transformed, t.y_train,
            t.classical_X_test_transformed[q], t.transformed_feature_names,
            total_cfs=2 if quick else 3,
            timeout_seconds=20 if quick else counterfactual.DICE_TIMEOUT_SECONDS)
        dice["query_row_index"] = int(q)

    figures = {}
    if with_figures:
        figures = {
            "shap_summary_dnn": visualize.shap_summary(
                shap_dnn, t.X_test, f"{target} - SHAP summary (deep network)",
                ML_FIGURES_DIR / f"xai_{target}_shap_summary.png").name,
            "importance": visualize.importance_comparison(
                imp_cmp, f"{target} - permutation importance: deep network vs {t.classical_name}",
                ML_FIGURES_DIR / f"xai_{target}_importance_comparison.png").name,
            "fairness": visualize.fairness_plot(
                fair, f"{target} - identity vs race-state attribution",
                ML_FIGURES_DIR / f"xai_{target}_fairness.png").name,
        }

    return {
        "task": t.task,
        "classical_name": t.classical_name,
        "source_dataset": t.source_dataset,
        "dataset_source": _data_source(),
        "n_test": int(len(t.X_test)),
        "features": t.features,
        "identity_features": t.identity_features,
        "importance": {"dnn": imp_dnn, "classical": imp_cls},
        "importance_comparison": imp_cmp,
        "shap": {
            "dnn": {"ranking": rank_dnn, "note": shap_dnn["note"],
                    "explainer": shap_dnn["explainer"], "exact": shap_dnn["exact"]},
            "classical": ({"ranking": rank_cls, "note": shap_cls["note"],
                           "explainer": shap_cls["explainer"], "exact": shap_cls["exact"]}
                          if shap_cls else
                          {"ranking": [], "note": "No SHAP explainer applies to this estimator type.",
                           "explainer": None, "exact": None}),
        },
        "fairness": fair,
        "examples": examples,
        "trust_summary": trust.summarise(scores),
        "dice": dice,
        "figures": figures,
    }


def _classical_shap(t) -> dict | None:
    """TreeExplainer when Task 6's selected model is a tree ensemble, which is
    exact; KernelExplainer otherwise. Returns None only if both fail, and the
    report then says so instead of showing an empty table as if it were a
    result."""
    try:
        return shap_analysis.tree_shap(
            t.classical_estimator, t.classical_X_test_transformed,
            t.transformed_feature_names, t.task)
    except Exception as exc:
        log.info("  TreeExplainer not applicable to %s (%s); falling back to KernelExplainer",
                 t.classical_name, type(exc).__name__)
    try:
        return shap_analysis.kernel_shap(
            t.classical_predict, t.X_train, t.X_test, t.features, nsamples=60)
    except Exception as exc:
        log.warning("  no SHAP explainer succeeded for %s: %s", t.classical_name, exc)
        return None


def run_all(quick: bool = False) -> dict:
    """Compute Task 8 for every target and write the committed artifacts."""
    ensure_dirs()
    results = {target: explain_target(target, quick=quick) for target in TARGETS}

    rep = ML_REPORTS_DIR
    reports.shap_report(results, rep / "xai_shap_report.md")
    reports.lime_report(results, rep / "xai_lime_report.md")
    reports.counterfactual_report(results, rep / "xai_counterfactual_report.md")
    reports.trust_report(results, trust.WEIGHTS, rep / "xai_trust_score_report.md")
    reports.fairness_report(results, rep / "xai_fairness_report.md")
    reports.dashboard(results, rep / "xai_explainability_dashboard.md")

    XAI_RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    XAI_RESULTS_JSON.write_text(json.dumps(
        {"generated_at": datetime.now(timezone.utc).isoformat(),
         "task": "Task 8 - Explainable AI",
         "dataset_source": _data_source(),
         "targets": results},
        indent=2, default=_json_default))
    return results


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def artifacts_exist() -> bool:
    return XAI_RESULTS_JSON.exists()


def load_results() -> dict:
    """The committed Task 8 results, for the API to serve without recomputing."""
    if not XAI_RESULTS_JSON.exists():
        raise loading.ExplainerUnavailableError(
            f"No Task 8 artifact at {XAI_RESULTS_JSON}. Run `python scripts/build_all.py`."
        )
    return json.loads(XAI_RESULTS_JSON.read_text())
