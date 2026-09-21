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
import pandas as pd

from app.core.paths import (
    PROCESSED_DATA_SOURCE_JSON,
    TARGET_DIRNAME,
    XAI_RESULTS_JSON,
    ArtifactPaths,
    ensure_dirs,
)
from app.intelligence.xai import (
    counterfactual,
    fairness,
    importance,
    lime_analysis,
    loading,
    narrative,
    reports,
    shap_analysis,
    stratification,
    trust,
    visualize,
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


def explain_target(target: str, quick: bool = False, with_figures: bool = True,
                   out: ArtifactPaths | None = None) -> dict:
    """Compute the full Task 8 analysis for one target."""
    out = out or ArtifactPaths.default()
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
    # Classification: the decision threshold Task 7 actually uses, not 0.5.
    threshold = ((t.decision_threshold or 0.5) if t.task == "classification"
                 else float(np.median(t.y_train)))
    shap_dir, lime_dir, cf_dir = out.xai / "shap", out.xai / "lime", out.xai / "counterfactual"
    for d in (shap_dir, lime_dir, cf_dir):
        d.mkdir(parents=True, exist_ok=True)

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
            shap_top=shap_top3, lime_top=lime_top3, target_std=target_std,
            threshold=threshold, input_validity_share=trust.input_validity(row, t.X_train))
        scores.append(ts)

        if scan_name == counterfactual.TYRE_AGE:
            cf = counterfactual.tyre_age_scan(
                t.dnn_predict, row, t.features, lo, hi, threshold, t.task, step=1.0 if quick else 0.5)
        else:  # no tyre-age input: fall back to a single-feature scan
            cf = counterfactual.perturbation_scan(
                t.dnn_predict, row, scan_idx, scan_name, lo, hi, threshold, steps=25 if quick else 60)

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
                "shap_waterfall": _rel(out, visualize.shap_waterfall(
                    shap_row, shap_dnn["base_value"], p_dnn,
                    f"{target} - {label.replace('_', ' ')} (Task 7 DNN)",
                    shap_dir / f"{target}_{label}_waterfall.png")),
                "lime": _rel(out, visualize.lime_plot(
                    lm, f"{target} - {label.replace('_', ' ')} (LIME local surrogate)",
                    lime_dir / f"{target}_{label}_lime.png")),
                "counterfactual": _rel(out, visualize.counterfactual_curve(
                    cf, f"{target} - {label.replace('_', ' ')}: sweeping {scan_name}",
                    cf_dir / f"{target}_{label}_tyre_age.png")),
            }

        ident = t.test_ids.iloc[idx].to_dict() if t.test_ids is not None else {}
        examples[label] = {
            "row_index": int(idx),
            "lap": int(t.test_lap_numbers[idx]),
            "driver": ident.get("Driver"), "team": ident.get("Team"),
            "compound": ident.get("Compound"), "stint": ident.get("Stint"),
            "actual": float(t.y_test[idx]),
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

    # --- trust score for EVERY test lap (not only the showcase rows) ----------
    preds_all = np.asarray(t.dnn_predict(t.X_test)).ravel()
    cls_all = np.asarray(t.classical_predict(t.X_test)).ravel()
    trust_rows = []
    for i in range(len(t.X_test)):
        s_top = shap_analysis.stability(shap_dnn, i, top_k=3)
        l_top = lime_analysis.top_features(
            lime_analysis.explain_row(lime_exp, t.X_test[i], t.dnn_predict, t.task,
                                      num_features=min(6, len(t.features)),
                                      num_samples=200 if quick else 1000),
            t.features, top_k=3)
        ts_i = trust.compute(task=t.task, dnn_prediction=float(preds_all[i]),
                             classical_prediction=float(cls_all[i]), shap_top=s_top, lime_top=l_top,
                             target_std=target_std, threshold=threshold,
                             input_validity_share=trust.input_validity(t.X_test[i], t.X_train))
        ident = t.test_ids.iloc[i].to_dict() if t.test_ids is not None else {}
        trust_rows.append({
            "target": target, "row_index": i, "driver": ident.get("Driver"), "team": ident.get("Team"),
            "compound": ident.get("Compound"), "lap": int(t.test_lap_numbers[i]),
            "actual": float(t.y_test[i]), "dnn_prediction": float(preds_all[i]),
            "classical_prediction": float(cls_all[i]),
            **({"dnn_decision": "pit" if preds_all[i] >= threshold else "stay out",
                "decision_correct": bool((preds_all[i] >= threshold) == (t.y_test[i] == 1))}
               if t.task == "classification" else
               {"dnn_abs_error_s": float(abs(preds_all[i] - t.y_test[i]))}),
            "shap_top3": ", ".join(s_top), "lime_top3": ", ".join(l_top),
            **{f"component_{k}": v for k, v in ts_i["components"].items()},
            "trust_score": ts_i["trust_score"], "trust_band": ts_i["band"]["label"],
        })

    # --- F1-specific performance stratification -----------------------------
    strata = stratification.stratify(t.task, t.test_ids, t.y_test, preds_all,
                                      threshold if t.task == "classification" else None)
    strata.insert(0, "target", target)

    # --- SHAP values for every test lap -------------------------------------
    shap_table = pd.DataFrame(np.asarray(shap_dnn["values"]), columns=[f"shap_{f}" for f in t.features])
    if t.test_ids is not None:
        shap_table = pd.concat([t.test_ids, shap_table], axis=1)
    shap_table.insert(0, "dnn_prediction", preds_all)
    shap_table.insert(1, "actual", t.y_test)
    shap_table["shap_base_value"] = shap_dnn["base_value"]
    for j, f in enumerate(t.features):           # the race state itself, for the dashboard
        shap_table[f"value_{f}"] = t.X_test[:, j]
    shap_table.to_csv(shap_dir / f"{target}_shap_values.csv", index=False)

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
            "shap_summary_dnn": _rel(out, visualize.shap_summary(
                shap_dnn, t.X_test, f"{target} - SHAP summary (Task 7 DNN)",
                shap_dir / f"{target}_shap_summary.png")),
            "shap_importance_dnn": _rel(out, visualize.shap_bar(
                rank_dnn, f"{target} - SHAP feature importance (mean |SHAP|, Task 7 DNN)",
                shap_dir / f"{target}_shap_importance.png")),
            "importance": _rel(out, visualize.importance_comparison(
                imp_cmp, f"{target} - permutation importance: Task 7 DNN vs {t.classical_name}",
                out.xai / "importance" / f"{target}_importance_comparison.png")),
            "fairness": _rel(out, visualize.fairness_plot(
                fair, f"{target} - identity vs race-state attribution",
                out.xai / "stratification" / f"{target}_identity_attribution.png")),
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
        "model_explained": {
            "family": "deep", "name": "Task 7 DNN (dnn_mlp)",
            "file": f"artifacts/models/deep_learning/{TARGET_DIRNAME[target]}/f1_dnn_model.h5",
            "decision_threshold": threshold if t.task == "classification" else None,
            "comparison_model": f"Task 6 {t.classical_name} (used for model agreement and importance comparison)",
        },
        "fairness": fair,
        "stratification": strata.to_dict(orient="records"),
        "examples": examples,
        "trust_summary": trust.summarise(scores),
        "trust_all_laps": trust_rows,
        "trust_all_laps_summary": {
            "n": len(trust_rows),
            "mean": float(np.mean([r["trust_score"] for r in trust_rows])),
            "bands": {b: sum(1 for r in trust_rows if r["trust_band"] == b)
                      for b in ("HIGH", "MODERATE", "LOW", "DO NOT ACT")},
        },
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


def _rel(out: ArtifactPaths, path: Path) -> str:
    """Figure paths as stored in the results: relative to the artifacts root, so
    the API and dashboard can serve them from the /artifacts mount."""
    try:
        return str(Path(path).relative_to(out.root))
    except ValueError:
        return Path(path).name


def _counterfactual_rows(results: dict) -> list[dict]:
    rows = []
    for target, r in results.items():
        thr = r["model_explained"]["decision_threshold"]
        for case, ex in r["examples"].items():
            cf = ex["counterfactual"]
            base = {"target": target, "case": case, "driver": ex.get("driver"), "team": ex.get("team"),
                    "compound": ex.get("compound"), "lap": ex["lap"], "model": "Task 7 DNN",
                    "method": cf["method"], "changed_feature": cf["feature"],
                    "derived_features_recomputed": "; ".join(cf.get("derived_features_recomputed", [])),
                    "original_value": cf["original_value"], "original_prediction": cf["original_prediction"],
                    "searched_range": f"{cf['searched_range'][0]:g}-{cf['searched_range'][1]:g}"}
            if r["task"] == "classification":
                rows.append({**base,
                             "original_decision": "pit" if cf["original_prediction"] >= thr else "stay out",
                             "new_value": cf["crossing_value"], "change": cf["delta_required"],
                             "new_prediction": cf.get("prediction_at_crossing"),
                             "new_decision": (None if cf["crossing_value"] is None else
                                              ("pit" if cf["original_prediction"] < thr else "stay out")),
                             "decision_flipped": cf["reachable"],
                             "note": ("decision flips at the nearest tyre age shown" if cf["reachable"] else
                                      "no tyre age in the training range flips this decision")})
            else:
                for e in cf.get("regression_effects", []):
                    rows.append({**base, "new_value": e["tyre_age"], "change": e["tyre_age_change_laps"],
                                 "new_prediction": e["predicted"],
                                 "prediction_change_s": e["change_in_prediction"],
                                 "note": "predicted lap time for a realistic change in tyre age"})
        dice = r.get("dice") or {}
        for k, c in enumerate(dice.get("counterfactuals", []), start=1):
            ch = c["changed_features"]
            rows.append({"target": target, "case": f"dice_{k}", "model": f"Task 6 {r['classical_name']}",
                         "method": dice.get("method"), "changed_feature": "; ".join(ch),
                         "original_value": "; ".join(f"{f}={v['from']:.4g}" for f, v in ch.items()),
                         "new_value": "; ".join(f"{f}={v['to']:.4g}" for f, v in ch.items()),
                         "change": "; ".join(f"{f}:{v['delta']:+.4g}" for f, v in ch.items()),
                         "decision_flipped": True,
                         "note": "DiCE multi-feature counterfactual on Task 6's model (supplementary)"})
        if dice and not dice.get("counterfactuals"):
            rows.append({"target": target, "case": "dice", "model": f"Task 6 {r['classical_name']}",
                         "method": dice.get("method"), "decision_flipped": False,
                         "note": dice.get("reason", "no counterfactual found")})
    return rows


def run_all(quick: bool = False, output_root: Path | None = None) -> dict:
    """Compute Task 8 for every target and write every deliverable under
    ``artifacts/xai/``.

    ``output_root`` redirects every write beneath one directory, defaulting to
    the committed ``artifacts/`` layout, so the test suite can run without
    rewriting tracked files.
    """
    out = ArtifactPaths.default() if output_root is None else ArtifactPaths(root=Path(output_root))
    out.ensure()
    ensure_dirs()
    results = {target: explain_target(target, quick=quick, out=out) for target in TARGETS}
    x = out.xai

    for target, r in results.items():
        r["figures"]["stratification"] = _rel(out, visualize.stratification_plot(
            r["stratification"], r["task"], f"{target} — F1-specific performance stratification (Task 7 DNN)",
            x / "stratification" / f"{target}_stratification.png"))
    fi = visualize.feature_importance_overview({t: r["importance"]["dnn"] for t, r in results.items()},
                                               x / "feature_importance.png")

    pd.DataFrame(_counterfactual_rows(results)).to_csv(x / "counterfactual_analysis.csv", index=False)
    pd.DataFrame([row for r in results.values() for row in r["trust_all_laps"]]).to_csv(
        x / "trust_score_report.csv", index=False)
    fair_rows = [{"analysis": "F1-specific performance stratification", **row}
                 for r in results.values() for row in r["stratification"]]
    fair_rows += [{"analysis": "identity-feature attribution share", "target": t,
                   "group_type": "model inputs", "group": "driver/team one-hot features",
                   "identity_attribution_share": r["fairness"]["identity_attribution_share"],
                   "expected_share_if_uniform": r["fairness"]["expected_share_if_uniform"],
                   "sample_note": r["fairness"].get("reading", "")}
                  for t, r in results.items()]
    pd.DataFrame(fair_rows).to_csv(x / "fairness_assessment.csv", index=False)
    pd.DataFrame([{"target": t, "rank": i + 1, "feature": row["feature"],
                   "permutation_importance": row["importance"], "std": row["std"]}
                  for t, r in results.items() for i, row in enumerate(r["importance"]["dnn"])]).to_csv(
        x / "feature_importance.csv", index=False)
    pd.DataFrame([{"target": t, "case": case, "driver": ex.get("driver"), "lap": ex["lap"],
                   "lime_prediction": ex["lime"].get("local_prediction"),
                   "model_prediction": ex["dnn_prediction"], "lime_r2": ex["lime"].get("local_r2"),
                   "lime_top3": ", ".join(ex["lime_top3"]), "shap_top3": ", ".join(ex["shap_top3"])}
                  for t, r in results.items() for case, ex in r["examples"].items()]).to_csv(
        x / "lime" / "lime_explanations.csv", index=False)

    reports.shap_report(results, x / "SHAP_Report.md")
    reports.lime_report(results, x / "LIME_Report.md")
    reports.counterfactual_report(results, x / "Counterfactual_Report.md")
    reports.trust_report(results, trust.WEIGHTS, x / "Trust_Score_Report.md")
    reports.fairness_report(results, x / "Fairness_Report.md")
    reports.dashboard(results, x / "Explainability_Dashboard.md")

    # xai_metadata.json — read by the /api/xai/* endpoints
    out.xai_results_json.write_text(json.dumps(_json_safe(
        {"generated_at": datetime.now(timezone.utc).isoformat(),
         "task": "Task 8 - Explainable AI",
         "model_explained": "Task 7 DNN (artifacts/models/deep_learning/<target>/f1_dnn_model.h5)",
         "dataset_source": _data_source(),
         "feature_importance_figure": _rel(out, fi),
         "trust_weights": trust.WEIGHTS,
         "targets": results}),
        indent=2, default=_json_default, allow_nan=False))
    return results


def _json_safe(o):
    """NaN/inf -> None, recursively. A metric that is undefined (recall on a
    group with no pit laps) must reach the API as null: NaN is not valid JSON."""
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, (float, np.floating)) and not np.isfinite(o):
        return None
    return o


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
