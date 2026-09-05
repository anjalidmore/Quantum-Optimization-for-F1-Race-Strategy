#!/usr/bin/env python3
"""
practical07.py
==============

End-to-end driver for Task 8 (Explainable AI).

It:

1. Loads the Task 7 deep network (practical06) and a classical champion
   trained on the identical chronological split.
2. Computes global permutation importance for both model families - a
   model-agnostic method, so the two are directly comparable.
3. Computes SHAP attributions (TreeExplainer for the forest, KernelExplainer
   for the network) and LIME local surrogates on the same representative rows.
4. Searches for real counterfactuals: a single-feature bisection scan, plus
   DiCE for whole-row alternatives.
5. Scores each explained prediction with a defined trust metric.
6. Runs the fairness assessment - how much of the model's reasoning is driver
   or team identity rather than race state.
7. Writes a plain-English strategic explanation per prediction, and all six
   Markdown deliverables plus figures into ./outputs/.

Usage
-----
    python practical07.py [--output-dir OUT] [--quick]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")
warnings.filterwarnings("ignore")

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "src"))

import numpy as np  # noqa: E402

from f1xai import (  # noqa: E402
    counterfactual, fairness, importance, lime_analysis, loading,
    narrative, reports, shap_analysis, trust, visualize,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger("practical07")

TARGETS = ("target_laptime", "target_pit_next_lap")

# The feature a race engineer can actually act on, per target. The
# counterfactual scan moves this one and holds the rest of the race state
# fixed, because "pit one lap later" is an instruction and "change six
# features at once" is not.
SCAN_FEATURE = {
    "target_laptime": "tyre_life",
    "target_pit_next_lap": "tyre_life",
}


def run(output_dir: Path, quick: bool = False) -> dict:
    figures_dir = output_dir / "figures"
    results: dict = {}

    for target in TARGETS:
        log.info("=" * 70)
        log.info("STAGE: %s", target)
        log.info("=" * 70)

        t = loading.load_target(target)
        log.info("Features: %d (%d identity)  |  test rows: %d  |  source: %s",
                 len(t.features), t.n_identity, len(t.X_test), t.source_dataset)

        n_repeats = 3 if quick else 10
        nsamples = 60 if quick else shap_analysis.KERNEL_NSAMPLES
        lime_samples = 400 if quick else lime_analysis.NUM_SAMPLES

        # --- 1. Global permutation importance -------------------------------
        log.info("Permutation importance (%d repeats, model-agnostic) ...", n_repeats)
        imp_dnn = importance.permutation_importance(
            t.dnn_predict, t.X_test, t.y_test, t.features, t.task, n_repeats=n_repeats)
        imp_cls = importance.permutation_importance(
            t.classical_predict, t.X_test, t.y_test, t.features, t.task, n_repeats=n_repeats)
        imp_cmp = importance.compare_importance(imp_dnn, imp_cls)
        top = imp_dnn[0]
        log.info("  DNN top feature: %s (%s)", top["feature"],
                 f"{top['importance']:.6f}" if top["importance"] is not None else "undefined")

        # --- 2. SHAP ---------------------------------------------------------
        log.info("SHAP: TreeExplainer (classical, exact) ...")
        shap_cls = shap_analysis.tree_shap(
            t.classical_estimator, t.classical_X_test_scaled, t.features, t.task)
        log.info("SHAP: KernelExplainer (deep network, nsamples=%d) ...", nsamples)
        shap_dnn = shap_analysis.kernel_shap(
            t.dnn_predict, t.X_train, t.X_test, t.features, nsamples=nsamples)

        rank_dnn = shap_analysis.global_ranking(shap_dnn)
        rank_cls = shap_analysis.global_ranking(shap_cls)
        log.info("  SHAP top (DNN): %s", ", ".join(r["feature"] for r in rank_dnn[:3]))

        # --- 3. Fairness ------------------------------------------------------
        fair = fairness.assess(rank_dnn, t.identity_features)
        log.info("  Fairness: identity carries %.1f%% of attribution (uniform would be %.1f%%)",
                 fair["identity_attribution_share"] * 100, fair["expected_share_if_uniform"] * 100)

        # --- 4. Representative rows ------------------------------------------
        rows = loading.pick_representative_rows(t)
        lime_exp = lime_analysis.build_explainer(t.X_train, t.features, t.task)
        target_std = float(np.std(t.y_train)) if t.task == "regression" else None
        scan_name = SCAN_FEATURE[target]
        scan_idx = t.features.index(scan_name) if scan_name in t.features else 0
        scan_name = t.features[scan_idx]
        lo = float(np.min(t.X_train[:, scan_idx]))
        hi = float(np.max(t.X_train[:, scan_idx]))
        threshold = 0.5 if t.task == "classification" else float(np.median(t.y_train))

        examples: dict = {}
        trust_scores: list[dict] = []

        for label, idx in rows.items():
            log.info("  explaining %-24s (test row %d, lap %d)",
                     label, idx, int(t.test_lap_numbers[idx]))
            row = t.X_test[idx]
            p_dnn = float(np.asarray(t.dnn_predict(row[None, :])).ravel()[0])
            p_cls = float(np.asarray(t.classical_predict(row[None, :])).ravel()[0])

            shap_row = shap_analysis.explain_row(shap_dnn, idx, top_n=min(6, len(t.features)))
            shap_top3 = shap_analysis.stability(shap_dnn, idx, top_k=3)

            lm = lime_analysis.explain_row(
                lime_exp, row, t.dnn_predict, t.task,
                num_features=min(5, len(t.features)), num_samples=lime_samples)
            lime_top3 = lime_analysis.top_features(lm, t.features, top_k=3)

            ts = trust.compute(
                task=t.task, dnn_prediction=p_dnn, classical_prediction=p_cls,
                shap_top=shap_top3, lime_top=lime_top3, target_std=target_std,
            )
            trust_scores.append(ts)

            cf = counterfactual.perturbation_scan(
                t.dnn_predict, row, scan_idx, scan_name, lo, hi, threshold,
                steps=25 if quick else 60,
            )
            cf_sentence = narrative.counterfactual_sentence(cf)

            fvals = {f: float(v) for f, v in zip(t.features, row)}
            if t.task == "classification":
                sentence = narrative.pit_decision_sentence(
                    p_dnn, shap_row, fvals, ts["band"]["label"])
            else:
                sentence = narrative.laptime_sentence(
                    p_dnn, shap_row, fvals, ts["band"]["label"], base_value=shap_dnn["base_value"])

            figs = {
                "shap_waterfall": visualize.shap_waterfall(
                    shap_row, shap_dnn["base_value"], p_dnn,
                    f"{target} - {label.replace('_', ' ')} (deep network, KernelExplainer)",
                    figures_dir / f"{target}_{label}_shap_waterfall.png").name,
                "lime": visualize.lime_plot(
                    lm, f"{target} - {label.replace('_', ' ')} (LIME local surrogate)",
                    figures_dir / f"{target}_{label}_lime.png").name,
                "counterfactual": visualize.counterfactual_curve(
                    cf, f"{target} - {label.replace('_', ' ')}: sweeping {scan_name}",
                    figures_dir / f"{target}_{label}_counterfactual.png").name,
            }

            lm_serialisable = {k: v for k, v in lm.items() if k != "explanation"}
            examples[label] = {
                "row_index": int(idx),
                "lap": int(t.test_lap_numbers[idx]),
                "dnn_prediction": p_dnn,
                "classical_prediction": p_cls,
                "feature_values": fvals,
                "shap_dnn": shap_row,
                "shap_top3": shap_top3,
                "lime": lm_serialisable,
                "lime_top3": lime_top3,
                "trust": ts,
                "counterfactual": cf,
                "counterfactual_sentence": cf_sentence,
                "narrative": sentence,
                "figures": figs,
            }

        # --- 5. DiCE (classification only) ------------------------------------
        dice = None
        if t.task == "classification":
            # Query from the row closest to the decision boundary: that is
            # where a flip is most likely to be reachable, so a failure to find
            # one there is genuinely informative rather than an artefact of
            # starting from a hopeless row.
            q = loading.boundary_row_index(t)
            log.info("  DiCE whole-row counterfactuals from boundary row %d "
                     "(budget %ds) ...", q, counterfactual.DICE_TIMEOUT_SECONDS)
            dice = counterfactual.dice_counterfactuals(
                t.classical_estimator, t.classical_X_train_scaled, t.y_train,
                t.classical_X_test_scaled[q], t.features,
                total_cfs=2 if quick else 3,
                timeout_seconds=20 if quick else counterfactual.DICE_TIMEOUT_SECONDS,
            )
            dice["query_row_index"] = int(q)
            log.info("  DiCE: %s", "ok" if dice.get("counterfactuals") else dice.get("reason"))

        # --- figures ----------------------------------------------------------
        figs = {
            "shap_summary_dnn": visualize.shap_summary(
                shap_dnn, t.X_test, f"{target} - SHAP summary (deep network)",
                figures_dir / f"{target}_shap_summary_dnn.png").name,
            "importance": visualize.importance_comparison(
                imp_cmp, f"{target} - permutation importance: DNN vs {t.classical_name}",
                figures_dir / f"{target}_importance_comparison.png").name,
            "fairness": visualize.fairness_plot(
                fair, f"{target} - identity vs race-state attribution",
                figures_dir / f"{target}_fairness.png").name,
        }

        results[target] = {
            "task": t.task,
            "classical_name": t.classical_name,
            "source_dataset": t.source_dataset,
            "n_test": int(len(t.X_test)),
            "features": t.features,
            "identity_features": t.identity_features,
            "importance": {"dnn": imp_dnn, "classical": imp_cls},
            "importance_comparison": imp_cmp,
            "shap": {
                "dnn": {"ranking": rank_dnn, "note": shap_dnn["note"],
                        "explainer": shap_dnn["explainer"], "exact": shap_dnn["exact"]},
                "classical": {"ranking": rank_cls, "note": shap_cls["note"],
                              "explainer": shap_cls["explainer"], "exact": shap_cls["exact"]},
            },
            "fairness": fair,
            "examples": examples,
            "trust_summary": trust.summarise(trust_scores),
            "dice": dice,
            "figures": figs,
        }

    _write_outputs(results, output_dir)
    return results


def _write_outputs(results: dict, output_dir: Path) -> None:
    rep = output_dir / "reports"
    reports.shap_report(results, rep / "shap_report.md")
    reports.lime_report(results, rep / "lime_report.md")
    reports.counterfactual_report(results, rep / "counterfactual_report.md")
    reports.trust_report(results, trust.WEIGHTS, rep / "trust_score_report.md")
    reports.fairness_report(results, rep / "fairness_report.md")
    reports.dashboard(results, rep / "explainability_dashboard.md")

    p = output_dir / "metadata" / "xai_results.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(results, indent=2, default=_json_default))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def main() -> int:
    ap = argparse.ArgumentParser(description="Task 8 - Explainable AI")
    ap.add_argument("--output-dir", default=str(_HERE / "outputs"))
    ap.add_argument("--quick", action="store_true", help="fast smoke run (not the committed results)")
    args = ap.parse_args()

    out = Path(args.output_dir).resolve()
    results = run(out, quick=args.quick)

    log.info("=" * 70)
    log.info("TASK 8 SUMMARY")
    log.info("=" * 70)
    for target, r in results.items():
        f = r["fairness"]
        log.info("%s: identity attribution %.1f%% (uniform %.1f%%, concentration %sx)",
                 target, f["identity_attribution_share"] * 100,
                 f["expected_share_if_uniform"] * 100, f["concentration_ratio"])
        log.info("  trust across %d explained rows: mean %s, bands %s",
                 r["trust_summary"]["n"], r["trust_summary"]["mean"], r["trust_summary"]["bands"])
    log.info("All Task-8 deliverables written to %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
