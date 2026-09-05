"""
app.intelligence.xai.reports
============================

Markdown deliverable generators for Task 8.

Every value rendered here is passed in from a real explainer run on a real
fitted model. Where a quantity is undefined (ROC-AUC on a single-class split,
a counterfactual that does not exist inside the observed range) the report
says so and gives the reason, rather than printing a number that looks
plausible.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _write(out_path: Path, lines: list[str]) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def _f(v, nd=4):
    return "_undefined_" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def shap_report(results: dict, out_path: Path) -> Path:
    lines = [
        "# Task 8 - SHAP Report", "", f"_Generated {_stamp()}._", "",
        "SHAP distributes the gap between a prediction and the average prediction",
        "among the input features, using Shapley values from cooperative game theory.",
        "The attributions below are computed on the untouched chronological test set.",
        "",
        "## Explainer choice",
        "",
        "| Model | Explainer | Exact? | Why |",
        "|---|---|---|---|",
        "| Classical (random forest) | `TreeExplainer` | **Yes** | Walks the ensemble directly; exact Shapley values in polynomial time. |",
        "| Deep network (Keras) | `KernelExplainer` | No - sampled | Model-agnostic, needs only a `predict` function. `DeepExplainer`'s Keras 3 support targets the TensorFlow backend; this project runs Keras on PyTorch, so `KernelExplainer` is the correct choice. |",
        "",
        "> **Trade-off:** the network's values carry sampling noise the forest's do not.",
        "> Compare them by **rank and sign**, not by magnitude.",
        "",
    ]
    for target, r in results.items():
        lines += [f"## {target}", "",
                  f"**Task:** {r['task']}  |  **Test rows explained:** {r['n_test']}  |  "
                  f"**Dataset:** `{r['source_dataset']}`", ""]
        for family in ("dnn", "classical"):
            s = r["shap"][family]
            lines += [
                f"### Global ranking - {'deep network' if family == 'dnn' else r['classical_name']}",
                "", f"_{s['note']}_", "",
                "| Rank | Feature | Mean \\|SHAP\\| |", "|---:|---|---:|",
            ]
            for i, row in enumerate(s["ranking"], 1):
                lines.append(f"| {i} | `{row['feature']}` | {row['mean_abs_shap']:.6f} |")
            lines.append("")
        lines += ["### Representative predictions explained", ""]
        for label, ex in r["examples"].items():
            lines += [
                f"#### {label.replace('_', ' ').title()} (test row {ex['row_index']}, lap {ex['lap']})",
                "",
                f"Deep network prediction: **{ex['dnn_prediction']:.4f}**  |  "
                f"{r['classical_name']}: **{ex['classical_prediction']:.4f}**",
                "",
                "| Feature | Value | SHAP | Effect |", "|---|---:|---:|---|",
            ]
            for c in ex["shap_dnn"]:
                lines.append(
                    f"| `{c['feature']}` | {ex['feature_values'].get(c['feature'], float('nan')):.4g} | "
                    f"{c['shap_value']:+.6f} | {c['direction']} the prediction |"
                )
            lines += ["", f"![waterfall]({ex['figures']['shap_waterfall']})", ""]
        lines += [f"![summary]({r['figures']['shap_summary_dnn']})", "", "---", ""]
    return _write(out_path, lines)


def lime_report(results: dict, out_path: Path) -> Path:
    lines = [
        "# Task 8 - LIME Report", "", f"_Generated {_stamp()}._", "",
        "## How this differs from the SHAP report",
        "",
        "LIME does **not** compute Shapley values. It perturbs the row being explained,",
        "asks the real model what it predicts for each perturbation, and fits a weighted",
        "**linear surrogate** to those answers in the neighbourhood of that row. The",
        "reported weights are that surrogate's coefficients.",
        "",
        "| | SHAP | LIME |",
        "|---|---|---|",
        "| Question answered | How is credit for this prediction fairly divided? | What simple model behaves like the real one *around here*? |",
        "| Basis | Shapley values (game theory) | Local weighted linear regression |",
        "| Guarantee | Attributions sum to (prediction - base value) | None; quality is reported as the surrogate's R2 |",
        "| Output units | Same units as the model output | Surrogate coefficients on discretised conditions |",
        "",
        "`local_r2` is the diagnostic that matters: a low value means a straight line is a",
        "poor stand-in for the model near this row, so the LIME explanation should be",
        "discounted regardless of how confident it looks.",
        "",
    ]
    for target, r in results.items():
        lines += [f"## {target}", ""]
        for label, ex in r["examples"].items():
            lm = ex["lime"]
            lines += [
                f"### {label.replace('_', ' ').title()} (test row {ex['row_index']}, lap {ex['lap']})",
                "",
                f"Local surrogate R2: **{lm['local_r2']:.3f}** over {lm['num_samples']} perturbations.",
                "",
                "| Condition | Weight | Effect |", "|---|---:|---|",
            ]
            for c in lm["contributions"]:
                lines.append(f"| `{c['condition']}` | {c['weight']:+.6f} | {c['direction']} the prediction |")
            lines += [
                "",
                f"**SHAP top-3:** {', '.join(f'`{f}`' for f in ex['shap_top3'])}",
                f"**LIME top-3:** {', '.join(f'`{f}`' for f in ex['lime_top3'])}",
                f"**Agreement (Jaccard):** {ex['trust']['components']['explanation_stability']:.3f}",
                "",
                f"![lime]({ex['figures']['lime']})", "",
            ]
        lines += ["---", ""]
    return _write(out_path, lines)


def counterfactual_report(results: dict, out_path: Path) -> Path:
    lines = [
        "# Task 8 - Counterfactual Analysis Report", "", f"_Generated {_stamp()}._", "",
        "*What would have to change about the race state for the recommendation to flip?*",
        "",
        "Two methods, for two different questions:",
        "",
        "* **Single-feature bisection scan** - hold the entire race state fixed, move one",
        "  feature, find the exact value at which the model's output crosses the decision",
        "  threshold. This is the question a race engineer actually asks (*how many more",
        "  laps on these tyres?*), because it yields an actionable instruction.",
        "* **DiCE (random search)** - find complete alternative race states the model would",
        "  classify the other way. Useful when several different routes to a different call",
        "  exist; less actionable, because changing six things at once is not an instruction.",
        "",
        "Every number below is the result of a real search against the real model. Where no",
        "counterfactual exists inside the feature's observed range, that is reported as",
        "*not reachable* together with the range searched.",
        "",
    ]
    for target, r in results.items():
        lines += [f"## {target}", ""]
        for label, ex in r["examples"].items():
            cf = ex.get("counterfactual")
            if not cf:
                continue
            lines += [
                f"### {label.replace('_', ' ').title()} (test row {ex['row_index']}, lap {ex['lap']})", "",
                f"**Scanned feature:** `{cf['feature']}`  |  "
                f"**Current value:** {cf['original_value']:.4g}  |  "
                f"**Current prediction:** {cf['original_prediction']:.4f}  |  "
                f"**Threshold:** {cf['threshold']:g}", "",
                f"**Searched range:** [{cf['searched_range'][0]:.4g}, {cf['searched_range'][1]:.4g}] "
                f"in {cf['steps']} steps", "",
            ]
            if cf["reachable"]:
                lines += [
                    f"**Result: reachable.** The recommendation flips at "
                    f"`{cf['feature']}` = **{cf['crossing_value']:.4g}** - a {cf['direction']} of "
                    f"**{abs(cf['delta_required']):.4g}** from the current value.", "",
                ]
            else:
                lines += [f"**Result: not reachable.** {cf['note']}", ""]
            lines += [f"> {ex['counterfactual_sentence']}", "",
                      f"![cf]({ex['figures']['counterfactual']})", ""]

        dice = r.get("dice")
        if dice:
            lines += ["### DiCE - diverse whole-row counterfactuals", ""]
            if not dice.get("available") or not dice.get("counterfactuals"):
                head = "**Timed out.**" if dice.get("timed_out") else "**Not produced.**"
                lines += [f"{head} {dice.get('reason', 'no counterfactuals returned')}", ""]
            else:
                lines += [f"_{dice['note']}_", "",
                          "| # | Features changed | Changes |", "|---:|---:|---|"]
                for i, cfx in enumerate(dice["counterfactuals"], 1):
                    ch = "; ".join(
                        f"`{f}` {v['from']:+.3f} -> {v['to']:+.3f} ({v['delta']:+.3f})"
                        for f, v in list(cfx["changed_features"].items())[:4]
                    )
                    lines.append(f"| {i} | {cfx['n_changes']} | {ch} |")
                lines.append("")
        lines += ["---", ""]
    return _write(out_path, lines)


def trust_report(results: dict, weights: dict, out_path: Path) -> Path:
    lines = [
        "# Task 8 - Trust Score Report", "", f"_Generated {_stamp()}._", "",
        "## The formula", "",
        "```",
        "trust = 0.40 * confidence",
        "      + 0.30 * model_agreement",
        "      + 0.30 * explanation_stability",
        "```", "",
        "| Component | Definition | Why it is in the score |",
        "|---|---|---|",
        "| `confidence` | Classification: `2 * abs(p - 0.5)`. Regression: `1 - min(1, gap / target_std)`. | A prediction sitting on the decision boundary is unusable however well explained. |",
        "| `model_agreement` | `1 - abs(p_dnn - p_classical)` | Two model families trained on identical folds agreeing is real evidence. One disagreeing means at least one is wrong and you cannot tell which. |",
        "| `explanation_stability` | Jaccard overlap of SHAP's and LIME's top-3 features | If two established explanation methods disagree about *why*, the explanation you would show the engineer is not trustworthy even if the prediction is right. |",
        "",
        f"**Weights:** {weights}. These are a judgement, not a derivation - confidence carries",
        "the largest share because boundary predictions are useless regardless of explanation",
        "quality, while agreement and stability are weighted equally (a wrong-but-explained",
        "and a right-but-unexplainable prediction are both unsafe to act on). They are exposed",
        "as `f1xai.trust.WEIGHTS` so they can be challenged and changed.",
        "",
        "## What the bands mean to a race engineer", "",
        "| Score | Band | Action |", "|---|---|---|",
        "| >= 0.75 | **HIGH** | Both families agree, prediction far from the boundary, SHAP and LIME tell the same story. Safe to act on. |",
        "| 0.50 - 0.75 | **MODERATE** | One input among several. Read the SHAP factors before acting. |",
        "| 0.25 - 0.50 | **LOW** | A prompt to look at the evidence, not a recommendation. |",
        "| < 0.25 | **DO NOT ACT** | Carries no more information than a coin flip. |",
        "",
        "## Worked examples", "",
    ]
    for target, r in results.items():
        lines += [f"### {target}", "",
                  "| Case | Test row | Confidence | Agreement | Stability | **Trust** | Band |",
                  "|---|---:|---:|---:|---:|---:|---|"]
        for label, ex in r["examples"].items():
            t = ex["trust"]
            c = t["components"]
            lines.append(
                f"| {label.replace('_', ' ')} | {ex['row_index']} | {c['confidence']:.3f} | "
                f"{c['model_agreement']:.3f} | {c['explanation_stability']:.3f} | "
                f"**{t['trust_score']:.3f}** | {t['band']['label']} |"
            )
        summary = r["trust_summary"]
        lines += ["", f"**Across all {summary['n']} explained rows:** mean {_f(summary['mean'],3)}, "
                      f"range {_f(summary['min'],3)}-{_f(summary['max'],3)}. "
                      f"Bands: {summary['bands']}.", "", "---", ""]
    return _write(out_path, lines)


def fairness_report(results: dict, out_path: Path) -> Path:
    lines = [
        "# Task 8 - Fairness Assessment", "", f"_Generated {_stamp()}._", "",
        "**The question:** is the model predicting from *race state*, or from *who is driving*?",
        "",
        "This is a concrete risk in this project, not a hypothetical one. Task 5's feature",
        "selection retained one-hot driver and team dummies. A lap-time model leaning on",
        "`driver_sai` has learned \"Sainz laps look like this\" rather than \"a tyre this old",
        "on a track this hot laps like this\". Such a model cannot generalise to an unseen",
        "driver, silently encodes car performance as driver pace, and would give two cars in",
        "an identical race state different strategy calls purely because of the name on the car.",
        "",
        "**The measurement:** share of total mean-|SHAP| attribution going to identity",
        "features, compared against the share expected if attribution were spread evenly",
        "across all features. The ratio of the two is the concentration.",
        "",
    ]
    for target, r in results.items():
        a = r["fairness"]
        lines += [
            f"## {target}", "",
            "| | |", "|---|---|",
            f"| Selected features | {a['n_features']} |",
            f"| Identity features | {a['n_identity_features']} ({', '.join(f'`{f}`' for f in a['identity_features']) or 'none'}) |",
            f"| **Identity share of attribution** | **{a['identity_attribution_share'] * 100:.1f}%** |",
            f"| Expected share if uniform | {a['expected_share_if_uniform'] * 100:.1f}% |",
            f"| **Concentration ratio** | **{a['concentration_ratio'] if a['concentration_ratio'] is not None else 'n/a'}x** |",
            f"| Highest-ranked identity feature | {a['highest_ranked_identity_feature'] or 'n/a'} |",
            f"| Top race-state features | {', '.join(f'`{f}`' for f in a['top_race_state_features'])} |",
            "", f"**Reading:** {a['reading']}", "",
            f"![fairness]({r['figures']['fairness']})", "", "---", "",
        ]
    return _write(out_path, lines)


def dashboard(results: dict, out_path: Path) -> Path:
    """The explainability dashboard: importance, SHAP, trust and the
    engineer-facing sentence, per prediction, in one place."""
    lines = [
        "# Task 8 - Explainability Dashboard", "", f"_Generated {_stamp()}._", "",
        "One page per target bringing together **global importance**, **per-prediction SHAP**,",
        "the **trust score**, and the **plain-English recommendation** a race engineer would",
        "actually read. This is the view that answers \"why should I believe this?\".",
        "",
    ]
    for target, r in results.items():
        lines += [
            f"## {target}", "",
            f"**Task:** {r['task']}  |  **Deep network vs `{r['classical_name']}`**  |  "
            f"**Dataset:** `{r['source_dataset']}`", "",
            "### Global feature importance (permutation, model-agnostic)", "",
            "| Feature | Deep network | Classical | Rank gap |", "|---|---:|---:|---:|",
        ]
        for row in r["importance_comparison"][:10]:
            lines.append(
                f"| `{row['feature']}` | {_f(row['dnn_importance'], 6)} | "
                f"{_f(row['classical_importance'], 6)} | "
                f"{row['rank_gap'] if row['rank_gap'] is not None else '-'} |"
            )
        lines += ["", f"![importance]({r['figures']['importance']})", "",
                  "### Per-prediction explanations", ""]
        for label, ex in r["examples"].items():
            t = ex["trust"]
            lines += [
                f"#### {label.replace('_', ' ').title()} - test row {ex['row_index']}, lap {ex['lap']}", "",
                f"> **{ex['narrative']}**", "",
                f"| Trust | {t['trust_score']:.3f} ({t['band']['label']}) |",
                "|---|---|",
                f"| Confidence | {t['components']['confidence']:.3f} |",
                f"| Model agreement | {t['components']['model_agreement']:.3f} |",
                f"| Explanation stability | {t['components']['explanation_stability']:.3f} |",
                "", f"_{t['band']['meaning']}_", "",
                "**Top factors (SHAP):** " + ", ".join(
                    f"`{c['feature']}` ({c['shap_value']:+.4f})" for c in ex["shap_dnn"][:3]
                ), "",
                f"**Counterfactual:** {ex.get('counterfactual_sentence', '_not computed for this row_')}", "",
            ]
        lines += ["---", ""]
    return _write(out_path, lines)
