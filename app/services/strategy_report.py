"""
app.services.strategy_report
============================

The race-strategy report generator (Task 9).

It takes one race state, runs the full analysis, and renders a self-contained
Markdown briefing: the prediction, which expert rules fired, the search plan,
the SHAP explanation and the trust score — the things a race engineer would
want on one page before making a call.

It computes nothing itself. Every value comes from
``strategy_service.run_strategy_analysis``, which chains the Task 6 models, the
Task 2 expert system and the Task 3 search, plus the live Task 8 explainer. If a
component is unavailable the report says which and why, rather than leaving a
blank that reads like a zero.

Markdown rather than PDF on purpose: it needs no extra dependency, it diffs, and
``pandoc report.md -o report.pdf`` converts it for anyone who wants PDF.
"""
from __future__ import annotations

from datetime import datetime, timezone

TARGET_LABEL = {
    "target_laptime": "Lap time",
    "target_pit_next_lap": "Pit decision",
}


def _f(value, digits: int = 3, suffix: str = "") -> str:
    if value is None:
        return "not available"
    if isinstance(value, float):
        return f"{value:.{digits}f}{suffix}"
    return f"{value}{suffix}"


def filename_for(race_state: dict) -> str:
    """A stable, descriptive filename, e.g. ``strategy_VER_lap30_20260927.md``."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    driver = str(race_state.get("driver", "unknown")).replace(" ", "")
    return f"strategy_{driver}_lap{race_state.get('current_lap', 0)}_{stamp}.md"


def render_markdown(analysis: dict) -> str:
    """Render one analysis dict as a Markdown strategy briefing."""
    rs = analysis["race_state"]
    pred = analysis["prediction"]
    search = analysis["optimal_search_strategy"]
    rules = analysis["triggered_expert_rules"]
    explanation = analysis.get("explanation") or {}

    L: list[str] = [
        f"# Race Strategy Report — {rs.get('driver')} ({rs.get('team')})",
        "",
        f"_Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by "
        f"`app/services/strategy_report.py`._",
        "",
        f"**Data source:** {analysis.get('data_source')} — this system is trained on a single "
        "Grand Prix. Treat it as decision support for that race, not a general F1 model.",
        "",
        "## 1. Race state",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Driver | {rs.get('driver')} |",
        f"| Team | {rs.get('team')} |",
        f"| Lap | {rs.get('current_lap')} of {rs.get('total_laps')} |",
        f"| Tyre compound | {rs.get('tyre_compound')} |",
        f"| Tyre age | {rs.get('tyre_age')} laps |",
        f"| Track temperature | {_f(rs.get('track_temperature'), 1, ' °C')} |",
        f"| Weather | {rs.get('weather')} |",
        f"| Fuel | {_f(rs.get('fuel_kg'), 1, ' kg')} |",
        f"| Track status | {rs.get('track_status')} |",
        f"| Position | {rs.get('current_position')} |",
        "",
        "## 2. Recommendation",
        "",
        f"### {analysis.get('recommended_action') or 'No recommendation available'}",
        "",
        "| Output | Value | From |",
        "|---|---|---|",
        f"| Predicted lap time | {_f(pred.get('predicted_lap_time_seconds'), 3, ' s')} | "
        f"Task 6 model `{pred.get('laptime_model')}` |",
        f"| Pit probability | {_f(pred.get('probability_pit'), 4)} | "
        f"Task 6 model `{pred.get('pit_model')}` |",
        f"| Expected cost, remaining stint | {_f(analysis.get('expected_cost_seconds'), 1, ' s')} | "
        f"Task 3 {search.get('algorithm')} search |",
        "",
    ]

    if pred.get("errors"):
        L += ["> **Model errors:** " + "; ".join(str(e) for e in pred["errors"]), ""]

    # --- honesty block: approximations and extrapolation --------------------
    approx = {k: v for k, v in (pred.get("approximated_features") or {}).items() if v}
    out_of_range = {k: v for k, v in (pred.get("out_of_range") or {}).items() if v}
    if approx or out_of_range:
        L += ["### How far to trust the inputs", ""]
        if out_of_range:
            L += ["**Extrapolation warning.** These inputs fall outside the range the models were "
                  "trained on, so the prediction is unvalidated there:", ""]
            for target, items in out_of_range.items():
                for item in items:
                    L.append(f"- `{item['feature']}` = {item['value']:.2f} "
                             f"(trained on {item['training_min']:.2f} – {item['training_max']:.2f}) "
                             f"— {TARGET_LABEL.get(target, target)}")
            L.append("")
        if approx:
            L += ["**Approximated features.** A single race-state snapshot cannot supply features "
                  "that need multi-lap history, so these were filled from training-data medians:", ""]
            for target, names in approx.items():
                L.append(f"- {TARGET_LABEL.get(target, target)}: {', '.join(f'`{n}`' for n in names)}")
            L.append("")

    # --- expert system -------------------------------------------------------
    L += ["## 3. Expert-system rules that fired", ""]
    if not rules:
        L += ["No rules fired for this race state. The recommendation above rests on the models and "
              "the search only.", ""]
    else:
        L += ["| Rule | Name | Conditions matched |", "|---|---|---:|"]
        for r in rules:
            L.append(f"| `{r['rule_id']}` | {r.get('name', '')} | "
                     f"{len(r.get('matched_conditions') or [])} |")
        L.append("")
        evidence = analysis.get("evidence") or {}
        if evidence:
            L += ["Conclusions asserted by those rules:", ""]
            for key, value in evidence.items():
                L.append(f"- **{str(key).replace('_', ' ')}**: {value}")
            L.append("")

    # --- search --------------------------------------------------------------
    L += ["## 4. Search plan (Task 3)", "",
          f"Algorithm **{search.get('algorithm')}**, "
          f"{'solution found' if search.get('found') else 'no solution found'}. "
          f"Expected cost for the remaining stint: "
          f"{_f(search.get('expected_cost_seconds'), 1, ' s')}.", ""]
    next_action = search.get("next_action") or {}
    if next_action:
        action = next_action.get("type", "unknown")
        compound = next_action.get("compound")
        L += [f"**Next action:** {action}" + (f" → fit {compound}" if compound else ""), ""]
    if search.get("note"):
        L += [f"> {search['note']}", ""]

    # --- explanation ---------------------------------------------------------
    L += ["## 5. Why this prediction (Task 8)", "",
          "**Read the model names carefully.** Section 2 is the Task 6 classical model, which is what "
          "this endpoint serves. The explanation below is of the **Task 7 neural network** on the same "
          "race state, because that is the model Task 8 explains. The two predictions will differ, and "
          "how much they differ is exactly the `model_agreement` term in the trust score.", ""]
    if not explanation:
        L += ["Explanations were not requested for this run. Set `explain: true` on the request to "
              "include SHAP factors and a trust score.", ""]
    else:
        for target, e in explanation.items():
            label = TARGET_LABEL.get(target, target)
            L += [f"### {label}", "",
                  f"Task 7 DNN prediction: **{_f(e.get('deep_prediction'), 4)}** "
                  f"(Task 6 `{e.get('classical_model')}`: {_f(e.get('classical_prediction'), 4)})", ""]
            if not e.get("available"):
                L += [f"Not available: {e.get('reason')}", ""]
                continue
            L += [f"> {e.get('narrative')}", "",
                  f"**Trust score {_f(e.get('trust_score'), 3)} — "
                  f"{(e.get('trust_band') or {}).get('label', 'unknown')}**. "
                  f"{(e.get('trust_band') or {}).get('meaning', '')}", ""]
            components = e.get("trust_components") or {}
            if components:
                L += ["| Trust component | Value |", "|---|---:|"]
                for name, value in components.items():
                    L.append(f"| {name.replace('_', ' ')} | {_f(value, 3)} |")
                L.append("")
            factors = e.get("shap_factors") or []
            if factors:
                # The explainer returns attributions, not inputs; the values come
                # from the feature row this request built.
                row = (pred.get("feature_rows") or {}).get(target) or {}
                L += ["Top factors behind this prediction (SHAP):", "",
                      "| Feature | Value on this lap | SHAP | Effect |", "|---|---:|---:|---|"]
                for f in factors:
                    L.append(f"| `{f['feature']}` | {_f(row.get(f['feature']), 4)} | "
                             f"{f['shap_value']:+.4f} | {f.get('direction', '')} |")
                L.append("")
            method = e.get("method") or {}
            if method.get("note"):
                L += [f"_{method['note']}_", ""]

    # --- provenance ----------------------------------------------------------
    L += ["## 6. Provenance", "",
          "Every number above was produced by this system at report time:", "",
          "| Section | Produced by |",
          "|---|---|",
          "| Prediction | `app/intelligence/ml/` (Task 6), served through `app/services/model_cache.py` |",
          "| Rules | `app/intelligence/expert_system/` (Task 2) |",
          "| Search plan | `app/intelligence/search/` (Task 3) |",
          "| Explanation and trust | `app/intelligence/xai/live.py` (Task 8) |",
          "| This report | `app/services/strategy_report.py` (Task 9) |",
          "",
          "### Limits of this report", "",
          "- Trained on one Grand Prix (2023 Bahrain). Do not read it as a general F1 model.",
          "- The pit classifier's operating point is tuned, but its holdout contained one pit stop, "
          "so its precision and recall carry little information.",
          "- The trust score is a project-defined heuristic. On our holdout it did not correlate with "
          "error — read its components, not the total.",
          "- Decision support for engineers, not an autonomous strategy system, and not for betting.",
          ""]
    return "\n".join(L) + "\n"
