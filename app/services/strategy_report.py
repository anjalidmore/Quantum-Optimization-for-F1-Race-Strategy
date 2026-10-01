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

Markdown is the default: it needs no extra dependency and it diffs cleanly.
``render_html`` below renders the identical content as a self-contained HTML
page (``GET/POST /api/strategy/report?format=html``) for a caller that wants
something to open directly in a browser or print to PDF, without pulling in a
PDF-rendering dependency for one endpoint.
"""
from __future__ import annotations

import html as _html
import re
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


def filename_for(race_state: dict, extension: str = "md") -> str:
    """A stable, descriptive filename, e.g. ``strategy_VER_lap30_20260927.md``."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    driver = str(race_state.get("driver", "unknown")).replace(" ", "")
    return f"strategy_{driver}_lap{race_state.get('current_lap', 0)}_{stamp}.{extension}"


def render_markdown(analysis: dict) -> str:
    """Render one analysis dict as a Markdown strategy briefing."""
    rs = analysis["race_state"]
    pred = analysis["prediction"]
    dl = analysis.get("dl_prediction") or {}
    rec = analysis.get("recommendation") or {}
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
        f"### {rec.get('action') or analysis.get('recommended_action') or 'No recommendation available'}",
        "",
        f"**Confidence: {rec.get('confidence', 'unknown')}** — {rec.get('reason', '')}",
        "",
    ]
    if rec.get("disagreement"):
        L += ["> **ML and DL disagree on this call.** See the table below — the recommendation "
              "engine's combining rule (documented in `app/services/strategy_service.py`) falls "
              "back to ML in this case, not because DL is wrong, but because ML had the better "
              "holdout precision on this dataset's one real pit stop.", ""]
    L += [
        "| Output | Value | From |",
        "|---|---|---|",
        f"| Predicted lap time | {_f(pred.get('predicted_lap_time_seconds'), 3, ' s')} | "
        f"Task 6 model `{pred.get('laptime_model')}` |",
        f"| Predicted lap time (DL) | {_f(dl.get('predicted_lap_time_seconds'), 3, ' s')} | "
        "Task 7 network |",
        f"| Pit probability (ML) | {_f(pred.get('probability_pit'), 4)} | "
        f"Task 6 model `{pred.get('pit_model')}` |",
        f"| Pit probability (DL) | {_f(dl.get('probability_pit'), 4)} | "
        f"Task 7 network, tuned threshold {_f(dl.get('threshold'), 4)} |",
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
          "| ML prediction | `app/intelligence/ml/` (Task 6), served through `app/services/model_cache.py` |",
          "| DL prediction | `app/intelligence/dl/` (Task 7), served through `app/intelligence/xai/live.py`'s cache |",
          "| Rules | `app/intelligence/expert_system/` (Task 2) |",
          "| Search plan | `app/intelligence/search/` (Task 3) |",
          "| Recommendation engine | `app/services/strategy_service.py::_combine_recommendation` (Task 9) |",
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


def _inline_md(text: str) -> str:
    text = _html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    return text


def _markdown_to_html_body(markdown: str) -> str:
    """Converts exactly the Markdown subset ``render_markdown`` produces
    (headers, tables, blockquotes, bold, inline code, paragraphs, ``---``) —
    not a general-purpose parser. Kept in lock-step with the generator above
    so the HTML and Markdown reports never say different things.
    """
    lines = markdown.split("\n")
    out: list[str] = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        if stripped == "---":
            out.append("<hr>")
            i += 1
        elif stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            level = min(max(level, 1), 6)
            out.append(f"<h{level}>{_inline_md(stripped.lstrip('#').strip())}</h{level}>")
            i += 1
        elif stripped.startswith(">"):
            block = []
            while i < n and lines[i].strip().startswith(">"):
                block.append(_inline_md(lines[i].strip().lstrip(">").strip()))
                i += 1
            out.append(f"<blockquote>{' '.join(block)}</blockquote>")
        elif stripped.startswith("|"):
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            if len(rows) >= 2 and set(rows[1][0]) <= set("-: "):
                header, body = rows[0], rows[2:]
            else:
                header, body = rows[0], rows[1:]
            out.append("<table><thead><tr>" +
                        "".join(f"<th>{_inline_md(c)}</th>" for c in header) +
                        "</tr></thead><tbody>" +
                        "".join("<tr>" + "".join(f"<td>{_inline_md(c)}</td>" for c in r) + "</tr>" for r in body) +
                        "</tbody></table>")
        elif stripped.startswith("- "):
            items = []
            while i < n and lines[i].strip().startswith("- "):
                items.append(f"<li>{_inline_md(lines[i].strip()[2:])}</li>")
                i += 1
            out.append(f"<ul>{''.join(items)}</ul>")
        else:
            out.append(f"<p>{_inline_md(stripped)}</p>")
            i += 1
    return "\n".join(out)


def render_html(analysis: dict) -> str:
    """The same report as ``render_markdown``, as a self-contained HTML page.

    Converts the Markdown output rather than re-deriving the content, so the
    two formats cannot drift apart — there is exactly one place that decides
    what goes in a strategy report.
    """
    rs = analysis.get("race_state", {})
    title = f"Race Strategy Report — {rs.get('driver')} ({rs.get('team')})"
    body = _markdown_to_html_body(render_markdown(analysis))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{_html.escape(title)}</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; max-width: 860px;
          margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; line-height: 1.5; }}
  h1 {{ font-size: 1.6rem; border-bottom: 2px solid #1a1a1a; padding-bottom: .4rem; }}
  h2 {{ font-size: 1.25rem; margin-top: 2rem; border-bottom: 1px solid #ccc; padding-bottom: .3rem; }}
  h3 {{ font-size: 1.05rem; color: #b00020; }}
  table {{ border-collapse: collapse; width: 100%; margin: .75rem 0; font-size: .9rem; }}
  th, td {{ border: 1px solid #ddd; padding: .4rem .6rem; text-align: left; }}
  th {{ background: #f5f5f5; }}
  blockquote {{ border-left: 4px solid #b00020; margin: .75rem 0; padding: .3rem .8rem;
                 background: #fdf2f2; color: #444; }}
  code {{ background: #f0f0f0; padding: .1rem .3rem; border-radius: 3px; font-size: .85em; }}
  hr {{ border: none; border-top: 1px solid #ccc; margin: 1.5rem 0; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""
