#!/usr/bin/env python3
"""
build_task_summary.py
=====================

Renders ``docs/task_summary.png``: the one-page "10 tasks summary" infographic
for this project — one row per lab task with its problem statement, inputs,
steps, expected output and deliverables, a workflow strip and the key benefits.

The layout follows the lab's reference summary sheet; the content is this
project's. Every number on it is read from the code or a committed artifact
through ``build_study_guide.collect_facts`` (the same facts the study guide
uses), so a retrain changes the sheet on the next build instead of leaving it
wrong. The deliverable names follow ``docs/DELIVERABLES_CHECKLIST.html``.

Dev-only: needs Playwright with Chromium (``pip install -r requirements-dev.txt``
then ``playwright install chromium``).

Usage
-----
    python scripts/build_task_summary.py                  # docs/task_summary.png
    python scripts/build_task_summary.py --html out.html  # also keep the HTML source
"""
from __future__ import annotations

import argparse
import html
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.build_study_guide import collect_facts, fill  # noqa: E402
from scripts.study_guide.icons import ICONS  # noqa: E402

DEFAULT_OUTPUT = _REPO_ROOT / "docs" / "task_summary.png"

# Column header colours, left to right (as on the reference sheet).
COLUMNS = [("Task", "#1d2b6b"), ("Problem statement", "#139a96"), ("Input provided / required", "#ec7d10"),
           ("Steps involved", "#6a3d9a"), ("Expected output", "#2f9a48"), ("Deliverables", "#a3246f")]


# One row per task. Steps and deliverables follow the lab sheet's wording where
# the project does the same thing, and say what the project actually did.
TASKS = [
    dict(n=1, name="Knowledge Representation", color="#1d2b6b", icon="network",
         problem="Represent the F1 race-strategy domain with entities, relationships, an ontology and a "
                 "knowledge graph.",
         inputs=(["Problem statement", "F1 domain (FastF1 + Ergast schemas)", "Race workflow: stints, pit stops"],
                 "doc"),
         steps=["Domain & stakeholder analysis", "Entity identification ([[kr_entities]])", "Attribute definition",
                "Relationships & cardinality ([[kr_relationships]])", "OWL ontology (owlready2)",
                "Knowledge graph (NetworkX, RDF)", "Validation"],
         output=("Structured F1 knowledge base of the race domain.", "network"),
         deliverables=["Entity Table", "Attribute Table", "Relationship Table", "Ontology (.owl)",
                       "Knowledge Graph (GraphML, TTL)", "Validation Report"]),
    dict(n=2, name="Rule-Based Expert System", color="#2f9a48", icon="ifthen",
         problem="Recommend pit, tyre and pace decisions with IF–THEN production rules.",
         inputs=(["Knowledge base from Task 1", "Race state: lap, tyre, weather, track status"], "folder"),
         steps=["Knowledge acquisition", "Rule formulation ([[es_rules]] rules)", "Inference engine",
                "Forward chaining", "Backward chaining", "Rule validation", "Explanation generation"],
         output=("Rule-based strategy call with a full explanation trace.", "laptop"),
         deliverables=["Rule Base (JSON)", "Rule Tree", "Decision Table", "Inference Traces (5 scenarios)",
                       "Rule Validation Report"]),
    dict(n=3, name="State Space Search", color="#ec7d10", icon="tree",
         problem="Model pit strategy as a search problem and find the minimum-time race plan.",
         inputs=(["Expert-system rules", "Tyre, fuel and pit-loss cost model"], "stopwatch"),
         steps=["Define state space", "Initial & goal state (2-compound rule)", "State transitions: run / pit",
                "BFS, DFS, UCS, Greedy, A*", "Cost function & admissible heuristic", "Performance comparison",
                "Validation"],
         output=("Optimal pit plan: A* = UCS at [[search_astar_cost]] s.", "flagpath"),
         deliverables=["State-Space Formulation", "Search Tree Diagram", "BFS / DFS / UCS / A* Results",
                       "Heuristic Table", "Comparison Report"]),
    dict(n=4, name="Data Preparation & Exploratory Data Analysis", color="#1d2b6b", icon="database",
         problem="Prepare high-quality race data for AI model development.",
         inputs=(["Real FastF1 laps ([[data_year]] [[data_event]] GP)", "Kaggle-style history tables (synthetic)"],
                 "table"),
         steps=["Data loading (schema-validated)", "Data understanding", "Missing-value handling",
                "Duplicate removal", "Outlier detection (IQR cap)", "Encoding & scaling",
                "EDA & visualization (9 analyses)", "Data-quality assessment"],
         output=("Clean, audited lap dataset: [[data_n_laps]] laps, [[data_n_drivers]] drivers.", "pie"),
         deliverables=["Clean Dataset", "Cleaning Audit", "EDA Report", "Statistical Summary",
                       "Visualization Report", "Data Quality Report"]),
    dict(n=5, name="Feature Engineering & Feature Selection", color="#2f9a48", icon="gears",
         problem="Engineer race-strategy features and select the most predictive, leakage-free set.",
         inputs=(["Clean laps from Task 4"], "table"),
         steps=["Derived features: tyre life, stint, pace gaps", "Field pace & weather interactions",
                "Targets: lap time, pit next lap", "Encoding: compound, team, driver",
                "Leakage exclusion (sector times)", "Correlation & VIF pruning",
                "Feature importance (MI + RF + L1)", "Feature selection (K sweep)"],
         output=("Feature contract: [[feat_reg]] lap-time / [[feat_clf]] pit features, [[feat_rows]] rows.",
                 "heatmap"),
         deliverables=["Feature Engineering Report", "Feature Matrix (CSV)", "Correlation Matrix",
                       "Feature Importance Report", "Selected Features (JSON)"]),
    dict(n=6, name="Machine Learning Model Development", color="#139a96", icon="brain",
         problem="Train and compare classical models for lap time and the pit decision.",
         inputs=(["Feature matrix from Task 5"], "table"),
         steps=["Chronological train–test split", "Lap-forward CV folds", "Linear / Logistic Regression",
                "Decision Tree", "Random Forest", "SVM / SVR", "XGBoost",
                "Threshold tuning, selection & saving"],
         output=("Best models: [[ml_reg_best]] (lap time), [[ml_clf_best]] (pit).", "trophy"),
         deliverables=["ML Models (.joblib)", "Evaluation Report", "ROC & PR Curves", "Confusion Matrix",
                       "Model Comparison Report"]),
    dict(n=7, name="Deep Learning Model Development", color="#6a3d9a", icon="neural",
         problem="Design, train and evaluate deep neural networks on the same folds and holdout as Task 6.",
         inputs=(["Feature matrix from Task 5", "Task 6 folds & holdout"], "table"),
         steps=["Network design (Keras MLP)", "Hyperparameter search", "Model training (early stopping)",
                "Validation", "Overfitting prevention (dropout + L2)", "Performance evaluation",
                "Model saving (.h5, reload-verified)"],
         output=("Lap-time DNN: test MAE [[dl_reg_mae]] s, R² [[dl_reg_r2]].", "neural"),
         deliverables=["DNN Models (.h5)", "Training History", "Accuracy / Loss Curves", "Evaluation Report",
                       "Hyperparameter Report"]),
    dict(n=8, name="Explainable Artificial Intelligence (XAI)", color="#a3246f", icon="xai",
         problem="Explain every prediction so a race engineer can trust or reject it.",
         inputs=(["Trained ML / DL models & test laps"], "monitor"),
         steps=["Feature importance (permutation)", "SHAP analysis (Tree + Kernel)", "LIME explanation",
                "Counterfactuals (tyre-age scan, DiCE)", "Trust score", "Fairness: driver, team, compound",
                "Strategy explanation"],
         output=("Transparent, interpretable strategy predictions.", "shield"),
         deliverables=["SHAP Report", "LIME Report", "Counterfactual Analysis", "Explainability Dashboard",
                       "Trust Score Report", "Fairness Report"]),
    dict(n=9, name="System Integration & Deployment", color="#1f6fc0", icon="puzzle",
         problem="Integrate every module into one race-strategy decision system.",
         inputs=(["Outputs of Tasks 1–8"], "cubes"),
         steps=["Input validation", "Feature construction", "Expert-system execution", "ML prediction",
                "DL prediction", "A* search plan", "XAI explanation", "Recommendation engine",
                "Report generation & UI integration"],
         output=("Strategy assistant: STAY OUT / PIT NOW / PIT IN N LAPS.", "monitor"),
         deliverables=["Next.js + FastAPI App (Streamlit equivalent)", "Integrated Source Code", "Dashboard",
                       "Strategy Report Generator", "Deployment Docs & Docker"]),
    dict(n=10, name="System Evaluation, Responsible AI & Documentation", color="#1d2b6b", icon="clipshield",
         problem="Evaluate the integrated system for performance, domain validity, explainability and "
                 "responsible AI.",
         inputs=(["Complete integrated system"], "monitor"),
         steps=["Functional testing (pytest)", "Performance evaluation (latency)", "Domain validation (race backtest)",
                "Explainability validation", "Responsible-AI assessment", "Documentation",
                "Deployment readiness", "Future enhancement"],
         output=("Evaluated, deployment-ready system: [[eval_passed]]/[[eval_total]] tests passed.", "target"),
         deliverables=["Evaluation Report", "Responsible AI Report", "User Manual", "SRS, SDD", "Testing Report",
                       "Final Project Report", "Deployment Guide"]),
]

JOURNEY = ["Knowledge Representation", "Expert System", "State Space Search", "Data Preparation & EDA",
           "Feature Engineering", "Machine Learning", "Deep Learning", "Explainable AI", "Integration",
           "Evaluation & Deployment"]

BENEFITS = ["Data-driven pit-stop timing", "Explainable, auditable strategy calls",
            "Rules + ML + DL + search in one decision", "No fabricated numbers — all from artifacts",
            "Reproducible, containerised deployment"]


def icon(name: str, color: str, size: int = 44, stroke: float = 1.6) -> str:
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" '
            f'stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
            f'{ICONS[name]}</svg>')


def _bullets(items, facts) -> str:
    return "<ul>" + "".join(f"<li>{html.escape(fill(i, facts))}</li>" for i in items) + "</ul>"


def render_html(facts: dict) -> str:
    head = "".join(f'<div class="h" style="background:{c}">{html.escape(t)}</div>' for t, c in COLUMNS)
    rows = []
    for t in TASKS:
        c = t["color"]
        inputs, in_icon = t["inputs"]
        out_text, out_icon = t["output"]
        rows.append(f"""
<div class="row" style="--c:{c}">
  <div class="cell task"><div class="tnum">{t['n']}</div><div class="tname">{html.escape(t['name'])}</div>
    <div class="ticon">{icon(t['icon'], c, 52)}</div></div>
  <div class="cell">{html.escape(fill(t['problem'], facts))}</div>
  <div class="cell">{_bullets(inputs, facts)}<div class="cicon">{icon(in_icon, c, 40)}</div></div>
  <div class="cell steps">{_bullets(t['steps'], facts)}<span class="arrow" style="color:{c}">➜</span></div>
  <div class="cell out"><p>{html.escape(fill(out_text, facts))}</p><div class="cicon">{icon(out_icon, c, 46)}</div>
    <span class="arrow" style="color:{c}">➜</span></div>
  <div class="cell deliv">{_bullets(t['deliverables'], facts)}<div class="dicon">{icon('report', c, 34)}</div></div>
</div>""")
    journey = "".join(
        f'<div class="j"><div class="jc" style="--c:{t["color"]}">{icon(t["icon"], t["color"], 26)}</div>'
        f'<span>{t["n"]}<br>{html.escape(label)}</span></div>'
        + ('<div class="ja">➜</div>' if t["n"] < 10 else "")
        for t, label in zip(TASKS, JOURNEY))
    benefits = "".join(f"<li>{html.escape(b)}</li>" for b in BENEFITS)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>10 Tasks Summary</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700;800&family=Inter:wght@400;500;600&display=swap">
<style>
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: #fff; font-family: Inter, "Helvetica Neue", Arial, sans-serif; color: #1c1f2a; }}
.sheet {{ width: 1200px; padding: 22px 18px 18px; background: #fff; }}
header {{ position: relative; text-align: center; margin-bottom: 14px; }}
header h1 {{ margin: 0; font: 800 36px/1.05 "Barlow Condensed", Arial Narrow, sans-serif; letter-spacing: .02em; color: #1d2b6b; }}
header h2 {{ margin: 4px 0 0; font: 800 30px/1.1 "Barlow Condensed", Arial Narrow, sans-serif; color: #1d2b6b; }}
header h2 em {{ font-style: normal; color: #a3246f; }}
.hero {{ position: absolute; right: 8px; top: 4px; display: flex; align-items: center; gap: 4px; }}
.hero svg {{ display: block; }}
.speed {{ width: 70px; height: 3px; background: linear-gradient(90deg, transparent, #e10600); border-radius: 2px; box-shadow: 0 8px 0 -0.5px #e1060088, 0 -8px 0 -0.5px #e1060055; }}
.grid {{ display: grid; grid-template-columns: 170px 175px 175px 245px 175px 1fr; column-gap: 6px; }}
.h {{ color: #fff; font: 700 15px/1.15 "Barlow Condensed", sans-serif; text-transform: uppercase; letter-spacing: .03em;
     text-align: center; padding: 10px 6px; border-radius: 6px 6px 0 0; display: flex; align-items: center; justify-content: center; min-height: 46px; }}
.row {{ display: contents; }}
.cell {{ position: relative; border-top: 2px solid var(--c); background: color-mix(in srgb, var(--c) 5%, #fff);
        padding: 9px 9px 10px; font-size: 12.4px; line-height: 1.38; margin-top: 6px; }}
.cell:first-child {{ border-left: 4px solid var(--c); }}
.row .cell:nth-child(1) {{ border-radius: 6px 0 0 6px; }}
.task {{ display: grid; grid-template-columns: 30px 1fr; grid-template-rows: auto 1fr; gap: 4px 8px; align-items: start; }}
.tnum {{ width: 30px; height: 30px; border-radius: 50%; background: var(--c); color: #fff; display: grid; place-items: center;
        font: 800 17px "Barlow Condensed", sans-serif; }}
.tname {{ font: 800 14.5px/1.15 "Barlow Condensed", sans-serif; text-transform: uppercase; color: var(--c); letter-spacing: .01em; padding-top: 2px; }}
.ticon {{ grid-column: 1 / -1; padding: 4px 0 0 6px; }}
ul {{ margin: 0; padding-left: 14px; }}
li {{ margin: 0 0 1.5px; }}
li::marker {{ color: var(--c); }}
.cicon {{ margin-top: 6px; }}
.out p {{ margin: 0 22px 6px 0; font-weight: 500; }}
.out .cicon {{ text-align: center; margin-top: 10px; }}
.steps {{ padding-right: 22px; }}
.arrow {{ position: absolute; right: -9px; top: 50%; transform: translateY(-50%); font-size: 18px; z-index: 2; }}
.deliv {{ padding-right: 46px; }}
.dicon {{ position: absolute; right: 8px; bottom: 8px; }}
footer {{ display: grid; grid-template-columns: 1fr 250px; gap: 10px; margin-top: 12px; }}
.journey {{ border: 2px solid #1d2b6b; border-radius: 8px; padding: 18px 10px 10px; position: relative; display: flex; align-items: flex-start; justify-content: space-between; }}
.jt, .bt {{ position: absolute; top: -12px; left: 12px; background: #1d2b6b; color: #fff; font: 700 13px "Barlow Condensed", sans-serif;
           letter-spacing: .05em; text-transform: uppercase; padding: 3px 12px; border-radius: 4px; }}
.j {{ display: flex; flex-direction: column; align-items: center; gap: 4px; width: 76px; text-align: center; font-size: 10.5px; line-height: 1.2; font-weight: 600; color: #1d2b6b; }}
.jc {{ width: 42px; height: 42px; border-radius: 50%; border: 2px solid var(--c); display: grid; place-items: center; background: color-mix(in srgb, var(--c) 8%, #fff); }}
.ja {{ color: #1d2b6b; font-size: 14px; padding-top: 12px; }}
.benefits {{ border: 2px solid #e10600; border-radius: 8px; padding: 18px 12px 10px; position: relative; background: #fff7f5; }}
.bt {{ background: #e10600; left: auto; right: 12px; }}
.benefits ul {{ list-style: none; padding: 0; font-size: 11.6px; font-weight: 500; }}
.benefits li {{ display: flex; gap: 6px; margin-bottom: 4px; }}
.benefits li::before {{ content: "✓"; color: #e10600; font-weight: 700; }}
</style></head>
<body><div class="sheet">
<header>
  <h1>COMPUTATIONAL INTELLIGENCE LABORATORY</h1>
  <h2>10 TASKS SUMMARY – <em>F1 RACE STRATEGY INTELLIGENCE</em></h2>
  <div class="hero"><span class="speed"></span>{icon("flag", "#1d2b6b", 54, 1.4)}</div>
</header>
<div class="grid">{head}{''.join(rows)}</div>
<footer>
  <div class="journey"><span class="jt">Complete workflow journey</span>{journey}</div>
  <div class="benefits"><span class="bt">Key benefits</span><ul>{benefits}</ul></div>
</footer>
</div></body></html>"""


def build(output: Path = DEFAULT_OUTPUT, html_out: Path | None = None) -> Path:
    from playwright.sync_api import sync_playwright

    page_html = render_html(collect_facts())
    if html_out:
        Path(html_out).write_text(page_html, encoding="utf-8")
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "summary.html"
        src.write_text(page_html, encoding="utf-8")
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1200, "height": 1000}, device_scale_factor=2)
            page.goto(src.as_uri())
            page.wait_for_load_state("networkidle")
            page.evaluate("document.fonts.ready")
            page.locator(".sheet").screenshot(path=str(output))
            browser.close()
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--html", type=Path, default=None, help="Also write the HTML source here.")
    args = parser.parse_args()
    out = build(args.output, args.html)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
