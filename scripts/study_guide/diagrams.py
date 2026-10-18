"""
scripts.study_guide.diagrams
============================

The "Diagrams & Workflow" tab: the system's architecture and end-to-end
workflow as seven diagrams, in the order a viva walks through them.

Every diagram is drawn from the implementation, not from memory:

* the build pipeline is parsed from the ``@_stage`` decorators in
  ``scripts/build_all.py``, in file order (which is the order ``main`` runs);
* the validation timeline is the real split — the ML pipeline's own ``_split``
  is called, so the folds shown are the folds training uses;
* the deployment diagram reads ``docker-compose.yml``;
* the request workflow, recommendation flowchart and inference cycle follow
  ``strategy_service.run_strategy_analysis``, ``_combine_recommendation`` and
  ``InferenceEngine.forward_chain`` (each node names the function it shows).

Paths a diagram cites are checked to exist; a missing one is reported by the
builder rather than shown as if it were real.
"""
from __future__ import annotations

import html
import re
from pathlib import Path

from scripts.study_guide.icons import svg

REPO_ROOT = Path(__file__).resolve().parents[2]


def _e(text) -> str:
    return html.escape(str(text), quote=True)


def _code(text: str) -> str:
    return f"<code>{_e(text)}</code>"


# ---------------------------------------------------------------------------
# 1. System at a glance
# ---------------------------------------------------------------------------
LAYERS = [
    ("Presentation", "monitor", [
        ("Next.js dashboard", "frontend/", "Every value is fetched from the API; nothing is typed in.", ["Next.js", "React", "Tailwind"]),
        ("Strategy report", "app/services/strategy_report.py", "One race state → a Markdown or HTML briefing.", ["Markdown", "HTML"]),
    ]),
    ("API", "bolt", [
        ("FastAPI app", "app/api/main.py", "[[api_routers]] routers, Pydantic validation, CORS allow-list, read-only artifact mounts.", ["FastAPI", "Pydantic", "Uvicorn"]),
    ]),
    ("Services", "flow", [
        ("Strategy service", "app/services/strategy_service.py", "Runs the 8-stage pipeline and combines the verdicts.", ["Practical 9"]),
        ("Feature approximation", "app/services/feature_approximation.py", "Turns a race state into the exact feature row each model expects.", ["Practical 5 contract"]),
        ("Model cache", "app/services/model_cache.py", "Loads the registry's models once per process.", ["joblib"]),
    ]),
    ("Intelligence", "brain", [
        ("Ontology & graph", "app/intelligence/knowledge_representation/", "Domain vocabulary.", ["P1", "owlready2", "rdflib"]),
        ("Expert system", "app/intelligence/expert_system/", "[[es_rules]] rules, forward/backward chaining.", ["P2"]),
        ("Search", "app/intelligence/search/", "A* over the remaining laps.", ["P3"]),
        ("Machine learning", "app/intelligence/ml/", "[[ml_n_models]] classical models.", ["P6", "scikit-learn", "XGBoost"]),
        ("Deep learning", "app/intelligence/dl/", "Two Keras networks.", ["P7", "Keras", "PyTorch"]),
        ("Explainable AI", "app/intelligence/xai/", "SHAP, LIME, counterfactuals, trust.", ["P8", "SHAP", "LIME"]),
    ]),
    ("Data", "data", [
        ("Clean laps", "data/processed/fastf1_laps_clean.csv", "Practical 4's audited lap table.", ["P4", "pandas", "FastF1"]),
        ("Feature contract", "data/processed/feature_metadata.json", "Which features each model uses, and how.", ["P5"]),
        ("Artifacts", "artifacts/", "Models, metrics, figures and reports.", ["generated"]),
    ]),
]


def _architecture(fill) -> str:
    rows = []
    for name, icon, boxes in LAYERS:
        cells = "".join(
            f'<div class="dg-box"><strong>{_e(title)}</strong><span class="dg-path">{_code(path)}</span>'
            f'<span>{_e(fill(desc))}</span><span class="chips">'
            + "".join(f"<i>{_e(c)}</i>" for c in chips) + "</span></div>"
            for title, path, desc, chips in boxes)
        rows.append(f'<div class="dg-layer"><div class="dg-layer-name">{svg(icon, 22)}<span>{_e(name)}</span></div>'
                    f'<div class="dg-boxes">{cells}</div></div>')
    return '<div class="dg-arch">' + '<div class="dg-down" aria-hidden="true"></div>'.join(rows) + "</div>"


# ---------------------------------------------------------------------------
# 2. How the artifacts are built (scripts/build_all.py)
# ---------------------------------------------------------------------------
# What each stage reads and writes. Keyed by the task number in the stage's own
# name, so a renamed or reordered stage still lines up.
STAGE_IO = {
    1: ("app/intelligence/knowledge_representation/schema.py", "artifacts/knowledge_representation/"),
    2: ("app/intelligence/expert_system/rule_base.py", "artifacts/expert_system/"),
    3: ("app/intelligence/search/problem.py", "artifacts/search/"),
    4: ("data/raw/", "data/processed/fastf1_laps_clean.csv"),
    5: ("data/processed/fastf1_laps_clean.csv", "data/processed/feature_metadata.json"),
    6: ("data/processed/f1_features_selected.csv", "artifacts/metadata/model_registry.json"),
    7: ("data/processed/f1_features_selected.csv", "artifacts/deep_learning/"),
    8: ("artifacts/models/", "artifacts/xai/"),
}


def build_stages() -> list[str]:
    """Stage names from ``@_stage("...")`` in scripts/build_all.py, in file order."""
    text = (REPO_ROOT / "scripts" / "build_all.py").read_text()
    return re.findall(r'@_stage\("([^"]+)"\)', text)


def _pipeline(missing: list[str]) -> str:
    items = []
    for name in build_stages():
        m = re.search(r"\(Task (\d+)\)", name)
        n = int(m.group(1)) if m else None
        io = STAGE_IO.get(n)
        label = re.sub(r"\s*\(Task \d+\)", "", name)
        if io:
            for p in io:
                if not (REPO_ROOT / p).exists():
                    missing.append(f"diagram path: {p}")
            io_html = (f'<span class="dg-io"><em>in</em>{_code(io[0])}</span>'
                       f'<span class="dg-io"><em>out</em>{_code(io[1])}</span>')
        else:
            io_html = '<span class="dg-io muted">checks only</span>'
        badge = f'<a class="dg-num" href="#p-{n}" title="Practical {n}">{n}</a>' if n else '<span class="dg-num ghost">✓</span>'
        items.append(f'<li>{badge}<div><strong>{_e(label)}</strong>{io_html}</div></li>')
    return (f'<ol class="dg-stages">{"".join(items)}</ol>'
            '<div class="dg-decision">' + svg("decision", 20) +
            '<span><strong>Every stage:</strong> outputs already on disk? → skip it, unless '
            f'{_code("--force")}. Tasks 7–8 refuse to run when Task 6 has no results to compare against.</span></div>')


# ---------------------------------------------------------------------------
# 3. Time-aware validation (the real split)
# ---------------------------------------------------------------------------
def _validation() -> str:
    from app.intelligence.ml import pipeline
    from app.intelligence.ml.data_contract import load_and_validate

    _, holdout, folds, X_dev, _, X_test, _ = pipeline._split(load_and_validate(), "target_laptime")
    lo, hi = holdout.dev_laps[0], holdout.test_laps[-1]
    span = hi - lo + 1
    W, label_w, row_h = 760, 120, 30

    def x(lap):  # left edge of a lap
        return label_w + (lap - lo) / span * (W - label_w)

    def bar(y, a, b, cls, text):
        x0, x1 = x(a), x(b + 1)
        return (f'<rect class="{cls}" x="{x0:.1f}" y="{y}" width="{x1 - x0:.1f}" height="{row_h - 8}" rx="3"/>'
                f'<text class="bt" x="{(x0 + x1) / 2:.1f}" y="{y + row_h / 2 + 1}">{_e(text)}</text>')

    rows, y = [], 26
    for f in folds:
        rows.append(f'<text class="rl" x="0" y="{y + row_h / 2 + 1}">Fold {f.fold_id}</text>')
        rows.append(bar(y, f.train_laps[0], f.train_laps[-1], "tr", f"train {len(f.train_index)}"))
        rows.append(bar(y, f.val_laps[0], f.val_laps[-1], "va", f"val {len(f.val_index)}"))
        y += row_h
    y += 8
    rows.append(f'<text class="rl" x="0" y="{y + row_h / 2 + 1}">Final model</text>')
    rows.append(bar(y, holdout.dev_laps[0], holdout.dev_laps[-1], "tr", f"development {len(X_dev)} rows"))
    rows.append(bar(y, holdout.test_laps[0], holdout.test_laps[-1], "te", f"test {len(X_test)}"))
    y += row_h
    ticks = "".join(f'<text class="tk" x="{x(lap):.1f}" y="14">{lap}</text>'
                    for lap in range(lo, hi + 1) if lap in (lo, hi) or lap % 10 == 0)
    height = y + 6
    return (f'<figure class="dg-timeline"><svg viewBox="0 0 {W} {height}" role="img" '
            f'aria-label="Four expanding-window folds over laps {lo} to {holdout.dev_laps[-1]}, then a final fit '
            f'tested once on laps {holdout.test_laps[0]} to {hi}">'
            f'<text class="tk" x="0" y="14">Lap →</text>{ticks}{"".join(rows)}</svg>'
            '<figcaption><span class="lgd tr"></span>train <span class="lgd va"></span>validate '
            '<span class="lgd te"></span>test — row counts are lap rows (all drivers). '
            f'Computed by {_code("ml.pipeline._split")} when this page was built.</figcaption></figure>')


# ---------------------------------------------------------------------------
# 4. One strategy request
# ---------------------------------------------------------------------------
REQUEST_STAGES = [
    ("Validate", "RaceStateRequest + dataset options", "race state (driver, team, lap, tyre, weather…)",
     "422 if inconsistent", "app/api/schemas.py", "Pydantic"),
    ("Construct features", "build_feature_row", "race state", "one feature row per model + out-of-range flags",
     "app/services/feature_approximation.py", "Practical 5 contract"),
    ("ML prediction", "_run_ml", "feature rows", "lap time, P(pit), class at 0.5", "app/services/strategy_service.py", "Practical 6"),
    ("Expert system", "_run_expert_system", "16 derived race facts", "rules fired, pit_decision?",
     "app/services/strategy_service.py", "Practical 2"),
    ("DL prediction", "_run_dl", "same feature rows", "lap time, P(pit) at tuned threshold",
     "app/services/strategy_service.py", "Practical 7"),
    ("Search plan", "_run_search", "remaining laps, compound, track temp", "A* plan + expected cost",
     "app/services/strategy_service.py", "Practical 3"),
    ("Recommend", "_combine_recommendation", "rules, ML, DL, plan", "STAY OUT / PIT NOW / PIT IN N LAPS",
     "app/services/strategy_service.py", "Practical 9"),
    ("Explain (opt-in)", "explain_feature_row", "feature rows (explain=true)", "SHAP factors + trust score",
     "app/intelligence/xai/live.py", "Practical 8"),
]


def _request() -> str:
    cards = "".join(
        f'<li><span class="dg-step">{i}</span><strong>{_e(name)}</strong>'
        f'<span class="dg-fn">{_code(fn)}</span>'
        f'<span class="dg-io"><em>in</em>{_e(inp)}</span><span class="dg-io"><em>out</em>{_e(out)}</span>'
        f'<span class="chips"><i>{_e(tech)}</i></span></li>'
        for i, (name, fn, inp, out, _path, tech) in enumerate(REQUEST_STAGES, 1))
    return (f'<div class="dg-endpoint">{svg("bolt", 18)}{_code("POST /api/strategy/predict")} → '
            f'{_code("run_strategy_analysis")}</div><ol class="dg-flow">{cards}</ol>')


# ---------------------------------------------------------------------------
# 5. How the recommendation is decided (_combine_recommendation)
# ---------------------------------------------------------------------------
def _decision() -> str:
    def q(text):
        return f'<div class="dq">{svg("decision", 18)}<span>{text}</span></div>'

    def leaf(text, conf, cls=""):
        return f'<div class="dl {cls}"><strong>{text}</strong><span>confidence: {conf}</span></div>'

    return f"""
<div class="dtree">
  {q(f'Did a rule assert {_code("pit_decision")}?')}
  <div class="dbr">
    <div class="dcol"><span class="dyes">yes</span>{leaf("Use the rule's verdict", "high", "rule")}</div>
    <div class="dcol"><span class="dno">no</span>
      {q("Are both ML and DL pit predictions available?")}
      <div class="dbr">
        <div class="dcol"><span class="dyes">yes</span>{q("Do they predict the same class?")}
          <div class="dbr">
            <div class="dcol"><span class="dyes">yes</span>{leaf("Use the shared class", "high if both are ≥ 0.1 from their cut-off, else moderate")}</div>
            <div class="dcol"><span class="dno">no</span>{leaf("Use ML (tie-break), flag disagreement", "low", "warn")}</div>
          </div></div>
        <div class="dcol"><span class="dno">no</span>{leaf("Use whichever one exists", "moderate · none if neither")}</div>
      </div></div>
  </div>
  <div class="dg-down" aria-hidden="true"></div>
  {q("Is the verdict “pit”?")}
  <div class="dbr">
    <div class="dcol"><span class="dyes">yes</span>{leaf("PIT NOW — or PIT IN N LAPS, where N is the first PIT step of the A* plan", "")}</div>
    <div class="dcol"><span class="dno">no</span>{leaf("STAY OUT", "")}</div>
  </div>
</div>"""


# ---------------------------------------------------------------------------
# 6. Expert-system inference cycle (forward_chain)
# ---------------------------------------------------------------------------
def _inference(fill) -> str:
    steps = [
        ("data", "Working memory", "input facts, each with provenance"),
        ("flow", "Match", "every rule whose conditions hold → conflict set"),
        ("layers", "Resolve", "salience › specificity › rule id"),
        ("bolt", "Fire", "assert the actions; a repeat signature is skipped"),
    ]
    nodes = "".join(f'<li>{svg(i, 22)}<strong>{_e(t)}</strong><span>{_e(d)}</span></li>' for i, t, d in steps)
    return (f'<div class="dg-cycle"><ol>{nodes}</ol>'
            f'<div class="dg-loop">{svg("loop", 22)}<span><strong>Memory changed?</strong> yes → match again · '
            f'no → fixed point (stop). Capped at 200 iterations. [[es_rules]] rules.</span></div></div>').replace(
        "[[es_rules]]", fill("[[es_rules]]"))


# ---------------------------------------------------------------------------
# 7. Deployment (docker-compose.yml)
# ---------------------------------------------------------------------------
def _host_port(mapping: str) -> tuple[str, str]:
    host, _, container = str(mapping).rpartition(":")
    m = re.search(r":-(\d+)\}", host)
    return (m.group(1) if m else host), container


def _deployment() -> str:
    try:
        import yaml
        compose = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text())["services"]
    except Exception as exc:  # pragma: no cover - the guide still builds without it
        return f'<p class="muted">docker-compose.yml could not be read: {_e(exc)}</p>'
    fe, be = compose["frontend"], compose["backend"]
    fe_host, fe_ct = _host_port(fe["ports"][0])
    be_host, be_ct = _host_port(be["ports"][0])
    health = " ".join(be.get("healthcheck", {}).get("test", [])[-1:])
    mounts = "".join(f"<li>{_code(v.split(':')[0])} → {_code(v.split(':')[1])}</li>" for v in be.get("volumes", []))
    return f"""
<div class="dg-deploy">
  <div class="dd-node"><strong>{svg("monitor", 20)} Browser</strong><span>localhost:{_e(fe_host)}</span></div>
  <div class="dd-arrow"><span>HTTP</span></div>
  <div class="dd-node box"><strong>{svg("box", 20)} frontend</strong><span>Next.js · container port {_e(fe_ct)}</span>
    <span class="muted">starts once backend is healthy</span></div>
  <div class="dd-arrow"><span>fetch /api/*</span></div>
  <div class="dd-node box"><strong>{svg("box", 20)} backend</strong><span>FastAPI · host {_e(be_host)} → {_e(be_ct)}</span>
    <span class="muted">health: {_code(health)}</span></div>
  <div class="dd-arrow"><span>bind mounts</span></div>
  <div class="dd-node"><strong>{svg("data", 20)} Host folders</strong><ul>{mounts}</ul></div>
</div>
<p class="dg-note">Images hold code only; the containers serve whatever {_code("scripts/build_all.py")} last wrote
to {_code("artifacts/")} and {_code("data/")}. Without Docker, {_code("./run.sh")} starts the same two servers directly.</p>"""


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def render_diagrams_tab(fill, missing: list[str]) -> str:
    sections = [
        ("arch", "layers", "System at a glance",
         "Five layers. Requests go down, data and artifacts come up; no layer skips another.",
         _architecture(fill),
         ["The intelligence modules never talk to the browser — only the API does.",
          "Models are loaded from artifacts/, never trained per request."],
         [("Practical 9", "#p-9")]),
        ("build", "flow", "How every artifact is built",
         f"{_code('python scripts/build_all.py')} runs these stages in order; each one's output is the next one's input.",
         _pipeline(missing),
         ["Practicals 1–3 are symbolic and need no data; 4 → 5 → 6 → 7 → 8 is a strict chain.",
          "The quantum stage reuses Practical 6's split and metrics, so the comparison is fair."],
         [("Practical 4", "#p-4"), ("Practical 5", "#p-5"), ("Practical 6", "#p-6")]),
        ("validation", "calendar", "Time-aware validation",
         "A race is a time series, so every model trains on earlier laps and is judged on later ones.",
         _validation(),
         ["A shuffled K-fold would train on lap 50 and validate on lap 10 — leaking the future.",
          "Folds choose hyperparameters and thresholds; the test laps are used once, at the end.",
          "Practicals 6, 7, 8 and the quantum models all use this same split."],
         [("Practical 6", "#p-6"), ("Practical 7", "#p-7")]),
        ("request", "bolt", "One strategy request, end to end",
         "Eight stages turn one race state into one recommendation, with every stage's evidence attached.",
         _request(),
         ["Rules, ML and DL each see the same race state; disagreement is reported, never hidden.",
          "Stage 8 is opt-in because sampled SHAP adds about a second per target."],
         [("Practical 9", "#p-9"), ("Practical 8", "#p-8")]),
        ("decision", "decision", "How the recommendation is decided",
         f"The decision rule inside {_code('_combine_recommendation')}.",
         _decision(),
         ["Rules win because they encode knowledge one dry race cannot teach a model (safety cars, weather).",
          "Cut-offs differ: DL uses its tuned threshold; the live ML class comes from predict(), i.e. 0.5."],
         [("Practical 2", "#p-2"), ("Practical 3", "#p-3")]),
        ("inference", "loop", "Inside the expert system",
         f"{_code('InferenceEngine.forward_chain')}: data-driven inference to a fixed point.",
         _inference(fill),
         ["Every derived fact records the rule that produced it — that is the explanation.",
          "Backward chaining proves a goal the other way round and returns a proof tree."],
         [("Practical 2", "#p-2")]),
        ("deploy", "rocket", "Deployment",
         f"{_code('docker compose up')} — read from {_code('docker-compose.yml')}.",
         _deployment(),
         ["The frontend waits for the backend's health check before starting."],
         [("Practical 9", "#p-9"), ("Practical 10", "#p-10")]),
    ]
    toc = "".join(f'<a href="#dg-{sid}">{svg(icon, 16)}<span>{i}. {_e(title)}</span></a>'
                  for i, (sid, icon, title, *_rest) in enumerate(sections, 1))
    body = []
    for i, (sid, icon, title, lead, diagram, points, links) in enumerate(sections, 1):
        pts = "".join(f"<li>{_e(p)}</li>" for p in points)
        deeper = " ".join(f'<a class="chip-link" href="{h}">{_e(t)} {svg("chevron", 14)}</a>' for t, h in links)
        body.append(
            f'<section class="dg-section" id="dg-{sid}"><header><span class="dg-icon">{svg(icon, 22)}</span>'
            f'<div><p class="crumbs">Diagram {i} of {len(sections)}</p><h2>{_e(title)}</h2>'
            f'<p class="dg-lead">{lead}</p></div></header>'
            f'<div class="dg-canvas">{diagram}</div>'
            f'<div class="dg-foot"><ul class="dg-points">{pts}</ul><div class="dg-deeper">{deeper}</div></div></section>')
    return f"""
<section class="tab-panel diagrams-tab" id="tab-diagrams" data-tab="diagrams" role="tabpanel" aria-labelledby="tabbtn-diagrams" hidden>
  <article class="dg-page" id="diagrams" data-view data-crumb="Diagrams &amp; Workflow" data-parent="home">
    <header class="practical-head"><p class="crumbs">Diagrams &amp; Workflow</p>
      <h1>How the system fits together</h1>
      <p class="lead">Seven diagrams, in the order to explain them: the layers, how artifacts are built,
      how models are validated, what one request does, how it decides, the rule engine, and deployment.</p>
      <nav class="dg-toc" aria-label="Diagrams">{toc}</nav>
    </header>
    {''.join(body)}
  </article>
</section>"""
