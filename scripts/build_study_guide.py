#!/usr/bin/env python3
"""
build_study_guide.py
====================

Builds ``docs/study_guide.html``: one self-contained page with five tabs.

* **Study Guide** — a short hub: key facts, a study path and the ten-task
  summary sheet.
* **Diagrams & Workflow** — the architecture and end-to-end workflow as seven
  diagrams drawn from the code and config (``scripts/study_guide/diagrams.py``).
* **Reports** — every Markdown report the pipelines write under ``artifacts/``,
  discovered at build time and grouped by category, with headings, tables,
  code and figures rendered and working cross-links.
* **Showcase** — the project as a slide deck, with a full-screen mode.
* **Understand the Code** — a walkthrough of the ten practicals.

The reports and the source code stay the source of truth: they are read, never
rewritten, and the page is regenerated from them. The guide states only what the
repository can back up:

* every number comes from the code or a committed artifact
  (:func:`collect_facts`) and every code excerpt is pulled from the source
  through :mod:`ast` (:func:`extract_source`), so a renamed function or a
  missing artifact fails the build instead of shipping a stale page;
* links between documents are rewritten to in-page anchors, links to other
  repository files to paths relative to the output, and a link whose target
  does not exist is reported and rendered inert rather than left to 404.

Usage
-----
    python scripts/build_study_guide.py                 # write docs/study_guide.html
    python scripts/build_study_guide.py --no-live       # skip running the strategy pipeline
    python scripts/build_study_guide.py --output /tmp/guide.html

Needs the dev-only ``markdown`` package (``pip install -r requirements-dev.txt``);
nothing at runtime imports it.
"""
from __future__ import annotations

import argparse
import ast
import html
import json
import os
import re
import sys
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import markdown  # noqa: E402

from scripts.study_guide.content import PRACTICALS, SLIDES  # noqa: E402
from scripts.study_guide.diagrams import render_diagrams_tab  # noqa: E402
from scripts.study_guide.icons import svg  # noqa: E402

try:  # Pygments ships with pytest, so it is normally present; highlighting is optional.
    from pygments import highlight
    from pygments.formatters import HtmlFormatter
    from pygments.lexers import get_lexer_by_name
except ImportError:  # pragma: no cover
    highlight = None

REPO_ROOT = _REPO_ROOT
ASSET_DIR = Path(__file__).resolve().parent / "study_guide"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "study_guide.html"

# Report categories: one per artifacts/ sub-folder, in study order. Reports
# are *discovered* (every artifacts/**/*.md at build time); this table only
# names and describes the folders. A folder not listed here still appears,
# under its own name, so a new pipeline's reports are never dropped.
REPORT_CATEGORIES: dict[str, tuple[str, int | None, str, str]] = {
    "knowledge_representation": ("Knowledge representation", 1, "network",
                                 "Entity, attribute and relationship tables, the ontology and its validation."),
    "expert_system": ("Expert system", 2, "ifthen",
                      "The rule base, decision table, worked inference traces and rule validation."),
    "search": ("State-space search", 3, "tree",
               "The search formulation, the admissible heuristic and the five-algorithm comparison."),
    "data_engineering": ("Data engineering & EDA", 4, "database",
                         "The cleaning audit, data quality, statistics, correlations and the nine analyses."),
    "feature_engineering": ("Feature engineering", 5, "gears",
                            "How candidate features were built and selected, with importance and stability."),
    "reports": ("Machine learning & quantum", 6, "brain",
                "Model selection and evaluation for both targets, and the classical-vs-quantum comparison."),
    "deep_learning": ("Deep learning", 7, "neural",
                      "Network evaluation on train/validation/test and every hyperparameter tried."),
    "xai": ("Explainable AI", 8, "xai",
            "SHAP, LIME, counterfactuals, the trust score, fairness and the explainability dashboard."),
}

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def esc(text) -> str:
    return html.escape(str(text), quote=True)


def gh_slug(text: str) -> str:
    """GitHub's heading-anchor rule, so links written for GitHub keep working."""
    text = re.sub(r"[^\w\- ]", "", text.strip().lower())
    return text.replace(" ", "-")


def rel_link(target: Path, out_dir: Path) -> str:
    return Path(os.path.relpath(target, out_dir)).as_posix()


def _load_json(rel: str):
    path = REPO_ROOT / rel
    return json.loads(path.read_text()) if path.exists() else None


def md_inline(text: str) -> str:
    """Markdown for one paragraph of authored prose, without the <p> wrapper."""
    out = markdown.markdown(text)
    return re.sub(r"^<p>(.*)</p>$", r"\1", out, flags=re.S)


@dataclass
class BuildReport:
    docs: int = 0
    broken_links: list[str] = field(default_factory=list)
    broken_anchors: list[str] = field(default_factory=list)
    missing_outputs: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Facts: every number the authored content quotes
# ---------------------------------------------------------------------------
def _f(x, digits=3) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def collect_facts() -> dict[str, str]:
    """Read every quoted number from the code or a committed artifact."""
    from collections import Counter

    import pandas as pd

    from app.intelligence.expert_system.rule_base import build_rule_base
    from app.intelligence.knowledge_representation.schema import ENTITIES, RELATIONSHIPS, categories
    from app.intelligence.search.problem import RaceProblem
    from app.intelligence.xai import shap_analysis, trust

    facts: dict[str, str] = {}

    facts["api_routers"] = str(len(re.findall(r"app\.include_router\(", (REPO_ROOT / "app/api/main.py").read_text())))
    facts["kr_entities"] = str(len(ENTITIES))
    facts["kr_relationships"] = str(len(RELATIONSHIPS))
    facts["kr_categories"] = str(len(categories()))

    rules = build_rule_base()
    facts["es_rules"] = str(len(rules))
    facts["es_categories"] = str(len(Counter(r.category for r in rules)))

    facts["search_pit_loss"] = _f(RaceProblem.__dataclass_fields__["pit_loss"].default, 0)
    comp = _load_json("artifacts/search/reports/comparison.json")
    by_alg = {a["algorithm"]: a for a in comp["algorithms"]}
    facts["search_astar_cost"] = _f(by_alg["A*"]["solution_cost"], 2)
    facts["search_astar_nodes"] = str(by_alg["A*"]["nodes_expanded"])
    facts["search_ucs_nodes"] = str(by_alg["UCS"]["nodes_expanded"])

    src = _load_json("data/processed/data_source.json")
    facts.update(data_n_laps=f"{src['n_laps']:,}", data_n_drivers=str(src["n_drivers"]),
                 data_year=str(src["year"]), data_event=src["event"])
    features = pd.read_csv(REPO_ROOT / "data/processed/f1_features_selected.csv")
    cap = float(features["target_laptime"].max())
    facts["data_cap_value"] = _f(cap, 3)
    facts["data_capped_laps"] = str(int((features["target_laptime"] == cap).sum()))
    facts["data_pit_events"] = str(int(features["target_pit_next_lap"].sum()))
    facts["data_lap_mean"] = _f(float(features["target_laptime"].mean()), 2)
    facts["data_lap_sd"] = _f(float(features["target_laptime"].std()), 2)

    qml = _load_json("artifacts/metrics/qml_metrics.json")
    enc = qml["targets"]["target_pit_next_lap"]["encoding"]
    facts["qml_simulator"] = qml["simulator"].split(" ")[0]
    facts["qml_qubits"] = str(enc["n_qubits"])
    facts["qml_variance"] = _f(enc["explained_variance_ratio"] * 100, 1)

    meta = _load_json("data/processed/feature_metadata.json")
    facts["feat_rows"] = str(meta["rows"])
    facts["feat_reg"] = str(len(meta["selected_features"]["target_laptime"]))
    facts["feat_clf"] = str(len(meta["selected_features"]["target_pit_next_lap"]))
    funnel = meta["selection_funnel"]
    dropped = [f"`{c}` ({stage.split('_', 1)[1].replace('_dropped', '').replace('_', ' ')})"
               for stage, cols in funnel.items() if isinstance(cols, list) for c in cols]
    facts["feat_funnel"] = ", ".join(dropped) or "nothing"

    mc = _load_json("artifacts/metrics/model_comparison.json")
    reg = next(r for r in mc["regression"] if r["selected"])
    clf = next(r for r in mc["classification"] if r["selected"])
    facts["ml_n_models"] = str(len(mc["regression"]) + len(mc["classification"]))
    facts.update(ml_reg_best=reg["model"], ml_reg_mae=_f(reg["test_mae"]), ml_reg_r2=_f(reg["test_r2"]),
                 ml_clf_best=clf["model"], ml_clf_prauc=_f(clf["test_pr_auc"]),
                 ml_clf_threshold=_f(clf["decision_threshold"], 4))

    dl = _load_json("artifacts/deep_learning/evaluation_report.json")["models"]
    dr, dc = dl["target_laptime"], dl["target_pit_next_lap"]
    facts.update(
        dl_reg_mae=_f(dr["test_metrics"]["mae"]), dl_reg_r2=_f(dr["test_metrics"]["r2"]),
        dl_reg_params=f"{dr['architecture']['total_parameters']:,}",
        dl_clf_params=f"{dc['architecture']['total_parameters']:,}",
        dl_clf_roc=_f(dc["test_metrics"]["roc_auc"]), dl_clf_threshold=_f(dc["threshold"]["threshold"], 4),
        dl_reg_verdict=dr["overfitting"]["verdict"].split(" (")[0],
        dl_clf_verdict=dc["overfitting"]["verdict"].split(" (")[0],
        dl_test_n=str(dr["test_metrics"]["n"]), dl_test_pos=str(dc["test_metrics"]["n_positive"]),
    )

    xai = _load_json("artifacts/xai/xai_metadata.json")["targets"]
    facts["xai_trust_reg"] = _f(xai["target_laptime"]["trust_all_laps_summary"]["mean"], 2)
    facts["xai_trust_clf"] = _f(xai["target_pit_next_lap"]["trust_all_laps_summary"]["mean"], 2)
    facts["xai_bg_k"] = str(shap_analysis.BACKGROUND_K)
    facts["xai_nsamples"] = str(shap_analysis.KERNEL_NSAMPLES)
    facts["xai_weights"] = ", ".join(f"{k.replace('_', ' ')} {v:.2f}" for k, v in trust.WEIGHTS.items())
    lime = (REPO_ROOT / "artifacts/xai/LIME_Report.md").read_text()
    tops = re.findall(r"\*\*LIME top-3:\*\* `([^`]+)`", lime)
    facts["xai_lime_n"] = str(len(tops))
    facts["xai_lime_ts"] = str(sum(t == "track_status" for t in tops))

    tests = _load_json("artifacts/evaluation/test_results.json")
    perf = _load_json("artifacts/evaluation/performance.json")["api_latency"]["explain_false"]
    dom = _load_json("artifacts/evaluation/domain_validation.json")
    faith = _load_json("artifacts/evaluation/explainability_validation.json")["faithfulness_check"]["target_laptime"]
    reach = dom["rule_base_reachability_check"]
    facts.update(
        eval_date=tests["generated_at"][:10], eval_passed=str(tests["passed"]), eval_total=str(tests["total"]),
        eval_coverage=_f(tests["coverage_percent"], 1), eval_p50=_f(perf["p50_ms"], 0),
        eval_p95=_f(perf["p95_ms"], 0), eval_n_requests=str(perf["n_requests"]),
        eval_dom_laps=str(dom["n_laps_replayed"]), eval_dom_precision=_f(dom["precision"]),
        eval_dom_recall=_f(dom["recall"]),
        es_unreachable=str(reach["n_rules_structurally_unreachable_via_live_pipeline"]),
        es_populated_keys=str(len(reach["populated_input_keys"])),
        xai_faith_top=faith["top_feature"],
        xai_faith_top_delta=_f(faith["mean_abs_delta_ablating_top_feature"]),
        xai_faith_rand_delta=_f(faith["mean_abs_delta_ablating_random_feature"]),
    )
    return facts


_PLACEHOLDER = re.compile(r"\[\[(\w+)\]\]")


def fill(text: str, facts: dict[str, str]) -> str:
    def sub(m):
        if m.group(1) not in facts:
            raise KeyError(f"Unknown placeholder [[{m.group(1)}]] in study-guide content")
        return facts[m.group(1)]
    return _PLACEHOLDER.sub(sub, text)


# ---------------------------------------------------------------------------
# Source extraction
# ---------------------------------------------------------------------------
@dataclass
class SourceRef:
    path: str
    qualname: str
    code: str
    start: int   # 1-based line numbers of the shown code
    end: int
    def_line: int


def extract_source(path: str, qualname: str, markers: tuple[str, str] | None = None) -> SourceRef:
    """The source of ``qualname`` (``func``, ``Class`` or ``Class.method``) in
    ``path``. ``markers`` narrows it to the lines from the first one containing
    ``markers[0]`` up to (not including) the next one containing ``markers[1]``."""
    file = REPO_ROOT / path
    text = file.read_text()
    lines = text.splitlines()
    nodes = ast.parse(text).body
    node = None
    for part in qualname.split("."):
        node = next((n for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                     and n.name == part), None)
        if node is None:
            raise LookupError(f"{qualname} not found in {path}")
        nodes = node.body
    start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
    end = node.end_lineno
    if markers:
        body = range(start, end + 1)
        try:
            s = next(i for i in body if markers[0] in lines[i - 1])
            e = next(i for i in range(s + 1, end + 1) if markers[1] in lines[i - 1]) - 1
        except StopIteration:
            raise LookupError(f"markers {markers!r} not found in {qualname} ({path})") from None
        start, end = s, e
        while end > start and not lines[end - 1].strip():
            end -= 1
    code = textwrap.dedent("\n".join(lines[start - 1:end]))
    return SourceRef(path, qualname, code, start, end, node.lineno)


_COPY_BUTTON = '<button class="copy" type="button" aria-label="Copy code">Copy</button>'


def _wrap_code(highlighted: str) -> str:
    """Pygments and codehilite both emit ``<div class="hl"><pre>``; give every
    code block the same shape: ``div.code > button.copy + pre.hl``."""
    return highlighted.replace('<div class="hl"><pre>', f'<div class="code">{_COPY_BUTTON}<pre class="hl">')


def code_block(code: str, lang: str = "python", start: int | None = None) -> str:
    """Highlighted code with an optional line-number gutter and a copy button."""
    if highlight is None:  # pragma: no cover
        return f'<div class="code">{_COPY_BUTTON}<pre class="hl">{esc(code)}</pre></div>'
    try:
        lexer = get_lexer_by_name(lang)
    except Exception:
        lexer = get_lexer_by_name("text")
    formatter = HtmlFormatter(cssclass="hl", linenos="inline" if start else False, linenostart=start or 1)
    return _wrap_code(highlight(code, lexer, formatter))


# ---------------------------------------------------------------------------
# Reports: every artifacts/**/*.md, discovered at build time
# ---------------------------------------------------------------------------
@dataclass
class Doc:
    path: Path
    rel: str
    slug: str
    group: str
    title: str = ""
    body: str = ""
    toc: list = field(default_factory=list)
    summary: str = ""
    generated: str = ""


def doc_slug(rel: str) -> str:
    s = re.sub(r"^artifacts/", "r-", rel).replace("/reports/", "/")
    s = re.sub(r"\.md$", "", s)
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def report_category(rel: str) -> str:
    return rel.split("/")[1]


def discover_reports() -> list[Doc]:
    """Every Markdown report under artifacts/, ordered by category (study order)
    then file name."""
    order = list(REPORT_CATEGORIES)
    found = [p.relative_to(REPO_ROOT).as_posix() for p in (REPO_ROOT / "artifacts").rglob("*.md")]
    found.sort(key=lambda r: (order.index(report_category(r)) if report_category(r) in order else len(order),
                              report_category(r), r))
    return [Doc(REPO_ROOT / r, r, doc_slug(r), report_category(r)) for r in found]


def _markdown_doc_body(d: Doc) -> str:
    slugger = (lambda prefix: lambda value, sep: f"{prefix}--{gh_slug(value)}")(d.slug)
    md = markdown.Markdown(
        extensions=["tables", "fenced_code", "codehilite", "toc", "sane_lists"],
        extension_configs={"codehilite": {"css_class": "hl", "guess_lang": False},
                           "toc": {"slugify": slugger}},
    )
    body = md.convert(d.path.read_text(encoding="utf-8"))
    return body.replace("<table>", '<div class="table-wrap"><table>').replace("</table>", "</table></div>")


def _summary_and_date(text: str) -> tuple[str, str]:
    """The report's first prose paragraph (for its card) and its generated date."""
    date = re.search(r"Generated[^0-9]{0,12}(\d{4}-\d{2}-\d{2})", text)
    para, lines = [], text.splitlines()
    for line in lines:
        t = line.strip()
        if not t:
            if para:
                break
            continue
        if re.match(r"^(#|\||```|_Generated|\*Generated|!\[|---|>|- |\* |\d+\.)", t) or t.startswith("<"):
            if para:
                break
            continue
        para.append(t)
    summary = re.sub(r"[`*_]|\[([^\]]*)\]\([^)]*\)", lambda m: m.group(1) or "", " ".join(para))
    if not summary:
        # Table-only reports have no prose; describe what is there instead.
        rows = [ln for ln in lines if ln.startswith("|") and not re.match(r"^\|[\s:|-]+\|$", ln.strip())]
        heads = [re.sub(r"^#+\s*", "", ln) for ln in lines if re.match(r"^##\s", ln)]
        if rows:
            summary = f"A table of {max(len(rows) - 1, 0)} rows."
        elif heads:
            summary = "Sections: " + ", ".join(heads[:4]) + ("…" if len(heads) > 4 else "")
    if len(summary) > 180:
        summary = summary[:180].rsplit(" ", 1)[0] + "…"
    return summary, (date.group(1) if date else "")


_ATTR = re.compile(r'(<(a|img)\b[^>]*?\s)(href|src)="([^"]*)"')


def _rewrite_links(doc: Doc, body: str, by_rel: dict[str, Doc], out_dir: Path, report: BuildReport) -> str:
    def sub(m):
        prefix, tag, attr, url = m.group(1), m.group(2), m.group(3), html.unescape(m.group(4))
        if re.match(r"^[a-z][a-z0-9+.-]*:", url):            # http:, https:, mailto:
            extra = ' target="_blank" rel="noopener"' if tag == "a" else ""
            return f'{prefix}{attr}="{esc(url)}"{extra}'
        path, _, frag = url.partition("#")
        if not path:                                          # same-document anchor
            return f'{prefix}{attr}="#{doc.slug}--{gh_slug(frag)}"'
        target = (doc.path.parent / path).resolve()
        try:
            target_rel = target.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            target_rel = None
        if target_rel in by_rel:
            dest = by_rel[target_rel]
            anchor = f"{dest.slug}--{gh_slug(frag)}" if frag else dest.slug
            return f'{prefix}{attr}="#{anchor}"'
        if target.exists():
            href = rel_link(target, out_dir) + (f"#{frag}" if frag else "")
            extra = ' loading="lazy"' if tag == "img" else ""
            return f'{prefix}{attr}="{esc(href)}"{extra}'
        report.broken_links.append(f"{doc.rel}: {url}")
        return (f'{prefix}data-broken="{esc(url)}" '
                f'title="Link target not found in the repository: {esc(url)}"')
    return _ATTR.sub(sub, body)


def render_reports(out_dir: Path, report: BuildReport) -> list[Doc]:
    docs = discover_reports()
    by_rel = {d.rel: d for d in docs}
    for d in docs:
        d.summary, d.generated = _summary_and_date(d.path.read_text(encoding="utf-8"))
        d.body = _rewrite_links(d, _wrap_code(_markdown_doc_body(d)), by_rel, out_dir, report)
        heads = [(int(lvl), hid, html.unescape(re.sub(r"<[^>]+>", "", inner)).strip())
                 for lvl, hid, inner in re.findall(r'<h([1-3]) id="([^"]+)">(.*?)</h\1>', d.body, flags=re.S)]
        h1 = next((h for h in heads if h[0] == 1), None)
        d.title = h1[2] if h1 else d.path.stem.replace("_", " ")
        d.toc = [h for h in heads if h[0] in (2, 3)]
    report.docs = len(docs)
    return docs


def _category_meta(key: str) -> tuple[str, int | None, str, str]:
    return REPORT_CATEGORIES.get(key, (key.replace("_", " ").capitalize(), None, "report",
                                      f"Reports found in artifacts/{key}/."))


def render_reports_tab(docs: list[Doc], out_dir: Path) -> str:
    groups: dict[str, list[Doc]] = {}
    for d in docs:
        groups.setdefault(d.group, []).append(d)

    nav, sections = [], []
    for key, items in groups.items():
        label, practical, icon, desc = _category_meta(key)
        links = "".join(f'<li><a href="#{d.slug}" data-doc-link="{d.slug}">{esc(d.title)}</a></li>' for d in items)
        nav.append(f'<details class="nav-group" open><summary>{svg(icon, 16)} {esc(label)} '
                   f'<span class="count">{len(items)}</span></summary><ul>{links}</ul></details>')
        badge = f'<a class="pbadge" href="#p-{practical}">Practical {practical}</a>' if practical else ""
        cards = "".join(
            f'<a class="rcard" href="#{d.slug}"><strong>{esc(d.title)}</strong>'
            f'<span>{esc(d.summary) or "&nbsp;"}</span>'
            f'<em>{svg("calendar", 13)} {esc(d.generated or "date not recorded")}</em></a>' for d in items)
        sections.append(
            f'<section class="rcat" id="cat-{key}"><header><span class="rcat-icon">{svg(icon, 24)}</span>'
            f'<div><h2>{esc(label)} {badge}</h2><p>{esc(desc)}</p></div>'
            f'<span class="count">{len(items)}</span></header><div class="rcards">{cards}</div></section>')

    articles = []
    for d in docs:
        label, practical, icon, _ = _category_meta(d.group)
        same = groups[d.group]
        i = same.index(d)
        prev_d, next_d = (same[i - 1] if i else None), (same[i + 1] if i + 1 < len(same) else None)
        toc = "".join(f'<li class="lvl{lvl}"><a href="#{hid}">{esc(name)}</a></li>' for lvl, hid, name in d.toc)
        toc_html = (f'<nav class="doc-toc" aria-label="On this page"><p class="toc-title">On this page</p>'
                    f'<ul>{toc}</ul></nav>') if toc else ""
        pager = '<nav class="pager" aria-label="Previous and next report">'
        pager += (f'<a class="prev" href="#{prev_d.slug}"><span>Previous</span>{esc(prev_d.title)}</a>'
                  if prev_d else "<span></span>")
        pager += (f'<a class="next" href="#{next_d.slug}"><span>Next</span>{esc(next_d.title)}</a>'
                  if next_d else "<span></span>")
        pager += "</nav>"
        src = rel_link(d.path, out_dir)
        related = f' · <a href="#p-{practical}">Practical {practical} code walkthrough</a>' if practical else ""
        articles.append(
            f'<article class="doc" id="{d.slug}" data-doc="{d.slug}" data-view data-crumb="{esc(d.title)}" '
            f'data-parent="reports-home" data-crumb-mid="{esc(label)}" hidden>'
            f'<header class="doc-head"><p class="crumbs">{svg(icon, 14)} {esc(label)}</p>'
            f'<p class="src">Source: <a href="{esc(src)}"><code>{esc(d.rel)}</code></a>'
            f'{" · generated " + esc(d.generated) if d.generated else ""}{related}</p></header>'
            f'<div class="doc-layout"><div class="prose">{d.body}{pager}</div>{toc_html}</div></article>')

    return f"""
<section class="tab-panel reports-tab" id="tab-reports" data-tab="reports" role="tabpanel" aria-labelledby="tabbtn-reports" hidden>
  <aside class="sidebar" id="reports-sidebar" aria-label="Reports">
    <div class="search">
      <label class="visually-hidden" for="doc-search">Search the reports</label>
      <input id="doc-search" type="search" placeholder="Search {len(docs)} reports…" autocomplete="off">
      <p class="search-status" id="search-status" aria-live="polite"></p>
    </div>
    <nav class="doc-nav">{''.join(nav)}</nav>
  </aside>
  <div class="guide-main">
    <article class="doc doc-home" id="reports-home" data-doc="reports-home" data-view data-crumb="Reports" data-parent="home">
      <header class="practical-head"><p class="crumbs">Reports</p>
        <h1>Generated reports</h1>
        <p class="lead">{len(docs)} reports written by the pipelines into <code>artifacts/</code>, found when this
        page was built and shown exactly as they were generated.</p></header>
      {''.join(sections)}
    </article>
    {''.join(articles)}
  </div>
</section>"""


# ---------------------------------------------------------------------------
# Study Guide: the hub
# ---------------------------------------------------------------------------
def _summary_figure(out_dir: Path) -> str:
    """The one-page 10-task summary sheet (scripts/build_task_summary.py), if built."""
    png = REPO_ROOT / "docs" / "task_summary.png"
    if not png.exists():
        return ""
    src = esc(rel_link(png, out_dir))
    return (f'<figure class="figure summary-sheet"><a href="{src}"><img src="{src}" loading="lazy" '
            f'alt="Ten-task summary sheet: problem, inputs, steps, output and deliverables for each task"></a>'
            f'<figcaption>All ten tasks on one page — select to open full size.</figcaption></figure>')


def render_guide_tab(facts: dict, n_reports: int, out_dir: Path) -> str:
    tiles = [(facts["data_n_laps"], "real laps", f"{facts['data_year']} {facts['data_event']} GP"),
             (facts["data_n_drivers"], "drivers", "FastF1 timing data"),
             ("2", "predictions", "lap time · pit next lap"),
             ("10", "practicals", "one application")]
    tiles_html = "".join(f'<div class="tile"><strong>{esc(v)}</strong><span>{esc(k)}</span><em>{esc(d)}</em></div>'
                         for v, k, d in tiles)
    path = [("#diagrams", "layers", "Diagrams & Workflow", "See how the system fits together, in seven diagrams."),
            ("#p-home", "brain", "Understand the Code", "The ten practicals, explained from the source."),
            ("#reports-home", "report", "Reports", f"{n_reports} generated reports, by category."),
            ("#slide-1", "monitor", "Showcase", "Present the project — full-screen slide deck.")]
    path_html = "".join(f'<a class="step-card" href="{h}"><span class="n">{i}</span>{svg(ic, 24)}'
                        f'<strong>{esc(t)}</strong><span>{esc(d)}</span></a>'
                        for i, (h, ic, t, d) in enumerate(path, 1))
    return f"""
<section class="tab-panel guide" id="tab-guide" data-tab="guide" role="tabpanel" aria-labelledby="tabbtn-guide">
  <div class="hub">
    <article class="hub-page" id="home" data-view data-crumb="Study Guide">
      <p class="crumbs">Study Guide</p>
      <h1>F1 Race Strategy Intelligence</h1>
      <p class="lead">When should a Formula 1 driver pit, and what will the next lap cost? One real race, attacked
      with symbolic reasoning, machine learning and explainability — built as one application.</p>
      <div class="tiles">{tiles_html}</div>
      <h2>Study path</h2>
      <div class="step-cards">{path_html}</div>
      <h2>The ten tasks on one page</h2>
      {_summary_figure(out_dir)}
    </article>
  </div>
</section>"""


# ---------------------------------------------------------------------------
# Understand the Code
# ---------------------------------------------------------------------------
def _table(headers: list[str], rows: list[list[str]], cls: str = "") -> str:
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="table-wrap"><table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _example(spec: tuple, facts: dict, out_dir: Path, live: bool) -> str:
    kind = spec[0]
    if kind == "file":
        _, path, n = spec
        lines = (REPO_ROOT / path).read_text().splitlines()
        shown = "\n".join(lines[:n]) + ("\n…" if len(lines) > n else "")
        slug = doc_slug(path)
        return (f'<p>The first {min(n, len(lines))} lines of <code>{esc(path)}</code>, as the pipeline last wrote it '
                f'(<a href="#{slug}">read it in the Reports tab</a>):</p>' + code_block(shown, "markdown"))
    if kind == "json":
        _, path, key = spec
        data = _load_json(path)[key]
        return (f'<p><code>{esc(key)}</code> from <code>{esc(path)}</code>:</p>'
                + code_block(json.dumps(data, indent=2, ensure_ascii=False), "json"))
    if kind == "search_table":
        comp = _load_json("artifacts/search/reports/comparison.json")
        p = comp["problem"]
        rows = [[f"<strong>{esc(a['algorithm'])}</strong>", _f(a["solution_cost"], 2),
                 f"{a['cost_gap_pct']:.2f}%", "yes" if a["is_optimal"] else "no",
                 f"{a['nodes_expanded']:,}", str(a["max_frontier_size"]), _f(a["elapsed_ms"], 1),
                 str(a["n_pit_stops"])] for a in comp["algorithms"]]
        return (f'<p>From <code>artifacts/search/reports/comparison.json</code> — a {p["total_laps"]}-lap race, '
                f'start on {p["start_compound"]}, pit loss {p["pit_loss_seconds"]:g} s, track '
                f'{p["track_temperature_c"]:g} °C, at most {p["max_stops"]} stops:</p>'
                + _table(["Algorithm", "Race time (s)", "Gap to optimal", "Optimal", "Nodes expanded",
                          "Max frontier", "Time (ms)", "Stops"], rows, "num"))
    if kind == "ml_table":
        mc = _load_json("artifacts/metrics/model_comparison.json")
        reg = [[f"<code>{esc(r['model'])}</code>" + (" ★" if r["selected"] else ""), _f(r["cv_mae"]),
                _f(r["test_mae"]), _f(r["test_r2"])] for r in mc["regression"]]
        clf = [[f"<code>{esc(r['model'])}</code>" + (" ★" if r["selected"] else ""), _f(r["cv_pr_auc"]),
                _f(r["test_roc_auc"]), _f(r["test_pr_auc"]), _f(r["decision_threshold"], 4)]
               for r in mc["classification"]]
        return ('<p>From <code>artifacts/metrics/model_comparison.json</code>, in rank order (★ = selected):</p>'
                + _table(["Lap-time model", "CV MAE (s)", "Test MAE (s)", "Test R²"], reg, "num")
                + _table(["Pit model", "CV PR-AUC", "Test ROC-AUC", "Test PR-AUC", "Threshold"], clf, "num"))
    if kind == "dl_table":
        dl = _load_json("artifacts/deep_learning/evaluation_report.json")["models"]
        r, c = dl["target_laptime"], dl["target_pit_next_lap"]
        rows = [[s.title(), _f(r[f"{s}_metrics"]["mae"]), _f(r[f"{s}_metrics"]["r2"]),
                 _f(c[f"{s}_metrics"].get("roc_auc")), _f(c[f"{s}_metrics"].get("pr_auc"))]
                for s in ("train", "validation", "test")]
        return ('<p>From <code>artifacts/deep_learning/evaluation_report.json</code>:</p>'
                + _table(["Split", "Lap time MAE (s)", "Lap time R²", "Pit ROC-AUC", "Pit PR-AUC"], rows, "num"))
    if kind == "trust_table":
        xai = _load_json("artifacts/xai/xai_metadata.json")["targets"]
        rows = []
        for t, label in (("target_laptime", "Lap time"), ("target_pit_next_lap", "Pit decision")):
            s = xai[t]["trust_all_laps_summary"]
            rows.append([label, str(s["n"]), _f(s["mean"])] + [str(s["bands"].get(b, 0))
                                                              for b in ("HIGH", "MODERATE", "LOW", "DO NOT ACT")])
        return ('<p>Trust bands over every test lap, from <code>artifacts/xai/xai_metadata.json</code>:</p>'
                + _table(["Target", "Laps", "Mean trust", "High", "Moderate", "Low", "Do not act"], rows, "num"))
    if kind == "eval_table":
        t = _load_json("artifacts/evaluation/test_results.json")
        dom = _load_json("artifacts/evaluation/domain_validation.json")
        rob = _load_json("artifacts/evaluation/responsible_ai.json")["robustness"]
        conf = dom["confusion"]
        rows = [["Tests", f"{t['passed']} passed, {t['failed']} failed, {t['skipped']} skipped",
                 "<code>test_results.json</code>"],
                ["Line coverage", f"{t['coverage_percent']:.1f}%", "<code>test_results.json</code>"],
                ["Backtest pit calls", f"{conf['hits']} hits, {conf['misses']} misses, {conf['false_calls']} false calls",
                 "<code>domain_validation.json</code>"],
                ["Robustness (flip rate)", ", ".join(f"{k.replace('_', ' ')}: {v:.0%}"
                                                     for k, v in rob["flip_rates"].items()),
                 "<code>responsible_ai.json</code>"]]
        return ("<p>Headline results of the last evaluation run, from <code>artifacts/evaluation/</code>:</p>"
                + _table(["Check", "Result", "Source"], rows))
    if kind == "strategy_example":
        return _strategy_example(live)
    raise ValueError(f"unknown example kind {kind!r}")


# The race state scripts/demo_predict.py sends — the project's own demo input.
EXAMPLE_RACE_STATE = {"driver": "ALO", "team": "ASTON MARTIN", "current_lap": 20, "total_laps": 55,
                      "tyre_compound": "MEDIUM", "tyre_age": 15, "track_temperature": 40.0, "weather": "dry",
                      "fuel_kg": 70, "track_status": "GREEN", "current_position": 6}


def _strategy_example(live: bool) -> str:
    request = code_block(json.dumps(EXAMPLE_RACE_STATE, indent=2), "json")
    intro = ('<p>Request — the race state <code>scripts/demo_predict.py</code> sends to '
             '<code>POST /api/strategy/predict</code>:</p>' + request)
    if not live:
        return intro + '<p class="muted">Response not included: this build ran with <code>--no-live</code>.</p>'
    try:
        from app.api.schemas import RaceStateRequest
        from app.services.strategy_service import run_strategy_analysis
        r = run_strategy_analysis(RaceStateRequest(**EXAMPLE_RACE_STATE))
    except Exception as exc:  # the guide must still build without trained models
        return intro + f'<p class="muted">Response not available at build time: {esc(type(exc).__name__)}: {esc(exc)}</p>'
    plan = r["search_plan"]
    excerpt = {
        "recommendation": r["recommendation"],
        "prediction": {k: r["prediction"][k] for k in ("predicted_lap_time_seconds", "laptime_model",
                                                       "probability_pit", "pit_model")},
        "dl_prediction": {k: r["dl_prediction"][k] for k in ("predicted_lap_time_seconds", "probability_pit",
                                                             "predicted_class", "threshold")},
        "triggered_expert_rules": [{"rule_id": t["rule_id"], "name": t["name"]} for t in r["triggered_expert_rules"]],
        "search_plan": {"algorithm": plan["algorithm"], "expected_cost_seconds": plan["expected_cost_seconds"],
                        "next_action": plan["next_action"], "plan_length_laps": len(plan["plan"])},
        "out_of_range": r["feature_construction"]["out_of_range"],
    }
    return (intro + '<p>Response excerpt — produced by running <code>run_strategy_analysis</code> on that state '
            'when this guide was built (feature rows and per-feature detail omitted):</p>'
            + code_block(json.dumps(excerpt, indent=2, default=str, ensure_ascii=False), "json"))


def render_practical(p: dict, facts: dict, out_dir: Path, report: BuildReport, live: bool) -> str:
    n = p["number"]
    F = lambda s: fill(s, facts)  # noqa: E731
    sid = f"p-{n}"
    what = "".join(f"<p>{md_inline(F(t))}</p>" for t in p["what"])
    steps = ""
    # A title like "3–5 · ML, rules, DL" names pipeline stages; that label
    # replaces the running number. In a list that uses stage labels, a step
    # without one is not a stage, so it gets a bullet rather than a number
    # that could repeat a stage's.
    staged = any(re.match(r"^[\d–-]+ · ", t) for t, _ in p["steps"])
    for k, (t, d) in enumerate(p["steps"], 1):
        m = re.match(r"^([\d–-]+) · (.*)$", t)
        label, t = (m.group(1), m.group(2)) if m else ("•" if staged else str(k), t)
        steps += f'<li data-n="{esc(label)}"><h4>{esc(F(t))}</h4><p>{md_inline(F(d))}</p></li>'
    tech = _table(["Technology", "Why and how it is used"],
                  [[f"<strong>{esc(name)}</strong>", md_inline(F(why))] for name, why in p["tech"]])

    func_rows = []
    for path, qual, role in p["functions"]:
        ref = extract_source(path, qual)
        href = rel_link(REPO_ROOT / path, out_dir)
        func_rows.append([f"<code>{esc(qual)}</code>",
                          f'<a href="{esc(href)}"><code>{esc(path)}:{ref.def_line}</code></a>', md_inline(F(role))])
    funcs = _table(["Function / class", "Location", "Role"], func_rows)

    snippets = []
    for spec in p["snippets"]:
        path, qual, caption = spec[:3]
        ref = extract_source(path, qual, spec[3] if len(spec) > 3 else None)
        snippets.append(
            f'<figure class="snippet"><figcaption><span>{md_inline(F(caption))}</span>'
            f'<code>{esc(path)} · lines {ref.start}–{ref.end}</code></figcaption>'
            f'{code_block(ref.code, "python", ref.start)}</figure>')

    out_rows = []
    for path, what_it_is in p["outputs"]:
        exists = (REPO_ROOT / path).exists()
        if not exists:
            report.missing_outputs.append(f"Practical {n}: {path}")
        label = (f'<a href="{esc(rel_link(REPO_ROOT / path, out_dir))}"><code>{esc(path)}</code></a>'
                 if exists else f"<code>{esc(path)}</code>")
        status = '<span class="badge ok">present</span>' if exists else '<span class="badge missing">not generated</span>'
        out_rows.append([label, esc(what_it_is), status])
    outputs = _table(["Output", "Contents", "Status"], out_rows)

    figure = ""
    if p.get("figure"):
        fpath, fcap = p["figure"]
        if (REPO_ROOT / fpath).exists():
            figure = (f'<figure class="figure"><img src="{esc(rel_link(REPO_ROOT / fpath, out_dir))}" '
                      f'alt="{esc(F(fcap))}" loading="lazy"><figcaption>{esc(F(fcap))} '
                      f'<code>{esc(fpath)}</code></figcaption></figure>')
        else:
            report.missing_outputs.append(f"Practical {n} figure: {fpath}")
    example = _example(p["example"], facts, out_dir, live) if p.get("example") else ""
    apps = "".join(f"<li>{md_inline(F(a))}</li>" for a in p["applications"])
    caveat = (f'<aside class="callout"><strong>Limitation to know</strong><p>{md_inline(F(p["caveat"]))}</p></aside>'
              if p.get("caveat") else "")
    viva = "".join(f'<details class="qa"><summary>{md_inline(F(q))}</summary><p>{md_inline(F(a))}</p></details>'
                   for q, a in p.get("viva", []))
    run = code_block(p["run"], "bash")

    sections = [("what", "What and why"), ("how", "How it works, step by step"), ("tech", "Technologies"),
                ("funcs", "Key functions"), ("code", "Code walkthrough"), ("out", "Outputs and examples"),
                ("apps", "Applications")] + ([("viva", "Key points & viva")] if viva else [])
    subnav = "".join(f'<a href="#{sid}-{k}">{label}</a>' for k, label in sections)
    prev_link = (f'<a class="prev" href="#p-{n - 1}"><span>Previous</span>Practical {n - 1}</a>'
                 if n > 1 else "<span></span>")
    next_link = (f'<a class="next" href="#p-{n + 1}"><span>Next</span>Practical {n + 1}</a>'
                 if n < len(PRACTICALS) else "<span></span>")
    return f"""
<article class="practical" id="{sid}" data-practical="{n}" data-view data-crumb="Practical {n} · {esc(p['title'])}" data-parent="p-home" hidden>
  <header class="practical-head">
    <p class="crumbs">Practical {n} of {len(PRACTICALS)} · <code>{esc(p['package'])}</code></p>
    <h1>{esc(p['title'])}</h1>
    <p class="lead">{md_inline(F(p['tagline']))}</p>
    <nav class="subnav" aria-label="Sections of Practical {n}">{subnav}</nav>
  </header>
  <section id="{sid}-what"><h2>What and why</h2>{what}<p class="run-label">Run it</p>{run}</section>
  <section id="{sid}-how"><h2>How it works, step by step</h2><ol class="steps">{steps}</ol></section>
  <section id="{sid}-tech"><h2>Technologies</h2>{tech}</section>
  <section id="{sid}-funcs"><h2>Key functions</h2>{funcs}</section>
  <section id="{sid}-code"><h2>Code walkthrough</h2>
    <p class="muted">Excerpts are read from the source when this guide is built; the line numbers are the real ones.</p>
    {''.join(snippets)}</section>
  <section id="{sid}-out"><h2>Outputs and examples</h2>{outputs}{figure}{example}{caveat}</section>
  <section id="{sid}-apps"><h2>Applications</h2><ul>{apps}</ul></section>
  {f'<section id="{sid}-viva"><h2>Key points &amp; viva</h2>{viva}</section>' if viva else ""}
  <nav class="pager" aria-label="Previous and next practical">{prev_link}{next_link}</nav>
</article>"""


def render_code_tab(facts: dict, out_dir: Path, report: BuildReport, live: bool) -> str:
    rail = "".join(
        f'<li><a href="#p-{p["number"]}" data-practical-link="{p["number"]}"><span class="num">{p["number"]:02d}</span>'
        f'{esc(p["title"])}</a></li>' for p in PRACTICALS)
    cards = "".join(
        f'<a class="pcard" href="#p-{p["number"]}"><span class="num">{p["number"]:02d}</span>'
        f'<strong>{esc(p["title"])}</strong><span>{md_inline(fill(p["tagline"], facts))}</span></a>'
        for p in PRACTICALS)
    body = "".join(render_practical(p, facts, out_dir, report, live) for p in PRACTICALS)
    return f"""
<section class="tab-panel code-tab" id="tab-code" data-tab="code" role="tabpanel" aria-labelledby="tabbtn-code" hidden>
  <aside class="sidebar rail" aria-label="Practicals"><p class="rail-title">The ten practicals</p><ol>{rail}</ol></aside>
  <div class="code-main">
    <article class="practical" id="p-home" data-practical="home" data-view data-crumb="Understand the Code" data-parent="home">
      <header class="practical-head"><p class="crumbs">Understand the Code</p>
        <h1>The ten practicals, from the source</h1>
        <p class="lead">Each practical is one package of a single application. Every page explains what was
        built and why, walks through how it works, and shows the real functions with their real line numbers.</p>
      </header>
      <div class="pcards">{cards}</div>
    </article>
    {body}
  </div>
</section>"""


# ---------------------------------------------------------------------------
# Showcase
# ---------------------------------------------------------------------------
def _requirement_versions() -> dict[str, str]:
    out = {}
    for line in (REPO_ROOT / "requirements.txt").read_text().splitlines():
        line = line.split("#")[0].strip()
        m = re.match(r"^([A-Za-z0-9_.\-\[\]]+)\s*([=<>!~]=?.*)?$", line)
        if m:
            out[re.sub(r"\[.*\]", "", m.group(1)).lower()] = (m.group(2) or "").strip()
    return out


def _frontend_versions() -> dict[str, str]:
    pkg = _load_json("frontend/package.json") or {}
    return {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}


def render_slide(i: int, s: dict, facts: dict) -> str:
    F = lambda t: fill(t, facts)  # noqa: E731
    kind = s["kind"]
    title = f'<h2 class="s-title">{md_inline(F(s["title"]))}</h2>' if kind != "title" else ""
    inner = ""
    if kind == "title":
        inner = (f'<p class="s-eyebrow">{esc(F(s["eyebrow"]))}</p><h1 class="s-hero">{esc(F(s["title"]))}</h1>'
                 f'<p class="s-lead">{md_inline(F(s["lead"]))}</p><p class="s-foot">{esc(F(s["foot"]))}</p>')
    elif kind == "bullets":
        lead = f'<p class="s-lead">{md_inline(F(s["lead"]))}</p>' if s.get("lead") else ""
        inner = lead + '<ul class="s-bullets">' + "".join(f"<li>{md_inline(F(b))}</li>" for b in s["bullets"]) + "</ul>"
        if s.get("note"):
            inner += f'<p class="s-note">{esc(F(s["note"]))}</p>'
    elif kind == "stats":
        inner = '<div class="s-stats">' + "".join(
            f'<div class="s-stat"><span class="v">{esc(F(v))}</span><span class="k">{esc(F(k))}</span>'
            f'<span class="d">{esc(F(d))}</span></div>' for v, k, d in s["stats"]) + "</div>"
    elif kind == "pipeline":
        layer = lambda label, items, cls="": (  # noqa: E731
            f'<div class="arch-row {cls}"><span class="arch-label">{label}</span><div class="arch-boxes">'
            + "".join(f'<div class="arch-box"><strong>{a}</strong><span>{b}</span></div>' for a, b in items)
            + "</div></div>")
        inner = ('<div class="arch">'
                 + layer("Presentation", [("Next.js dashboard", "frontend/ — every value fetched"),
                                          ("Strategy report", "Markdown / HTML download")])
                 + layer("API", [("FastAPI", f"app/api/ — {F('[[api_routers]]')} routers, Pydantic validation")])
                 + layer("Services", [("P9 Strategy service", "8-stage pipeline → one recommendation"),
                                      ("Model cache", "registry-driven loading"),
                                      ("Feature approximation", "race state → feature row")], "accent")
                 + layer("Intelligence", [("P1 Ontology", "KR"), ("P2 Rules", "expert system"),
                                          ("P3 Search", "A*"), ("P6 ML", "sklearn"), ("P7 DL", "Keras"),
                                          ("P8 XAI", "SHAP · LIME")])
                 + layer("Data", [("P4 Clean laps", "FastF1, audited"), ("P5 Feature contract", "CSV + JSON"),
                                  ("artifacts/", "models · metrics · reports"),
                                  ("P10 Evaluation", "evaluate_system.py → artifacts/evaluation/")])
                 + "</div>")
    elif kind == "grid":
        inner = '<div class="s-grid">' + "".join(
            f'<a class="s-cell" href="#p-{p["number"]}"><span class="num">{p["number"]:02d}</span>'
            f'<strong>{esc(p["title"])}</strong></a>' for p in PRACTICALS) + "</div>"
    elif kind == "tech":
        py, js = _requirement_versions(), _frontend_versions()
        groups = [
            ("Symbolic AI", [("owlready2", py), ("rdflib", py), ("networkx", py)]),
            ("Data & ML", [("pandas", py), ("scikit-learn", py), ("xgboost", py)]),
            ("Deep learning & XAI", [("keras", py), ("torch", py), ("shap", py), ("lime", py)]),
            ("Quantum (extension)", [("pennylane", py)]),
            ("Serving", [("fastapi", py), ("uvicorn", py), ("pydantic", py)]),
            ("Frontend", [("next", js), ("react", js), ("tailwindcss", js), ("typescript", js)]),
        ]
        inner = '<div class="s-tech">' + "".join(
            f'<div class="s-techgroup"><p>{esc(g)}</p><ul>' + "".join(
                f'<li><strong>{esc(name)}</strong><span>{esc(src.get(name, "") or "")}</span></li>'
                for name, src in items if name in src) + "</ul></div>" for g, items in groups) + "</div>"
        inner += '<p class="s-note">Versions read from requirements.txt and frontend/package.json at build time.</p>'
    elif kind == "workflow":
        stages = [("1", "Validate", "Pydantic + dataset options"), ("2", "Features", "race state → contract row"),
                  ("3", "ML", "Practical 6 pipelines"), ("4", "Rules", "forward chaining"),
                  ("5", "DL", "tuned threshold"), ("6", "Search", "A* over remaining laps"),
                  ("7", "Recommend", "rules › ML+DL agreement"), ("8", "Explain", "SHAP + trust (opt-in)")]
        inner = '<ol class="s-flow">' + "".join(
            f'<li><span class="n">{n}</span><strong>{a}</strong><span>{b}</span></li>' for n, a, b in stages) + "</ol>"
        inner += ('<p class="s-note">Output: <strong>STAY OUT</strong>, <strong>PIT NOW</strong> or '
                  '<strong>PIT IN N LAPS</strong> — with every stage\'s evidence, and any ML/DL disagreement flagged.</p>')
    elif kind == "results_laptime":
        mc = _load_json("artifacts/metrics/model_comparison.json")["regression"]
        dl = _load_json("artifacts/deep_learning/evaluation_report.json")["models"]["target_laptime"]
        rows = sorted(mc, key=lambda r: r["test_mae"])
        worst_mae = max([dl["test_metrics"]["mae"]] + [r["test_mae"] for r in rows])
        bar = lambda label, mae, r2, cls="": (  # noqa: E731
            f'<div class="bar {cls}"><span class="bl">{label}</span><span class="bt"><span class="bf" '
            f'style="width:{mae / worst_mae * 100:.1f}%"></span></span><span class="bv">{mae:.3f} s'
            f'<em>R² {r2:.2f}</em></span></div>')
        inner = ('<p class="s-lead">Mean absolute error on the test laps, in seconds — shorter is better.</p>'
                 '<div class="s-bars">' + bar("DNN (Practical 7)", dl["test_metrics"]["mae"], dl["test_metrics"]["r2"], "hi")
                 + "".join(bar(r["model"] + (" ★" if r["selected"] else ""), r["test_mae"], r["test_r2"]) for r in rows)
                 + '</div><p class="s-note">★ selected by Practical 6\'s rule. Source: model_comparison.json, '
                   'deep_learning/evaluation_report.json.</p>')
    elif kind == "results_pit":
        mc = _load_json("artifacts/metrics/model_comparison.json")["classification"]
        dl = _load_json("artifacts/deep_learning/evaluation_report.json")["models"]["target_pit_next_lap"]
        rows = [[f"<strong>{esc(r['model'])}</strong>" + (" ★" if r["selected"] else ""), _f(r["cv_pr_auc"]),
                 _f(r["test_roc_auc"]), _f(r["test_pr_auc"]), _f(r["decision_threshold"], 3)] for r in mc]
        rows.insert(0, ["<strong>DNN (Practical 7)</strong>", "—", _f(dl["test_metrics"]["roc_auc"]),
                        _f(dl["test_metrics"]["pr_auc"]), _f(dl["threshold"]["threshold"], 3)])
        inner = (_table(["Model", "CV PR-AUC", "Test ROC-AUC", "Test PR-AUC", "Threshold"], rows, "num")
                 + f'<p class="s-note">The test laps hold {dl["test_metrics"]["n_positive"]} pit stop — rank the models '
                   f'on the cross-validated column. Thresholds are tuned on out-of-fold predictions.</p>')
    elif kind == "results_search":
        comp = _load_json("artifacts/search/reports/comparison.json")["algorithms"]
        rows = [[f"<strong>{esc(a['algorithm'])}</strong>", _f(a["solution_cost"], 2), f"{a['cost_gap_pct']:.2f}%",
                 f"{a['nodes_expanded']:,}", "optimal" if a["is_optimal"] else "—"] for a in comp]
        inner = (_table(["Algorithm", "Race time (s)", "Gap", "Nodes expanded", ""], rows, "num")
                 + '<p class="s-note">UCS and A* both find the optimum; the admissible heuristic lets A* '
                   'expand fewer nodes. DFS finds a plan fastest but a costly one.</p>')
    else:
        raise ValueError(f"unknown slide kind {kind!r}")
    return (f'<section class="slide k-{kind}" id="slide-{i}" data-slide="{i}" aria-roledescription="slide" '
            f'aria-label="Slide {i} of {len(SLIDES)}" hidden><div class="s-inner">{title}{inner}</div>'
            f'<p class="s-num">{i} / {len(SLIDES)}</p></section>')


def render_showcase_tab(facts: dict) -> str:
    slides = "".join(render_slide(i + 1, s, facts) for i, s in enumerate(SLIDES))
    options = "".join(
        f'<option value="{i + 1}">{i + 1}. {esc(re.sub(r"<[^>]+>", "", md_inline(fill(s["title"], facts))))}</option>'
        for i, s in enumerate(SLIDES))
    return f"""
<section class="tab-panel showcase" id="tab-showcase" data-tab="showcase" role="tabpanel" aria-labelledby="tabbtn-showcase" hidden>
  <div class="deck" id="deck" aria-roledescription="carousel" aria-label="Project showcase" data-view data-crumb="Showcase" data-parent="home">
    <div class="stage-wrap"><div class="stage" id="stage">{slides}</div></div>
    <div class="deck-bar">
      <button type="button" class="ctl" id="prev-slide" aria-label="Previous slide">‹</button>
      <label class="visually-hidden" for="slide-jump">Go to slide</label>
      <select id="slide-jump">{options}</select>
      <button type="button" class="ctl" id="next-slide" aria-label="Next slide">›</button>
      <div class="progress" aria-hidden="true"><span id="progress-fill"></span></div>
      <button type="button" class="ctl wide" id="fullscreen" aria-label="Enter full screen">⛶ Present</button>
    </div>
    <p class="deck-hint">← → or Space to move · F for full screen · Esc to exit · swipe on touch screens</p>
  </div>
</section>"""


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def check_anchors(page: str, report: BuildReport) -> None:
    ids = set(re.findall(r'\sid="([^"]+)"', page))
    for target in sorted(set(re.findall(r'href="#([^"]+)"', page))):
        if target not in ids:
            report.broken_anchors.append(target)


def build(output: Path = DEFAULT_OUTPUT, live: bool = True) -> BuildReport:
    output = Path(output).resolve()
    out_dir = output.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    report = BuildReport()
    facts = collect_facts()

    reports = render_reports(out_dir, report)
    guide = render_guide_tab(facts, len(reports), out_dir)
    diagrams = render_diagrams_tab(lambda t: fill(t, facts), report.missing_outputs)
    code = render_code_tab(facts, out_dir, report, live)
    reports_tab = render_reports_tab(reports, out_dir)
    showcase = render_showcase_tab(facts)

    css = (REPO_ROOT / "docs" / "assets" / "docs.css").read_text() + (ASSET_DIR / "guide.css").read_text()
    js = (ASSET_DIR / "guide.js").read_text()
    built = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>F1 Strategy Study Guide</title>
<meta name="description" content="Study guide, diagrams, code walkthrough, reports and showcase deck for the F1 Race Strategy Intelligence project.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=Saira+Condensed:wght@600;700&display=swap">
<style>
{css}
</style>
</head>
<body>
<a class="skip" href="#main">Skip to content</a>
<header class="topbar">
  <button type="button" class="menu" id="menu-toggle" aria-label="Show navigation" aria-expanded="false">☰</button>
  <a class="brand" href="#home"><span class="mark" aria-hidden="true"></span>F1 Strategy <span>Study Guide</span></a>
  <nav class="tabs" role="tablist" aria-label="Views">
    <button type="button" role="tab" id="tabbtn-guide" aria-controls="tab-guide" data-target="guide" aria-selected="true" aria-label="Study Guide"><span class="lg">Study </span>Guide</button>
    <button type="button" role="tab" id="tabbtn-diagrams" aria-controls="tab-diagrams" data-target="diagrams" aria-selected="false" aria-label="Diagrams and Workflow">Diagrams<span class="lg"> &amp; Workflow</span></button>
    <button type="button" role="tab" id="tabbtn-code" aria-controls="tab-code" data-target="code" aria-selected="false" aria-label="Understand the Code"><span class="lg">Understand the </span>Code</button>
    <button type="button" role="tab" id="tabbtn-reports" aria-controls="tab-reports" data-target="reports" aria-selected="false">Reports</button>
    <button type="button" role="tab" id="tabbtn-showcase" aria-controls="tab-showcase" data-target="showcase" aria-selected="false">Showcase</button>
  </nav>
</header>
<nav class="pagebar" aria-label="Page navigation">
  <button type="button" class="back" id="back-btn" aria-label="Back">{svg("back", 18, 2.2)}<span>Back</span></button>
  <ol class="breadcrumb" id="breadcrumb" aria-label="You are here"></ol>
  <button type="button" class="theme" id="theme-toggle" aria-label="Toggle dark mode">◐</button>
</nav>
<main id="main">
{guide}
{diagrams}
{code}
{reports_tab}
{showcase}
</main>
<footer class="foot">Generated {built} by <code>scripts/build_study_guide.py</code> from the source code,
{report.docs} generated reports and the committed artifacts. Edit the sources, not this file.</footer>
<script>
{js}
</script>
</body>
</html>
"""
    check_anchors(page, report)
    output.write_text(page, encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-live", action="store_true",
                        help="Do not run the strategy pipeline for Practical 9's example response.")
    args = parser.parse_args()
    report = build(args.output, live=not args.no_live)
    print(f"Wrote {args.output} ({args.output.stat().st_size / 1024:.0f} KB): "
          f"{report.docs} reports, {len(PRACTICALS)} practicals, {len(SLIDES)} slides.")
    for label, items in (("Broken links in reports (rendered inert)", report.broken_links),
                         ("Broken in-page anchors", report.broken_anchors),
                         ("Outputs not generated yet", report.missing_outputs)):
        if items:
            print(f"\n{label}: {len(items)}")
            for item in items:
                print(f"  - {item}")
    return 1 if report.broken_anchors else 0


if __name__ == "__main__":
    sys.exit(main())
