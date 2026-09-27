"""
f1es.reports
============

Markdown deliverable generators for Task 2:

* **Rule catalogue** — every rule rendered as ``IF ... THEN ...`` with metadata,
  grouped by category.
* **Rule-base validation report** — the static-validation results.
* **Inference report** — a worked example: inputs, firing sequence, conclusions,
  and the full HOW/WHY explanation for a given scenario.

All output is Markdown so it renders on GitHub and converts cleanly to PDF/HTML.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

from .explanation import Explainer
from .inference import ForwardResult
from .rule_validation import RuleCheckResult
from .rules_schema import Rule


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def rule_catalogue_md(rules: Sequence[Rule]) -> str:
    by_cat: Dict[str, List[Rule]] = {}
    for r in rules:
        by_cat.setdefault(r.category, []).append(r)

    parts = [
        "# Rule Catalogue",
        "",
        f"_Generated {_ts()} — {len(rules)} rules across {len(by_cat)} categories._",
        "",
        "Salience convention: 100 weather-critical · 80 safety-car · 60 pit/deg · "
        "40 tyre · 20-30 tactical · 0 advisory.",
        "",
    ]
    for cat in sorted(by_cat):
        group = sorted(by_cat[cat], key=lambda r: r.rule_id)
        parts.append(f"## Category: `{cat}` ({len(group)} rules)")
        parts.append("")
        for r in group:
            parts.append(f"### {r.rule_id} — {r.name}")
            parts.append("")
            parts.append(f"- **Salience:** {r.salience} · **Specificity:** "
                         f"{r.specificity} · **Connective:** {r.connective.value}")
            parts.append(f"- **Logic:** `{r.describe()}`")
            if r.description:
                parts.append(f"- **Rationale:** {r.description}")
            parts.append("")
    return "\n".join(parts) + "\n"


def rule_tree_md(rules: Sequence[Rule]) -> str:
    """The rule base as a tree: category -> rule -> conditions and conclusions.

    The catalogue lists rules in prose; this shows the same rules as the
    structure the inference engine walks, which is what makes it possible to see
    at a glance which inputs can reach which conclusion.
    """
    by_cat: Dict[str, List[Rule]] = {}
    for r in rules:
        by_cat.setdefault(r.category, []).append(r)

    parts = [
        "# Rule Tree",
        "",
        f"_Generated {_ts()} — {len(rules)} rules across {len(by_cat)} categories._",
        "",
        "Read it as: **category → rule (salience) → conditions that must hold → what it asserts**.",
        "Higher salience fires first when several rules match.",
        "",
        "```text",
        "RULE BASE",
    ]
    cats = sorted(by_cat)
    for ci, cat in enumerate(cats):
        last_cat = ci == len(cats) - 1
        cbranch = "└──" if last_cat else "├──"
        cpad = "    " if last_cat else "│   "
        group = sorted(by_cat[cat], key=lambda r: (-r.salience, r.rule_id))
        parts.append(f"{cbranch} {cat}  ({len(group)} rules)")
        for ri, r in enumerate(group):
            last_rule = ri == len(group) - 1
            rbranch = "└──" if last_rule else "├──"
            rpad = "    " if last_rule else "│   "
            parts.append(f"{cpad}{rbranch} {r.rule_id}  salience {r.salience}  [{r.connective.value.upper()}]")
            for cond in r.conditions:
                value = "" if cond.value is None else f" {cond.value!r}"
                parts.append(f"{cpad}{rpad}│   IF   {cond.key} {cond.operator.value}{value}")
            for j, act in enumerate(r.actions):
                tip = "└──" if j == len(r.actions) - 1 else "├──"
                conf = "" if act.confidence >= 1.0 else f"  (confidence {act.confidence:.2f})"
                parts.append(f"{cpad}{rpad}{tip} THEN {act.key} = {act.value!r}{conf}")
    parts += ["```", ""]
    return "\n".join(parts) + "\n"


def decision_table_md(rules: Sequence[Rule]) -> str:
    """Condition/action matrix: every rule as a row, every input it reads as a column.

    A decision table makes two things checkable that prose cannot: which inputs
    the rule base actually consumes, and which conclusions more than one rule can
    assert (where salience decides the winner).
    """
    condition_keys = sorted({c.key for r in rules for c in r.conditions})
    action_keys = sorted({a.key for r in rules for a in r.actions})

    parts = [
        "# Decision Table",
        "",
        f"_Generated {_ts()} — {len(rules)} rules, {len(condition_keys)} input keys, "
        f"{len(action_keys)} output keys._",
        "",
        "Each row is one rule. A condition cell shows the test that input must pass; "
        "an action cell shows what the rule asserts. Empty means the rule ignores that key.",
        "",
        "## Conditions (inputs)",
        "",
        "| Rule | Salience | " + " | ".join(f"`{k}`" for k in condition_keys) + " |",
        "|---|---:|" + "---|" * len(condition_keys),
    ]
    for r in sorted(rules, key=lambda r: (-r.salience, r.rule_id)):
        tests = {c.key: f"{c.operator.value} {'' if c.value is None else c.value}".strip()
                 for c in r.conditions}
        row = " | ".join(tests.get(k, "") for k in condition_keys)
        parts.append(f"| `{r.rule_id}` | {r.salience} | {row} |")

    parts += ["", "## Actions (conclusions)", "",
              "| Rule | " + " | ".join(f"`{k}`" for k in action_keys) + " |",
              "|---|" + "---|" * len(action_keys)]
    for r in sorted(rules, key=lambda r: (-r.salience, r.rule_id)):
        asserts = {a.key: str(a.value) for a in r.actions}
        row = " | ".join(asserts.get(k, "") for k in action_keys)
        parts.append(f"| `{r.rule_id}` | {row} |")

    # Which conclusions are contested, and therefore decided by salience.
    writers: Dict[str, List[str]] = {}
    for r in rules:
        for a in r.actions:
            writers.setdefault(a.key, []).append(r.rule_id)
    contested = {k: v for k, v in writers.items() if len(v) > 1}
    parts += ["", "## Conclusions more than one rule can assert", ""]
    if not contested:
        parts.append("None: every conclusion has a single source rule.")
    else:
        parts += ["Conflict resolution is by salience, then specificity, then rule id "
                  "(see `app/intelligence/expert_system/inference.py`).", "",
                  "| Conclusion | Rules that can assert it |", "|---|---|"]
        for key in sorted(contested):
            parts.append(f"| `{key}` | {', '.join('`' + rid + '`' for rid in sorted(contested[key]))} |")
    parts.append("")
    return "\n".join(parts) + "\n"


def validation_report_md(results: Sequence[RuleCheckResult]) -> str:
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    status = "✅ ALL CHECKS PASSED" if passed == total else "❌ FAILURES PRESENT"
    rows = ["| Check | Result | Detail |", "|-------|--------|--------|"]
    for r in results:
        rows.append(f"| `{r.name}` | {'✅ pass' if r.passed else '❌ fail'} | {r.detail} |")
    return (
        f"# Rule-Base Validation Report\n\n_Generated {_ts()}._\n\n"
        f"**Summary:** {passed}/{total} checks passed — **{status}**\n\n"
        + "\n".join(rows) + "\n"
    )


def inference_report_md(
    title: str,
    inputs: Mapping[str, object],
    result: ForwardResult,
    rules: Sequence[Rule],
) -> str:
    explainer = Explainer(result, list(rules))
    parts = [
        f"# Inference Report — {title}",
        "",
        f"_Generated {_ts()}._",
        "",
        "## Inputs (GIVEN facts)",
        "",
        "| Fact | Value |",
        "|------|-------|",
    ]
    for k, v in inputs.items():
        parts.append(f"| `{k}` | {v!r} |")

    parts += ["", "## Firing sequence", ""]
    if result.firings:
        parts.append("| # | Iter | Rule | Name | Asserted |")
        parts.append("|---|------|------|------|----------|")
        for i, f in enumerate(result.firings, 1):
            asserted = ", ".join(f"{k}={v!r}" for k, v in f.asserted.items())
            parts.append(f"| {i} | {f.iteration} | `{f.rule_id}` | {f.rule_name} | {asserted} |")
    else:
        parts.append("_No rules fired for these inputs._")

    parts += ["", "## Conclusions (derived decisions)", ""]
    conclusions = result.conclusions
    if conclusions:
        parts.append("| Decision | Value |")
        parts.append("|----------|-------|")
        for k, v in conclusions.items():
            parts.append(f"| `{k}` | {v!r} |")
    else:
        parts.append("_No decisions derived._")

    parts += ["", "## WHY — narrative explanation", "", "```",
              explainer.narrative(), "```", ""]

    parts += ["## HOW — per-decision justification", ""]
    for j in explainer.justify_all():
        parts.append(f"**`{j.fact}` = {j.value!r}**  — via {j.rule_id} «{j.rule_name}»")
        for b in j.because:
            parts.append(f"  - because `{b}`")
        parts.append("")

    parts += ["## Audit trail", "", "```"]
    parts += explainer.audit_trail()
    parts += ["```", ""]

    return "\n".join(parts) + "\n"


def generate_all(
    output_dir: Path,
    rules: Sequence[Rule],
    validation_results: Sequence[RuleCheckResult],
    scenarios: Sequence,
) -> Dict[str, Path]:
    """
    Write all Task-2 reports.

    ``scenarios`` is a sequence of ``(title, inputs, ForwardResult)`` tuples.
    """
    output_dir = Path(output_dir)
    written = {
        "rule_catalogue": _write(output_dir / "rule_catalogue.md",
                                 rule_catalogue_md(rules)),
        "validation_report": _write(output_dir / "rule_validation_report.md",
                                    validation_report_md(validation_results)),
        "rule_tree": _write(output_dir / "rule_tree.md", rule_tree_md(rules)),
        "decision_table": _write(output_dir / "decision_table.md", decision_table_md(rules)),
    }
    for i, (title, inputs, result) in enumerate(scenarios, 1):
        slug = title.lower().replace(" ", "_").replace("/", "_")
        written[f"inference_{slug}"] = _write(
            output_dir / f"inference_{i:02d}_{slug}.md",
            inference_report_md(title, inputs, result, rules),
        )
    return written
