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
    }
    for i, (title, inputs, result) in enumerate(scenarios, 1):
        slug = title.lower().replace(" ", "_").replace("/", "_")
        written[f"inference_{slug}"] = _write(
            output_dir / f"inference_{i:02d}_{slug}.md",
            inference_report_md(title, inputs, result, rules),
        )
    return written
