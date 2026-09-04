"""
f1kr.reports
============

Generates the human-readable deliverable documents for Task 1:

* Entity table
* Attribute table
* Relationship table
* Ontology documentation
* Validation report

All documents are emitted as Markdown so they render directly on GitHub and can
be converted to PDF/HTML downstream. Every table is generated from the schema,
so documentation can never drift from the implementation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from .schema import ENTITIES, RELATIONSHIPS
from .validation import CheckResult


def _write(path: str | Path, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def entity_table_md() -> str:
    rows = ["| # | Entity | Category | Source | Parent | #Attrs | Description |",
            "|---|--------|----------|--------|--------|--------|-------------|"]
    for i, e in enumerate(ENTITIES, 1):
        rows.append(
            f"| {i} | `{e.name}` | {e.category} | {e.source} | "
            f"{e.parent or '—'} | {len(e.attributes)} | {e.description} |"
        )
    return (
        f"# Entity Table\n\n_Generated {_timestamp()} — "
        f"{len(ENTITIES)} entities._\n\n" + "\n".join(rows) + "\n"
    )


def attribute_table_md() -> str:
    rows = ["| Entity | Attribute | Type | Unit | Description |",
            "|--------|-----------|------|------|-------------|"]
    count = 0
    for e in ENTITIES:
        for a in e.attributes:
            count += 1
            rows.append(
                f"| `{e.name}` | `{a.name}` | {a.dtype.value} | "
                f"{a.unit or '—'} | {a.description} |"
            )
    return (
        f"# Attribute Table\n\n_Generated {_timestamp()} — "
        f"{count} attributes across {len(ENTITIES)} entities._\n\n"
        + "\n".join(rows) + "\n"
    )


def relationship_table_md() -> str:
    rows = ["| # | Relationship | Domain | → | Range | Cardinality | Description |",
            "|---|--------------|--------|---|-------|-------------|-------------|"]
    for i, r in enumerate(RELATIONSHIPS, 1):
        rows.append(
            f"| {i} | `{r.name}` | {r.domain} | → | {r.range} | "
            f"{r.cardinality.value} | {r.description} |"
        )
    return (
        f"# Relationship Table\n\n_Generated {_timestamp()} — "
        f"{len(RELATIONSHIPS)} relationships._\n\n" + "\n".join(rows) + "\n"
    )


def ontology_documentation_md(stats: Dict[str, int]) -> str:
    cats: Dict[str, List[str]] = {}
    for e in ENTITIES:
        cats.setdefault(e.category, []).append(e.name)

    parts = [
        "# Ontology Documentation",
        "",
        f"_Generated {_timestamp()}._",
        "",
        "## Overview",
        "",
        "The Formula 1 ontology is an OWL 2 ontology generated programmatically "
        "from the declarative domain schema (`f1kr.schema`). It formalises the "
        "racing domain as classes (entities), object properties (relationships) "
        "and data properties (attributes).",
        "",
        "* **Base IRI:** `http://f1kr.org/ontology/formula1#`",
        f"* **Classes:** {stats.get('classes', 'n/a')}",
        f"* **Object properties:** {stats.get('object_properties', 'n/a')}",
        f"* **Data properties:** {stats.get('data_properties', 'n/a')}",
        "",
        "## Class hierarchy by category",
        "",
    ]
    for cat in sorted(cats):
        parts.append(f"### {cat}")
        for name in cats[cat]:
            ent = next(e for e in ENTITIES if e.name == name)
            suffix = f" _(subclass of {ent.parent})_" if ent.parent else ""
            parts.append(f"- **{name}**{suffix} — {ent.description}")
        parts.append("")

    parts += [
        "## Cardinality encoding",
        "",
        "Relationship cardinalities are encoded as OWL property characteristics:",
        "",
        "| Cardinality | OWL characteristic |",
        "|-------------|--------------------|",
        "| `*..1` (many-to-one) | FunctionalProperty |",
        "| `1..1` (one-to-one) | FunctionalProperty + InverseFunctionalProperty |",
        "| `1..*` (one-to-many) | (plain object property) |",
        "| `*..*` (many-to-many) | (plain object property) |",
        "",
    ]
    return "\n".join(parts) + "\n"


def validation_report_md(results: List[CheckResult]) -> str:
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    status = "✅ ALL CHECKS PASSED" if passed == total else "❌ FAILURES PRESENT"
    rows = ["| Check | Result | Detail |", "|-------|--------|--------|"]
    for r in results:
        rows.append(f"| `{r.name}` | {'✅ pass' if r.passed else '❌ fail'} | {r.detail} |")
    return (
        f"# Knowledge Representation Validation Report\n\n"
        f"_Generated {_timestamp()}._\n\n"
        f"**Summary:** {passed}/{total} checks passed — **{status}**\n\n"
        + "\n".join(rows) + "\n"
    )


def generate_all(output_dir: str | Path, ontology_stats: Dict[str, int],
                 results: List[CheckResult]) -> Dict[str, Path]:
    """Write every report to ``output_dir`` and return a mapping of names -> paths."""
    output_dir = Path(output_dir)
    written = {
        "entity_table": _write(output_dir / "entity_table.md", entity_table_md()),
        "attribute_table": _write(output_dir / "attribute_table.md", attribute_table_md()),
        "relationship_table": _write(output_dir / "relationship_table.md",
                                     relationship_table_md()),
        "ontology_documentation": _write(output_dir / "ontology_documentation.md",
                                         ontology_documentation_md(ontology_stats)),
        "validation_report": _write(output_dir / "validation_report.md",
                                    validation_report_md(results)),
    }
    return written
