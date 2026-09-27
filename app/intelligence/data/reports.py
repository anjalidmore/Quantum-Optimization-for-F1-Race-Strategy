"""
f1data.reports
==============

Markdown deliverable generators for Task 4:

* EDA report (domain analyses)
* Correlation report
* Data-quality report
* Cleaning audit report
* Statistical-summary report
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .eda import CorrelationResult, QualityReport
from .pipeline import CleaningReport


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _df_md(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df is None or df.empty:
        return "_(no data)_"
    shown = df.head(max_rows)
    try:
        return shown.to_markdown(index=False)
    except Exception:
        # Fallback if the 'tabulate' backend is unavailable.
        return "```\n" + shown.to_string(index=False) + "\n```"


def cleaning_report_md(reports: List[CleaningReport]) -> str:
    parts = [f"# Data Cleaning Audit\n\n_Generated {_ts()}._\n"]
    for rep in reports:
        parts.append(f"## Table: `{rep.table}`\n")
        parts.append("| Step | Detail | Rows before | Rows after | Extra |")
        parts.append("|------|--------|-------------|------------|-------|")
        for s in rep.steps:
            extra = ", ".join(f"{k}={v}" for k, v in s.extra.items())
            parts.append(f"| `{s.step}` | {s.detail} | {s.rows_before} | "
                         f"{s.rows_after} | {extra} |")
        parts.append("")
    return "\n".join(parts) + "\n"


def quality_report_md(reports: List[QualityReport]) -> str:
    parts = [f"# Data Quality Report\n\n_Generated {_ts()}._\n"]
    parts.append("| Table | Rows | Cols | Completeness | Duplicates | "
                 "Constant cols | Quality score |")
    parts.append("|-------|------|------|--------------|------------|"
                 "---------------|---------------|")
    for r in reports:
        parts.append(
            f"| `{r.table}` | {r.n_rows} | {r.n_cols} | "
            f"{r.completeness:.1%} | {r.duplicate_rows} | "
            f"{len(r.constant_columns)} | **{r.quality_score:.1f}/100** |")
    parts.append("")
    for r in reports:
        worst = sorted(r.per_column_missing.items(), key=lambda kv: kv[1],
                       reverse=True)[:5]
        worst = [(c, m) for c, m in worst if m > 0]
        if worst:
            parts.append(f"### `{r.table}` — columns with missing values")
            parts.append("| Column | Missing % |")
            parts.append("|--------|-----------|")
            for c, m in worst:
                parts.append(f"| `{c}` | {m:.1%} |")
            parts.append("")
    return "\n".join(parts) + "\n"


def correlation_report_md(result: CorrelationResult) -> str:
    parts = [f"# Correlation Report\n\n_Generated {_ts()}._\n"]
    if not result.top_pairs:
        parts.append("_Insufficient numeric columns for correlation analysis._")
        return "\n".join(parts) + "\n"
    parts.append("## Strongest correlated feature pairs\n")
    parts.append("| Feature A | Feature B | Pearson r |")
    parts.append("|-----------|-----------|-----------|")
    for a, b, r in result.top_pairs:
        parts.append(f"| `{a}` | `{b}` | {r:+.3f} |")
    parts.append("\nSee `outputs/figures/correlation_heatmap.png` for the full "
                 "matrix.\n")
    return "\n".join(parts) + "\n"


def statistical_summary_md(summaries: Dict[str, pd.DataFrame]) -> str:
    parts = [f"# Statistical Summary\n\n_Generated {_ts()}._\n"]
    for table, summary in summaries.items():
        parts.append(f"## `{table}`\n")
        parts.append(_df_md(summary.reset_index().rename(
            columns={"index": "feature"})))
        parts.append("")
    return "\n".join(parts) + "\n"


def eda_report_md(analyses: Dict[str, pd.DataFrame]) -> str:
    titles = {
        "driver": "Driver Analysis",
        "constructor": "Constructor Analysis",
        "circuit": "Circuit Analysis",
        "pit_stop": "Pit-Stop Analysis",
        "tyre": "Tyre Analysis",
        "lap_time": "Lap-Time Analysis",
        "weather": "Weather Analysis",
        "season": "Season Analysis",
        "safety_car": "Safety-Car / Track-Status Analysis",
    }
    parts = [f"# Exploratory Data Analysis Report\n\n_Generated {_ts()}._\n",
             "This report summarises the domain analyses over the cleaned "
             "historical and FastF1 data. Figures are in `outputs/figures/`.\n"]
    for key, title in titles.items():
        if key in analyses:
            parts.append(f"## {title}\n")
            parts.append(_df_md(analyses[key]))
            parts.append("")
    return "\n".join(parts) + "\n"


def visualization_report_md(figures: Dict[str, Path], analyses: Dict[str, pd.DataFrame],
                            output_dir: Path) -> str:
    """Index every figure this run rendered, with what it shows and how to read it.

    The figures are the Task 4 visual deliverable; this report is the written
    companion, so a reader knows what each PNG is without opening all seven. A
    figure that failed to render is listed as missing rather than omitted.
    """
    captions = {
        "correlation": ("Correlation heatmap",
                        "Pairwise correlation across the numeric columns. The block that matters is the "
                        "sector times against lap time: they are near-perfectly correlated, which is the "
                        "evidence for excluding them as leakage in Task 5."),
        "driver": ("Points by driver",
                   "Championship points per driver in the loaded season tables — a sanity check that the "
                   "results table joins correctly to drivers."),
        "constructor": ("Points by constructor",
                        "The same aggregation at team level."),
        "laptime": ("Lap-time distribution",
                    "How lap times are spread. The right tail is in-laps, out-laps and traffic, which is "
                    "why Task 4 caps IQR outliers rather than deleting the rows."),
        "pit": ("Pit-stop duration",
                "Box plot of stationary time. Useful for the pit-loss constant the Task 3 search uses."),
        "tyre": ("Tyre degradation",
                 "Lap time against tyre age per compound — the physical relationship every later model "
                 "is trying to learn."),
        "dashboard": ("Composite dashboard",
                      "All of the above on one canvas, for a single-glance overview."),
    }

    parts = [
        "# Visualization Report",
        "",
        f"_Generated {_ts()} — {len(figures)} figures._",
        "",
        "Every figure below was rendered by `app/intelligence/data/visualize.py` from the cleaned "
        "tables in this same run. None is a stock image.",
        "",
        "| Figure | File | What it shows |",
        "|---|---|---|",
    ]
    for key in ("correlation", "driver", "constructor", "laptime", "pit", "tyre", "dashboard"):
        title, description = captions.get(key, (key.title(), ""))
        path = figures.get(key)
        if path is None:
            path = output_dir.parent / "figures" / f"{key}.png"
        exists = Path(path).exists()
        name = Path(path).name
        cell = f"`figures/{name}`" if exists else "**not generated**"
        parts.append(f"| {title} | {cell} | {description} |")

    parts += ["", "## How to read them together", "",
              "The correlation heatmap explains a modelling decision (the leakage exclusions). The "
              "tyre-degradation plot shows the effect the models are asked to predict. The lap-time "
              "distribution explains why the cleaning step caps outliers instead of dropping rows. The "
              "remaining three are data-integrity checks on the season tables.", ""]

    if "driver" in analyses and not analyses["driver"].empty:
        parts += [f"Figures cover {len(analyses['driver'])} drivers"
                  + (f" and {len(analyses['constructor'])} constructors" if "constructor" in analyses else "")
                  + " from the loaded tables.", ""]
    return "\n".join(parts) + "\n"


def generate_all(output_dir: Path, *, cleaning: List[CleaningReport],
                 quality: List[QualityReport], correlation: CorrelationResult,
                 summaries: Dict[str, pd.DataFrame],
                 analyses: Dict[str, pd.DataFrame],
                 figures: Dict[str, Path] | None = None) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    return {
        "cleaning": _write(output_dir / "cleaning_audit.md",
                           cleaning_report_md(cleaning)),
        "quality": _write(output_dir / "data_quality_report.md",
                          quality_report_md(quality)),
        "correlation": _write(output_dir / "correlation_report.md",
                              correlation_report_md(correlation)),
        "statistics": _write(output_dir / "statistical_summary.md",
                             statistical_summary_md(summaries)),
        "eda": _write(output_dir / "eda_report.md", eda_report_md(analyses)),
        "visualization": _write(output_dir / "visualization_report.md",
                                visualization_report_md(figures or {}, analyses, output_dir)),
    }
