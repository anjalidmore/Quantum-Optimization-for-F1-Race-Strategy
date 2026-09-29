"""
app.intelligence.charts
=======================

One JSON file per figure, written from the *same arrays* that draw the PNG.

Why this exists: the dashboard used to show matplotlib PNGs. A picture cannot
be themed, cannot be read by a screen reader and cannot be inspected. So each
plotting function now also emits its numbers here, and the website draws them
itself.

The single rule that keeps them honest: a chart is emitted from inside the
same function that renders the figure, using the variables already passed to
matplotlib. Nothing recomputes anything, so the JSON cannot drift from the
picture without the picture changing too.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from app.core.paths import CHART_DATA_DIR, REPO_ROOT


def _plain(v: Any) -> Any:
    """numpy scalars and arrays -> JSON-safe Python, with NaN dropped to None."""
    if hasattr(v, "item") and getattr(v, "ndim", 0) == 0:
        v = v.item()
    if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
        return None
    if hasattr(v, "tolist"):
        return [_plain(x) for x in v.tolist()]
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return v


@dataclass
class Axis:
    label: str
    unit: str | None = None
    kind: str = "number"  # "number" or "category"


@dataclass
class Series:
    name: str
    x: Sequence[Any]
    y: Sequence[Any]
    role: str | None = None  # e.g. a tyre compound, so the UI can pick its token

    def as_dict(self) -> dict:
        xs, ys = _plain(self.x), _plain(self.y)
        if len(xs) != len(ys):
            raise ValueError(f"series {self.name!r}: {len(xs)} x-values but {len(ys)} y-values")
        d = {"name": self.name, "points": [{"x": a, "y": b} for a, b in zip(xs, ys)]}
        if self.role:
            d["role"] = self.role
        return d


@dataclass
class Chart:
    """A figure's numbers. `kind` tells the UI which component to draw."""

    kind: str
    title: str
    x: Axis
    y: Axis
    series: list[Series] = field(default_factory=list)
    caption: str | None = None
    extra: dict = field(default_factory=dict)

    def as_dict(self, name: str, figure: Path | None) -> dict:
        return {
            "name": name,
            "kind": self.kind,
            "title": self.title,
            "caption": self.caption,
            "axes": {
                "x": {"label": self.x.label, "unit": self.x.unit, "kind": self.x.kind},
                "y": {"label": self.y.label, "unit": self.y.unit, "kind": self.y.kind},
            },
            "series": [s.as_dict() for s in self.series],
            "source": _relative(figure) if figure else None,
            "data_source": _dataset_source(),
            **self.extra,
        }


def _relative(path: Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return path.name


def _dataset_source() -> str:
    """Whether these numbers came from a real session or the synthetic demo set.

    Read lazily and never cached: a chart written after a re-fetch must say so.
    """
    try:
        from app.intelligence.features.contract import load_feature_contract

        return load_feature_contract().dataset_source.get("source", "unknown")
    except Exception:
        return "unknown"


def write(name: str, chart: Chart, figure: Path | None = None) -> Path:
    """Write ``artifacts/chart_data/<name>.json``. Returns the path."""
    CHART_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = CHART_DATA_DIR / f"{name}.json"
    out.write_text(json.dumps(chart.as_dict(name, figure), indent=2) + "\n", encoding="utf-8")
    return out


def histogram(values: Iterable[float], bins: int = 20) -> tuple[list[float], list[int]]:
    """Bin edges' midpoints and counts, matching ``ax.hist(values, bins=bins)``."""
    import numpy as np

    counts, edges = np.histogram(np.asarray(list(values), dtype=float), bins=bins)
    mids = ((edges[:-1] + edges[1:]) / 2).tolist()
    return mids, counts.tolist()
