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

import functools
import inspect
import json
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from app.core.paths import CHART_DATA_DIR, REPO_ROOT

# Where ``write`` puts chart JSON while a pipeline runs with a redirected
# ``output_root``. None means the committed location, ``CHART_DATA_DIR``.
_OUTPUT_DIR: ContextVar[Path | None] = ContextVar("chart_output_dir", default=None)


def follows_output_root(fn):
    """Send the charts a pipeline writes to ``<output_root>/chart_data``.

    The ML, DL and QML pipelines take ``output_root`` so the tests can write
    every model, metric and figure under a tmp directory. Chart JSON used to
    ignore it and always land in ``artifacts/chart_data/``, so a "hermetic"
    test run still rewrote tracked files there (with ~1e-14 float noise from
    re-fitting). With no ``output_root`` nothing changes.
    """
    sig = inspect.signature(fn)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        root = sig.bind_partial(*args, **kwargs).arguments.get("output_root")
        if root is None:
            return fn(*args, **kwargs)
        token = _OUTPUT_DIR.set(Path(root) / "chart_data")
        try:
            return fn(*args, **kwargs)
        finally:
            _OUTPUT_DIR.reset(token)
    return wrapper


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
    """Write ``artifacts/chart_data/<name>.json`` — or, inside a pipeline run
    with a redirected ``output_root``, ``<output_root>/chart_data/<name>.json``.
    Returns the path."""
    out_dir = _OUTPUT_DIR.get() or CHART_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{name}.json"
    out.write_text(json.dumps(chart.as_dict(name, figure), indent=2) + "\n", encoding="utf-8")
    return out


def histogram(values: Iterable[float], bins: int = 20) -> tuple[list[float], list[int]]:
    """Bin edges' midpoints and counts, matching ``ax.hist(values, bins=bins)``."""
    import numpy as np

    counts, edges = np.histogram(np.asarray(list(values), dtype=float), bins=bins)
    mids = ((edges[:-1] + edges[1:]) / 2).tolist()
    return mids, counts.tolist()
