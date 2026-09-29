"""
Chart-data endpoints.

The dashboard used to show matplotlib PNGs. A picture cannot follow the theme,
cannot be read by a screen reader and cannot be inspected. So every plotting
function now writes its numbers beside the figure, and these routes serve them.

The PNGs are still the report deliverable and still on disk; this is the same
data, in the form a browser can draw.
"""
from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException

from app.core.paths import CHART_DATA_DIR

router = APIRouter(prefix="/api/charts", tags=["charts"])


@router.get("")
def index() -> dict:
    """Which charts exist, so the UI can ask before it renders."""
    if not CHART_DATA_DIR.exists():
        return {"available": False, "charts": [], "reason": "No chart data generated yet."}
    names = sorted(p.stem for p in CHART_DATA_DIR.glob("*.json"))
    return {"available": bool(names), "charts": names}


@router.get("/{name}")
def chart(name: str) -> dict:
    """One chart's numbers.

    ``name`` is matched against the files on disk rather than joined onto a
    path, so a crafted name cannot walk out of the directory.
    """
    if name not in {p.stem for p in CHART_DATA_DIR.glob("*.json")}:
        raise HTTPException(
            status_code=404,
            detail=f"No chart data named {name!r}. Run scripts/build_all.py to generate it.",
        )
    return json.loads((CHART_DATA_DIR / f"{name}.json").read_text())
