"""
f1data.visualize
================

Visual analytics for Task 4. Produces the individual analysis figures and a
composite dashboard PNG. Uses seaborn/matplotlib on the headless Agg backend.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

sns.set_theme(style="whitegrid")

_COMPOUND_COLOURS = {"SOFT": "#E10600", "MEDIUM": "#F5C518", "HARD": "#BBBBBB"}


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def correlation_heatmap(matrix: pd.DataFrame, path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(matrix, annot=False, cmap="coolwarm", center=0, square=True,
                linewidths=0.4, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Correlation Matrix (numeric features)", fontsize=14,
                 fontweight="bold")
    return _save(fig, Path(path))


def driver_points_bar(driver_df: pd.DataFrame, path: str | Path,
                      top_n: int = 10) -> Path:
    d = driver_df.head(top_n)
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(data=d, x="total_points", y="surname", hue="surname",
                palette="viridis", legend=False, ax=ax)
    ax.set_title("Top Drivers by Total Points", fontsize=14, fontweight="bold")
    ax.set_xlabel("Total points")
    ax.set_ylabel("")
    return _save(fig, Path(path))


def constructor_points_bar(con_df: pd.DataFrame, path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=con_df, x="total_points", y="name", hue="name",
                palette="rocket", legend=False, ax=ax)
    ax.set_title("Constructor Points", fontsize=14, fontweight="bold")
    ax.set_xlabel("Total points")
    ax.set_ylabel("")
    return _save(fig, Path(path))


def laptime_distribution(lap_times: pd.DataFrame, path: str | Path) -> Path:
    lt = pd.to_numeric(lap_times["milliseconds"], errors="coerce") / 1000.0
    lt = lt.dropna()
    # Focus on the plausible racing range to keep the histogram legible.
    lo, hi = lt.quantile(0.01), lt.quantile(0.99)
    lt = lt[(lt >= lo) & (lt <= hi)]
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.histplot(lt, bins=60, kde=True, color="#4363D8", ax=ax)
    ax.set_title("Lap-Time Distribution (1st–99th percentile)", fontsize=14,
                 fontweight="bold")
    ax.set_xlabel("Lap time (s)")
    return _save(fig, Path(path))


def pit_duration_box(pit_stops: pd.DataFrame, path: str | Path) -> Path:
    ps = pit_stops.copy()
    dur = pd.to_numeric(ps.get("duration"), errors="coerce")
    dur = dur.fillna(pd.to_numeric(ps["milliseconds"], errors="coerce") / 1000.0)
    dur = dur.dropna()
    dur = dur[(dur > 0) & (dur < dur.quantile(0.99))]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(y=dur, color="#3CB44B", ax=ax)
    ax.set_title("Pit-Stop Duration Distribution", fontsize=14, fontweight="bold")
    ax.set_ylabel("Duration (s)")
    return _save(fig, Path(path))


def tyre_degradation(fastf1_laps: pd.DataFrame, path: str | Path) -> Path:
    df = fastf1_laps.copy()
    df["LapTime"] = pd.to_numeric(df["LapTime"], errors="coerce")
    df["TyreLife"] = pd.to_numeric(df["TyreLife"], errors="coerce")
    df = df.dropna(subset=["LapTime", "TyreLife", "Compound"])
    # Filter obvious outliers for a readable trend.
    df = df[df["LapTime"] < df["LapTime"].quantile(0.98)]
    fig, ax = plt.subplots(figsize=(11, 6))
    for compound, grp in df.groupby("Compound"):
        colour = _COMPOUND_COLOURS.get(str(compound), "#666")
        trend = grp.groupby("TyreLife")["LapTime"].mean()
        ax.plot(trend.index, trend.values, marker="o", markersize=3,
                label=str(compound), color=colour)
    ax.set_title("Tyre Degradation — Mean Lap Time vs Tyre Life", fontsize=14,
                 fontweight="bold")
    ax.set_xlabel("Tyre life (laps)")
    ax.set_ylabel("Mean lap time (s)")
    ax.legend(title="Compound")
    return _save(fig, Path(path))


def build_dashboard(figures: Dict[str, Path], path: str | Path) -> Path:
    """Compose the individual figures into a single dashboard image."""
    order = ["driver", "constructor", "laptime", "pit", "tyre", "correlation"]
    available = [k for k in order if k in figures]
    n = len(available)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, 6 * rows))
    axes = np.array(axes).reshape(-1)
    import matplotlib.image as mpimg
    for ax, key in zip(axes, available):
        img = mpimg.imread(figures[key])
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(key.capitalize(), fontsize=13, fontweight="bold")
    for ax in axes[len(available):]:
        ax.axis("off")
    fig.suptitle("F1 Data Engineering — Visual Analytics Dashboard",
                 fontsize=20, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return _save(fig, Path(path))
