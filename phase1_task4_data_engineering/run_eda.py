#!/usr/bin/env python3
"""
run_eda.py
==========

End-to-end driver for Phase 1 / Task 4 (Data Engineering & EDA).

It:

1. Generates a realistic synthetic dataset matching the Kaggle & FastF1 schemas
   (or loads real CSVs if ``--data-dir`` points at genuine files).
2. Loads and cleans every table (dedup, type coercion, lap-time parsing,
   categorical normalisation, missing-value imputation, outlier detection),
   recording a full cleaning audit.
3. Computes statistical summaries, correlation analysis and a data-quality
   scorecard.
4. Runs the nine domain analyses (driver, constructor, circuit, pit stop, tyre,
   lap time, weather, season, safety car).
5. Renders the analytics figures and a composite dashboard.
6. Writes all Markdown reports and a cleaned dataset.

Usage
-----
    python run_eda.py [--data-dir DIR] [--output-dir OUT] [--regenerate]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "src"))

from f1data import eda, reports as rep, synthetic, visualize as viz  # noqa: E402
from f1data.pipeline import clean_table, encode_and_scale, load_csv  # noqa: E402
from f1data.schemas import ALL_SCHEMAS  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("run_eda")

_TABLES = ["races", "drivers", "constructors", "circuits", "results",
           "pit_stops", "lap_times", "fastf1_laps"]


def main(data_dir: Path, output_dir: Path, regenerate: bool) -> int:
    data_dir = data_dir.resolve()
    output_dir = output_dir.resolve()
    reports_dir = output_dir / "reports"
    figures_dir = output_dir / "figures"
    clean_dir = output_dir / "clean"
    for d in (reports_dir, figures_dir, clean_dir):
        d.mkdir(parents=True, exist_ok=True)

    # --- 1. Data --------------------------------------------------------
    needs_gen = regenerate or not (data_dir / "results.csv").exists()
    if needs_gen:
        log.info("Generating synthetic dataset into %s ...", data_dir)
        synthetic.generate_all(data_dir)
    else:
        log.info("Using existing dataset in %s", data_dir)

    # --- 2. Load & clean ------------------------------------------------
    raw: dict[str, pd.DataFrame] = {}
    cleaned: dict[str, pd.DataFrame] = {}
    cleaning_reports = []
    quality_reports = []
    for name in _TABLES:
        schema = ALL_SCHEMAS[name]
        path = data_dir / f"{name}.csv"
        df = load_csv(path, schema)
        raw[name] = df
        cdf, creport = clean_table(df, schema, cap_outliers=True)
        cleaned[name] = cdf
        cleaning_reports.append(creport)
        quality_reports.append(eda.data_quality(cdf, name))
        cdf.to_csv(clean_dir / f"{name}_clean.csv", index=False)
        log.info("Cleaned %-14s rows %d -> %d, quality %.1f/100",
                 name, len(df), len(cdf), quality_reports[-1].quality_score)

    # --- 3. Statistics & correlation ------------------------------------
    log.info("Computing statistical summaries and correlation ...")
    summaries = {n: eda.statistical_summary(cleaned[n])
                 for n in ("results", "lap_times", "fastf1_laps", "pit_stops")}
    # Correlation over the FastF1 lap frame (richest numeric table).
    correlation = eda.correlation_analysis(cleaned["fastf1_laps"])

    # --- 4. Domain analyses ---------------------------------------------
    log.info("Running domain analyses ...")
    analyses = {
        "driver": eda.driver_analysis(cleaned["results"], cleaned["drivers"]),
        "constructor": eda.constructor_analysis(cleaned["results"],
                                                cleaned["constructors"]),
        "circuit": eda.circuit_analysis(cleaned["results"], cleaned["races"],
                                        cleaned["circuits"]),
        "pit_stop": eda.pit_stop_analysis(cleaned["pit_stops"]),
        "tyre": eda.tyre_analysis(cleaned["fastf1_laps"]),
        "lap_time": eda.lap_time_analysis(cleaned["lap_times"]),
        "weather": eda.weather_analysis(cleaned["fastf1_laps"]),
        "season": eda.season_analysis(cleaned["results"], cleaned["races"]),
        "safety_car": eda.safety_car_analysis(cleaned["fastf1_laps"]),
    }

    # --- 5. Encoding demo (for downstream ML) ---------------------------
    enc = encode_and_scale(cleaned["fastf1_laps"], ALL_SCHEMAS["fastf1_laps"])
    enc.frame.to_csv(clean_dir / "fastf1_laps_encoded_scaled.csv", index=False)
    log.info("Encoded/scaled FastF1 laps: %d columns, %d scaled.",
             enc.frame.shape[1], len(enc.scaled_columns))

    # --- 6. Visualisations ----------------------------------------------
    log.info("Rendering figures ...")
    figs = {}
    figs["correlation"] = viz.correlation_heatmap(
        correlation.matrix, figures_dir / "correlation_heatmap.png")
    figs["driver"] = viz.driver_points_bar(
        analyses["driver"], figures_dir / "driver_points.png")
    figs["constructor"] = viz.constructor_points_bar(
        analyses["constructor"], figures_dir / "constructor_points.png")
    figs["laptime"] = viz.laptime_distribution(
        cleaned["lap_times"], figures_dir / "laptime_distribution.png")
    figs["pit"] = viz.pit_duration_box(
        cleaned["pit_stops"], figures_dir / "pit_duration.png")
    figs["tyre"] = viz.tyre_degradation(
        cleaned["fastf1_laps"], figures_dir / "tyre_degradation.png")
    viz.build_dashboard(figs, figures_dir / "dashboard.png")

    # --- 7. Reports ------------------------------------------------------
    log.info("Generating reports ...")
    written = rep.generate_all(
        reports_dir, cleaning=cleaning_reports, quality=quality_reports,
        correlation=correlation, summaries=summaries, analyses=analyses)
    for name, path in written.items():
        log.info("  wrote %s -> %s", name, path.relative_to(output_dir))

    log.info("All Task-4 deliverables generated successfully in %s", output_dir)
    return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run F1 data engineering & EDA.")
    p.add_argument("--data-dir", type=Path, default=_HERE / "data" / "raw")
    p.add_argument("--output-dir", type=Path, default=_HERE / "outputs")
    p.add_argument("--regenerate", action="store_true",
                   help="Force regeneration of the synthetic dataset.")
    return p.parse_args()


if __name__ == "__main__":
    a = _parse_args()
    raise SystemExit(main(a.data_dir, a.output_dir, a.regenerate))
