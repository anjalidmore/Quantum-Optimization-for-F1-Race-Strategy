#!/usr/bin/env python3
"""
run_qml.py
==========

Runs the quantum machine-learning experiment and writes every artifact:

    artifacts/models/qml/          trained weights (.npy) and config JSON
    artifacts/metrics/qml_metrics.json
    artifacts/figures/qml_*.png    circuit diagram, loss curves, comparisons
    artifacts/reports/classical_vs_quantum_report.md

Everything is simulated on PennyLane's noiseless ``default.qubit``. Requires
Task 5's feature contract, and reads Task 6's committed metrics for the
classical reference, so run ``scripts/build_all.py`` first on a fresh clone.

Usage
-----
    python scripts/run_qml.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.intelligence.qml import pipeline  # noqa: E402


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                        datefmt="%H:%M:%S")
    results = pipeline.run_all()
    log = logging.getLogger("run_qml")
    for target, r in results["targets"].items():
        for m in r["quantum"] + r["classical_matched"]:
            key = "pr_auc" if r["task"] == "classification" else "mae"
            log.info("%-28s %-26s test %s = %s", target, m["model"], key.upper(),
                     m["test_metrics"].get(key))
    log.info("report: artifacts/reports/classical_vs_quantum_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
