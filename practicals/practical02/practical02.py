#!/usr/bin/env python3
"""
practical02.py
==============

End-to-end driver for Phase 1 / Task 2 (Rule-Based Expert System).

It:

1. Builds and statically validates the curated rule base.
2. Serialises the rule base to JSON (portable, human-inspectable).
3. Runs forward chaining on several representative race scenarios.
4. Demonstrates backward chaining on a goal query.
5. Generates the rule catalogue, validation report, and per-scenario inference
   reports (with full HOW/WHY explanations).

Exits non-zero if rule-base validation fails, so it doubles as a CI gate.

Usage
-----
    python practical02.py [--output-dir OUTPUTS]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "src"))

from f1es import reports as rep  # noqa: E402
from f1es import rule_validation as rv  # noqa: E402
from f1es.inference import ConflictResolution, InferenceEngine  # noqa: E402
from f1es.rule_base import build_rule_base  # noqa: E402
from f1es.rules_schema import Rule  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("practical02")


# Representative scenarios exercising different rule families.
SCENARIOS = [
    ("Wet weather changeover", {
        "rain_probability": 85, "track_wet": True, "current_compound": "SOFT",
        "tyre_wear": 30, "weather_severity": "wet", "track_status": "GREEN",
        "current_lap": 12, "total_laps": 57, "laps_remaining": 45,
    }),
    ("Safety car opportunity", {
        "track_status": "SC", "tyre_wear": 52, "in_pit_window": True,
        "current_position": 5, "laps_remaining": 30, "weather_severity": "dry",
    }),
    ("Monaco track position", {
        "circuit": "Monaco", "grid_position": 2, "overtaking_difficulty": "high",
        "tyre_wear": 20, "weather_severity": "dry", "track_status": "GREEN",
        "current_compound": "MEDIUM",
    }),
    ("Undercut battle in the heat", {
        "in_pit_window": True, "undercut_threat": True, "track_temperature": 46,
        "pit_loss": 18, "overtaking_difficulty": "medium", "tyre_wear": 58,
        "current_compound": "SOFT", "weather_severity": "dry",
        "track_status": "GREEN", "laps_remaining": 25,
    }),
    ("Fuel-critical defensive stint", {
        "fuel_margin": -0.8, "gap_behind": 0.7, "drs_enabled": True,
        "tyre_wear": 64, "weather_severity": "dry", "track_status": "GREEN",
        "current_compound": "HARD", "laps_remaining": 10,
    }),
]


def _rule_to_dict(r: Rule) -> dict:
    """Serialise a Rule to a JSON-friendly dict."""
    return {
        "rule_id": r.rule_id,
        "name": r.name,
        "category": r.category,
        "salience": r.salience,
        "connective": r.connective.value,
        "description": r.description,
        "conditions": [
            {"key": c.key, "operator": c.operator.value, "value": c.value}
            for c in r.conditions
        ],
        "actions": [
            {"key": a.key, "value": a.value, "confidence": a.confidence}
            for a in r.actions
        ],
    }


def main(output_dir: Path) -> int:
    output_dir = output_dir.resolve()
    reports_dir = output_dir / "reports"
    rules_dir = output_dir / "rules"
    for d in (reports_dir, rules_dir):
        d.mkdir(parents=True, exist_ok=True)

    # --- 1. Build & validate ----------------------------------------------
    rules = build_rule_base()
    log.info("Built rule base: %d rules across %d categories.",
             len(rules), len({r.category for r in rules}))
    results = rv.validate_rule_base(rules)
    for r in results:
        log.info("  [%s] %s — %s", "PASS" if r.passed else "FAIL", r.name, r.detail)

    # --- 2. Serialise to JSON ---------------------------------------------
    (rules_dir / "rule_base.json").write_text(
        json.dumps([_rule_to_dict(r) for r in rules], indent=2), encoding="utf-8")
    log.info("Serialised rule base -> %s",
             (rules_dir / "rule_base.json").relative_to(output_dir))

    # --- 3. Forward-chain each scenario -----------------------------------
    engine = InferenceEngine(rules, strategy=ConflictResolution.SALIENCE)
    scenario_results = []
    for title, inputs in SCENARIOS:
        fr = engine.forward_chain(inputs)
        scenario_results.append((title, inputs, fr))
        log.info("Scenario '%s': %d firing(s), decisions=%s",
                 title, len(fr.firings), fr.conclusions)

    # --- 4. Backward-chaining demo ----------------------------------------
    demo_inputs = dict(SCENARIOS[1][1])  # safety-car scenario
    proof = engine.backward_chain("pit_decision", "PIT_NOW", demo_inputs)
    log.info("Backward-chaining 'pit_decision == PIT_NOW': proven=%s", proof.proven)
    (reports_dir / "backward_chaining_demo.md").write_text(
        "# Backward-Chaining Demonstration\n\n"
        "Goal: prove `pit_decision == 'PIT_NOW'` for the safety-car scenario.\n\n"
        "```\n" + proof.render() + "\n```\n", encoding="utf-8")

    # --- 5. Reports --------------------------------------------------------
    written = rep.generate_all(reports_dir, rules, results, scenario_results)
    for name, path in written.items():
        log.info("  wrote %s -> %s", name, path.relative_to(output_dir))

    if not rv.all_passed(results):
        log.error("Rule-base validation FAILED.")
        return 1
    log.info("All Task-2 deliverables generated successfully in %s", output_dir)
    return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the F1 rule-based expert system.")
    p.add_argument("--output-dir", type=Path, default=_HERE / "outputs")
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main(_parse_args().output_dir))
