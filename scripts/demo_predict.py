#!/usr/bin/env python3
"""
demo_predict.py
=================

Calls the running backend API with real requests to prove the trained
models actually respond — used by `run.sh`. Builds its lap-time / pit
feature payloads from whatever the *current* model registry says the
selected features are (`GET /api/ml/models`), so it works unchanged whether
the platform was trained on the synthetic demo data or a real fetched
FastF1 session — it never hard-codes a feature list that could drift out of
sync with the data contract.
"""
from __future__ import annotations

import argparse
import json
import sys

import httpx


def _print_json(label: str, payload: dict) -> None:
    print(f"\n=== {label} ===")
    print(json.dumps(payload, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    registry = httpx.get(f"{args.base_url}/api/ml/models", timeout=10).json()
    reg_features = next(m["features"] for m in registry["models"] if m["target"] == "target_laptime")
    clf_features = next(m["features"] for m in registry["models"] if m["target"] == "target_pit_next_lap")

    laptime_payload = {f: 0.0 for f in reg_features}
    pit_payload = {f: 0.0 for f in clf_features}

    r = httpx.post(f"{args.base_url}/api/ml/predict/laptime", json=laptime_payload, timeout=10)
    _print_json("Sample lap-time prediction (POST /api/ml/predict/laptime)", r.json())

    r = httpx.post(f"{args.base_url}/api/ml/predict/pit", json=pit_payload, timeout=10)
    _print_json("Sample pit-decision prediction (POST /api/ml/predict/pit)", r.json())

    race_state = {
        "driver": "ALO",
        "team": "ASTON MARTIN",
        "current_lap": 20,
        "total_laps": 55,
        "tyre_compound": "MEDIUM",
        "tyre_age": 15,
        "track_temperature": 40.0,
        "weather": "dry",
        "fuel_kg": 70,
        "track_status": "GREEN",
        "current_position": 6,
    }
    r = httpx.post(f"{args.base_url}/api/strategy/predict", json=race_state, timeout=30)
    strategy = r.json()
    _print_json(
        "Sample race-strategy simulation (POST /api/strategy/predict — ML + Expert System + Search)",
        {
            "predicted_lap_time_seconds": strategy.get("prediction", {}).get("predicted_lap_time_seconds"),
            "probability_pit": strategy.get("prediction", {}).get("probability_pit"),
            "recommended_action": strategy.get("recommended_action"),
            "expected_cost_seconds": strategy.get("expected_cost_seconds"),
            "triggered_expert_rules": [r["rule_id"] for r in strategy.get("triggered_expert_rules", [])],
            "data_source": strategy.get("data_source"),
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
