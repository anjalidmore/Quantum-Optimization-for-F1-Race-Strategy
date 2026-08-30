"""
app.services.strategy_service
================================

Wires the race-strategy simulator (Task 6 spec, section 22) to the *real*
computational-intelligence modules built in Tasks 2, 3 and 6:

    User Race State
        -> Feature Construction   (app.services.feature_approximation)
        -> ML Prediction          (Task 6 cached pipelines)
        -> Expert System          (Task 2 forward-chaining inference)
        -> Search Optimisation    (Task 3 A* over the remaining stint)
        -> Strategy Recommendation

Every field in the response is produced by one of those real components.
Where a component can only work from an approximation (see
``feature_approximation.py``), the response says so explicitly — including,
per field (driver/team/compound), whether the currently trained model even
uses it at all ("context only" — see ``feature_approximation.relevance_for_target``).
"""
from __future__ import annotations

import pandas as pd

from app.intelligence.expert_system.inference import ConflictResolution, InferenceEngine
from app.intelligence.expert_system.rule_base import build_rule_base
from app.intelligence.features.contract import load_feature_contract
from app.intelligence.search.algorithms import astar_search
from app.intelligence.search.problem import Compound, RaceProblem
from app.services.feature_approximation import build_feature_row, relevance_for_target
from app.services.model_cache import ModelUnavailableError, get_model_cache

_RULE_BASE = None


def _rules():
    global _RULE_BASE
    if _RULE_BASE is None:
        _RULE_BASE = build_rule_base()
    return _RULE_BASE


_WEATHER_TO_RAIN_PROB = {"dry": 5, "damp": 40, "wet": 80, "extreme": 95}


def _run_ml(race_state, laptime_model: str | None, pit_model: str | None) -> dict:
    cache = get_model_cache()
    out = {
        "laptime": None,
        "pit_probability": None,
        "feature_rows": {},
        "approximated_features": {},
        "context_only": {},
        "errors": [],
    }

    for target, key, model_name in (
        ("target_laptime", "laptime", laptime_model),
        ("target_pit_next_lap", "pit_probability", pit_model),
    ):
        try:
            result = build_feature_row(target, race_state)
            out["feature_rows"][target] = result.row
            out["approximated_features"][target] = result.approximated
            out["context_only"][target] = {
                field: relevant == []
                for field, relevant in relevance_for_target(target).items()
            }

            name, pipeline = cache.get_pipeline(target, model_name)
            X = pd.DataFrame([result.row])
            if key == "laptime":
                out["laptime"] = {"model": name, "value": float(pipeline.predict(X)[0])}
            else:
                proba = float(pipeline.predict_proba(X)[0, 1])
                out["pit_probability"] = {"model": name, "value": proba, "predicted_class": int(pipeline.predict(X)[0])}
        except ModelUnavailableError as exc:
            out["errors"].append(str(exc))

    return out


def _run_expert_system(race_state) -> dict:
    engine = InferenceEngine(_rules(), strategy=ConflictResolution.SALIENCE)
    laps_remaining = max(race_state.total_laps - race_state.current_lap, 0)
    inputs = {
        "current_lap": race_state.current_lap,
        "total_laps": race_state.total_laps,
        "laps_remaining": laps_remaining,
        "current_position": race_state.current_position,
        "current_compound": race_state.tyre_compound.strip().upper(),
        "tyre_wear": min(100.0, race_state.tyre_age * 3.0),
        "track_temperature": race_state.track_temperature,
        "weather_severity": race_state.weather,
        "rain_probability": _WEATHER_TO_RAIN_PROB.get(race_state.weather, 5),
        "track_wet": race_state.weather in ("wet", "extreme"),
        "track_status": race_state.track_status,
        "fuel_margin": (race_state.fuel_kg - (laps_remaining * 1.8)) / 100.0,
    }
    result = engine.forward_chain(inputs)
    return {
        "inputs": inputs,
        "decisions": result.conclusions,
        "triggered_rules": [
            {"rule_id": f.rule_id, "name": f.rule_name, "matched_conditions": f.matched_conditions, "asserted": f.asserted}
            for f in result.firings
        ],
    }


def _run_search(race_state) -> dict:
    laps_remaining = max(race_state.total_laps - race_state.current_lap, 1)
    try:
        start_compound = Compound(race_state.tyre_compound.strip().upper())
    except ValueError:
        start_compound = Compound.MEDIUM

    problem = RaceProblem(
        total_laps=laps_remaining,
        start_compound=start_compound,
        track_temp=race_state.track_temperature,
        is_wet=race_state.weather in ("wet", "extreme"),
    )
    result = astar_search(problem)
    actions = [
        {"type": a.type.value, "compound": a.compound.value if a.compound else None}
        for a in result.solution_actions
    ]
    return {
        "algorithm": "A*",
        "found": result.found,
        "expected_cost_seconds": result.solution_cost if result.found else None,
        "plan": actions,
        "next_action": actions[0] if actions else None,
        "note": (
            f"Search re-plans the remaining {laps_remaining} laps from the current tyre "
            "as a fresh stint (search does not carry over tyre age already accumulated)."
        ),
    }


def run_strategy_analysis(race_state, laptime_model: str | None = None, pit_model: str | None = None) -> dict:
    ml = _run_ml(race_state, laptime_model, pit_model)
    expert = _run_expert_system(race_state)
    search = _run_search(race_state)

    pit_decision = expert["decisions"].get("pit_decision")
    if pit_decision is None and ml["pit_probability"] is not None:
        pit_decision = "PIT_NOW" if ml["pit_probability"]["predicted_class"] == 1 else "STAY_OUT"

    return {
        "race_state": race_state.model_dump(),
        "prediction": {
            "predicted_lap_time_seconds": ml["laptime"]["value"] if ml["laptime"] else None,
            "laptime_model": ml["laptime"]["model"] if ml["laptime"] else None,
            "probability_pit": ml["pit_probability"]["value"] if ml["pit_probability"] else None,
            "pit_model": ml["pit_probability"]["model"] if ml["pit_probability"] else None,
            "feature_rows": ml["feature_rows"],
            "approximated_features": ml["approximated_features"],
            "context_only": ml["context_only"],
            "errors": ml["errors"],
        },
        "recommended_action": pit_decision,
        "expected_cost_seconds": search["expected_cost_seconds"],
        "optimal_search_strategy": search,
        "triggered_expert_rules": expert["triggered_rules"],
        "evidence": expert["decisions"],
        "data_source": load_feature_contract().dataset_source["source"],
    }
