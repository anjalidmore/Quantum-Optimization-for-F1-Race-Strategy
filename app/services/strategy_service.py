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

from functools import lru_cache

import pandas as pd

from app.intelligence.expert_system.inference import ConflictResolution, InferenceEngine
from app.intelligence.expert_system.rule_base import build_rule_base
from app.intelligence.features.contract import load_feature_contract
from app.intelligence.search.algorithms import astar_search
from app.intelligence.search.problem import Compound, RaceProblem
from app.services.feature_approximation import build_feature_row, relevance_for_target
from app.services.model_cache import ModelUnavailableError, get_model_cache

_RULE_BASE = None

# ---------------------------------------------------------------------------
# Rule-base input coverage
#
# strategy_service._run_expert_system builds its ``inputs`` dict for a live
# request. An audit (scripts/evaluate_system.py's rule_base_reachability_check,
# run against the real 2023 Bahrain backtest) found the Task 2 rule base
# references 25 distinct input keys across its 32 rules, while this function
# originally populated only 12 — so 14 rules could never fire through the live
# pipeline however correct they were in isolation. Investigating each:
#
# Genuinely computable from data this system actually has (added below):
#   in_pit_window          — bucketed from tyre_wear, already computed here.
#                             This project's rule base never numerically
#                             defined "pit window"; 35-80% wear is this
#                             implementation's own definition, chosen to open
#                             before R-PIT-004's own 55% "priming" threshold
#                             (a window should open earlier than the point a
#                             crew is actively primed) and close at R-PIT-001's
#                             80% "critical" cutoff.
#   safety_car_probability — the empirical % of laps under SC/VSC in the
#                             training race (2023 Bahrain GP), read from the
#                             cleaned lap data. A historical base rate, not a
#                             live prediction — this system has no real-time
#                             incident model.
#   safety_car_likelihood  — the same rate, bucketed to low/medium/high.
#   tyre_age_laps           — a straight alias of race_state.tyre_age. Not a
#                             missing computation at all, just a naming gap
#                             between this function's inputs dict and the
#                             rule that reads it.
#
# NOT computable — genuine limitations, not fixed here, because the data
# simply isn't anywhere in this system:
#   undercut_threat, overcut_opportunity, gap_ahead, gap_behind, drs_enabled
#       — all require a specific rival car's state (its gap and tyre age).
#         RaceStateRequest is a single-car snapshot; this project has no
#         rival-car telemetry anywhere. (Note: the feature contract's
#         gap_roll3_mean/gap_expanding features mean "gap to the field
#         median pace", a different quantity — using them here would be a
#         real fabrication, not a reasonable proxy, so they are not reused.)
#   circuit, grid_position
#       — this system only ever runs on one circuit (2023 Bahrain GP, fixed
#         by the training data) and RaceStateRequest has no starting-grid
#         field.
#   overtaking_difficulty, pit_loss
#       — circuit characteristics not present in any dataset this system
#         reads. Asserting a number would be an opinion presented as a fact.
#   graining_risk
#       — no tyre-graining model exists anywhere in this codebase to ground
#         a value on (R-DEG-001/002's degradation thresholds are the only
#         precedent, and they model a different mechanism: thermal
#         degradation from track temperature, not graining).
#
# Net effect: R-SC-002, R-SC-003, R-PIT-004 and R-STRAT-004 become reachable.
# R-PIT-002, R-PIT-003, R-DEG-003, R-STRAT-001/002/003, R-ERS-001, R-TAC-001/
# 002/003 remain documented limitations — listed as such in the Task 9/10
# report and the SDD, not silently dropped.
# ---------------------------------------------------------------------------


def _map_track_status(raw) -> str:
    """FastF1 packs TrackStatus as a string of concatenated digit codes (a lap
    can carry more than one, e.g. '2671' = yellow then VSC then VSC-ending
    within one lap): 1=clear, 2/3=yellow, 4=SC, 5=red, 6/7=VSC. This is this
    project's own severity-priority mapping onto the simulator's
    GREEN|YELLOW|SC|VSC|RED enum — not an official FastF1 API."""
    s = str(raw)
    if "5" in s:
        return "RED"
    if "4" in s:
        return "SC"
    if "6" in s or "7" in s:
        return "VSC"
    if "2" in s or "3" in s:
        return "YELLOW"
    return "GREEN"


@lru_cache(maxsize=1)
def _historical_safety_car_rate() -> float:
    """Empirical % of laps under SC/VSC in the training race, read from the
    cleaned lap data and cached per process (the training data doesn't change
    at runtime). Returns 0.0 if the cleaned data isn't available rather than
    raising — a missing historical rate degrades the two rules that use it,
    it shouldn't fail the whole expert-system stage."""
    from app.core.paths import FASTF1_LAPS_CLEAN_CSV

    if not FASTF1_LAPS_CLEAN_CSV.exists():
        return 0.0
    laps = pd.read_csv(FASTF1_LAPS_CLEAN_CSV)
    statuses = laps["TrackStatus"].apply(_map_track_status)
    return float(statuses.isin(["SC", "VSC"]).mean() * 100)


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
        "out_of_range": {},
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
            out["out_of_range"][target] = result.out_of_range
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
    tyre_wear = min(100.0, race_state.tyre_age * 3.0)
    sc_rate = _historical_safety_car_rate()
    inputs = {
        "current_lap": race_state.current_lap,
        "total_laps": race_state.total_laps,
        "laps_remaining": laps_remaining,
        "current_position": race_state.current_position,
        "current_compound": race_state.tyre_compound.strip().upper(),
        "tyre_wear": tyre_wear,
        "track_temperature": race_state.track_temperature,
        "weather_severity": race_state.weather,
        "rain_probability": _WEATHER_TO_RAIN_PROB.get(race_state.weather, 5),
        "track_wet": race_state.weather in ("wet", "extreme"),
        "track_status": race_state.track_status,
        "fuel_margin": (race_state.fuel_kg - (laps_remaining * 1.8)) / 100.0,
        # Added so R-SC-002, R-SC-003, R-PIT-004 and R-STRAT-004 can fire —
        # see the module-level "Rule-base input coverage" note above for what
        # each means and isn't fabricated from.
        "in_pit_window": 35.0 <= tyre_wear < 80.0,
        "safety_car_probability": sc_rate,
        "safety_car_likelihood": "low" if sc_rate < 5 else ("medium" if sc_rate < 15 else "high"),
        "tyre_age_laps": race_state.tyre_age,
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


def _run_dl(ml: dict) -> dict:
    """Task 7 DL prediction stage — lap time and pit probability from the same
    feature rows the ML stage built, using the tuned decision threshold (never
    0.5). Runs on every call, unlike the opt-in SHAP explanation below: a
    caller asking for a strategy call gets both model families' verdicts,
    which is what the recommendation engine needs to compare them.
    """
    from app.intelligence.xai.live import predict_point

    out = {"laptime": None, "pit_probability": None, "errors": []}
    for target, key in (("target_laptime", "laptime"), ("target_pit_next_lap", "pit_probability")):
        row = ml["feature_rows"].get(target)
        if row is None:
            out["errors"].append(f"{target}: no feature row (ML stage did not produce one).")
            continue
        result = predict_point(target, row)
        if not result.get("available"):
            out["errors"].append(f"{target}: {result.get('reason')}")
            continue
        if key == "laptime":
            out["laptime"] = {"model": result["model"], "value": result["deep_prediction"]}
        else:
            out["pit_probability"] = {
                "model": result["model"],
                "value": result["deep_prediction"],
                "predicted_class": result["predicted_class"],
                "threshold": result["decision_threshold"],
            }
    return out


def _validation_summary(race_state) -> dict:
    """Stage 1 for the response: the API layer's Pydantic schema and the
    router's domain-option checks (driver/team/compound/model choice against
    the live dataset) already ran before this function was called — a 422
    was raised if they failed. This stage just echoes the normalised inputs
    so the pipeline's first card has something to show.
    """
    laps_remaining = max(race_state.total_laps - race_state.current_lap, 0)
    return {
        "passed": True,
        "current_lap": race_state.current_lap,
        "total_laps": race_state.total_laps,
        "laps_remaining": laps_remaining,
        "tyre_compound": race_state.tyre_compound.strip().upper(),
        "note": "Schema and domain-option validation already passed at the API layer (422 otherwise).",
    }


def _combine_recommendation(expert: dict, ml: dict, dl: dict, search: dict) -> dict:
    """The Task 9 recommendation engine.

    Combining rule, in order:
    1. If an expert-system rule fired a ``pit_decision`` conclusion, that
       verdict wins outright — rules encode domain knowledge (safety car,
       mandatory-stop windows) no statistical model has, and Task 2's
       validation report is the trust basis for that precedence.
    2. Otherwise ML's and DL's pit-probability predicted classes are compared.
       If they agree, use that class. If they disagree, this is reported
       explicitly (``disagreement: true``) rather than silently picking one
       — the tie-break falls to ML, which had the better holdout precision
       on the one real pit stop in this dataset (see README's Task 6 vs 7
       comparison), and both numbers stay in the response.
    3. A "pit" verdict is refined into "PIT NOW" or "PIT IN N LAPS" using the
       A* search plan's first PIT action index (each plan entry is one lap of
       the remaining stint); a "no pit" verdict is "STAY OUT".
    """
    ml_pit, dl_pit = ml.get("pit_probability"), dl.get("pit_probability")
    ml_class = ml_pit["predicted_class"] if ml_pit else None
    dl_class = dl_pit["predicted_class"] if dl_pit else None
    expert_verdict = expert["decisions"].get("pit_decision")

    disagreement = False
    reasons: list[str] = []

    if expert_verdict is not None:
        source = "expert system"
        pit_now = str(expert_verdict).upper().startswith("PIT")
        confidence = "high"
        reasons.append(f"Expert-system rule verdict: {expert_verdict}.")
    elif ml_class is not None and dl_class is not None:
        if ml_class == dl_class:
            source = f"ML ({ml_pit['model']}) + DL agreement"
            pit_now = bool(ml_class)
            margin = min(abs(ml_pit["value"] - 0.5), abs(dl_pit["value"] - dl_pit["threshold"]))
            confidence = "high" if margin >= 0.1 else "moderate"
            reasons.append(
                f"ML ({ml_pit['model']}, p={ml_pit['value']:.3f}) and DL (p={dl_pit['value']:.3f}, "
                f"threshold {dl_pit['threshold']:.4f}) agree."
            )
        else:
            disagreement = True
            source = f"ML ({ml_pit['model']}) — tie-break, ML/DL disagree"
            pit_now = bool(ml_class)
            confidence = "low"
            reasons.append(
                f"ML and DL disagree: ML ({ml_pit['model']}) says "
                f"{'PIT' if ml_class else 'NO PIT'} (p={ml_pit['value']:.3f}); DL says "
                f"{'PIT' if dl_class else 'NO PIT'} (p={dl_pit['value']:.3f} at threshold "
                f"{dl_pit['threshold']:.4f}). Falling back to ML, which had the better holdout "
                "precision on this dataset's one real pit stop."
            )
    elif ml_class is not None:
        source, pit_now, confidence = f"ML only ({ml_pit['model']})", bool(ml_class), "moderate"
        reasons.append("DL prediction unavailable for this request.")
    elif dl_class is not None:
        source, pit_now, confidence = "DL only", bool(dl_class), "moderate"
        reasons.append("ML prediction unavailable for this request.")
    else:
        source, pit_now, confidence = "no model available", False, "none"
        reasons.append("Neither ML nor DL pit prediction is available for this request.")

    if pit_now:
        plan = search.get("plan") or []
        first_pit = next((i for i, a in enumerate(plan) if a.get("type") == "PIT"), None)
        if not plan or first_pit == 0:
            action = "PIT NOW"
        elif first_pit is not None:
            action = f"PIT IN {first_pit} LAP{'S' if first_pit != 1 else ''}"
        else:
            action = "PIT NOW"
    else:
        action = "STAY OUT"

    return {
        "action": action,
        "confidence": confidence,
        "source": source,
        "disagreement": disagreement,
        "ml_pit_probability": ml_pit["value"] if ml_pit else None,
        "dl_pit_probability": dl_pit["value"] if dl_pit else None,
        "reason": " ".join(reasons),
    }


def _explain(ml: dict) -> dict:
    """Task 8 explanation of the feature rows this request just built.

    Opt-in, because the SHAP sampling adds latency a caller may not want. It
    explains *this* recommendation's rows, not a stored example. Import is
    local so a strategy call that does not ask for an explanation never pays
    the cost of loading the deep-learning stack.
    """
    from app.intelligence.xai.live import explain_feature_row

    out = {}
    for target, row in ml.get("feature_rows", {}).items():
        try:
            out[target] = explain_feature_row(target, row)
        except Exception as exc:  # pragma: no cover - never fail the prediction
            out[target] = {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
    return out


def run_strategy_analysis(
    race_state,
    laptime_model: str | None = None,
    pit_model: str | None = None,
    explain: bool = False,
) -> dict:
    validation = _validation_summary(race_state)
    ml = _run_ml(race_state, laptime_model, pit_model)
    expert = _run_expert_system(race_state)
    dl = _run_dl(ml)
    search = _run_search(race_state)
    recommendation = _combine_recommendation(expert, ml, dl, search)

    pit_decision = expert["decisions"].get("pit_decision")
    if pit_decision is None and ml["pit_probability"] is not None:
        pit_decision = "PIT_NOW" if ml["pit_probability"]["predicted_class"] == 1 else "STAY_OUT"

    response = {
        "race_state": race_state.model_dump(),
        "validation": validation,
        "feature_construction": {
            "feature_rows": ml["feature_rows"],
            "approximated_features": ml["approximated_features"],
            "out_of_range": ml["out_of_range"],
            "context_only": ml["context_only"],
        },
        "prediction": {
            "predicted_lap_time_seconds": ml["laptime"]["value"] if ml["laptime"] else None,
            "laptime_model": ml["laptime"]["model"] if ml["laptime"] else None,
            "probability_pit": ml["pit_probability"]["value"] if ml["pit_probability"] else None,
            "pit_model": ml["pit_probability"]["model"] if ml["pit_probability"] else None,
            "feature_rows": ml["feature_rows"],
            "approximated_features": ml["approximated_features"],
            "out_of_range": ml["out_of_range"],
            "context_only": ml["context_only"],
            "errors": ml["errors"],
        },
        "dl_prediction": {
            "predicted_lap_time_seconds": dl["laptime"]["value"] if dl["laptime"] else None,
            "probability_pit": dl["pit_probability"]["value"] if dl["pit_probability"] else None,
            "predicted_class": dl["pit_probability"]["predicted_class"] if dl["pit_probability"] else None,
            "threshold": dl["pit_probability"]["threshold"] if dl["pit_probability"] else None,
            "errors": dl["errors"],
        },
        "recommended_action": pit_decision,
        "recommendation": recommendation,
        "expected_cost_seconds": search["expected_cost_seconds"],
        "optimal_search_strategy": search,
        "search_plan": search,
        "triggered_expert_rules": expert["triggered_rules"],
        "evidence": expert["decisions"],
        "data_source": load_feature_contract().dataset_source["source"],
    }
    if explain:
        response["explanation"] = _explain(ml)
        response["xai_explanation"] = response["explanation"]
    else:
        response["xai_explanation"] = {
            "available": False,
            "reason": "explain=false on this request. Set explain=true to attach Task 8 SHAP "
                      "factors and a trust score for this prediction's feature rows.",
        }
    return response
