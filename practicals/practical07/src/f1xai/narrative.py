"""
f1xai.narrative
===============

Turns SHAP output into one or two sentences a race engineer can read mid-race.

The audience is not a data scientist. "SHAP value 0.31 for tyrelife_x_medium"
is not actionable on a pit wall; "tyre age is the dominant factor pushing
towards a stop" is. This module does that translation and nothing else - it
never invents a factor that SHAP did not surface, and the numbers it quotes
are the row's real feature values.

Feature names are mapped to plain English through an explicit glossary. Any
feature not in the glossary falls back to a readable form of its own name
rather than being dropped, so a new Task 5 feature can never silently vanish
from an explanation.
"""
from __future__ import annotations

GLOSSARY = {
    "tyre_life": "tyre age",
    "race_progress": "race progress",
    "gap_roll3_mean": "recent pace gap to the field",
    "gap_expanding": "pace gap trend across the stint",
    "field_median_lag1": "the field's pace last lap",
    "field_pace_trend": "the field's pace trend",
    "form_vs_baseline": "current form against this driver's baseline",
    "tracktemp_dev_x_tyrelife": "track temperature acting on tyre age",
    "tyrelife_x_medium": "tyre age on the medium compound",
    "tyrelife_x_soft": "tyre age on the soft compound",
    "stint_number": "stint number",
    "track_status": "track status",
    "wind_speed": "wind speed",
    "humidity": "humidity",
    "driver_dev": "this driver's pace deviation",
    "gap_roll3_std": "consistency of the recent pace gap",
}


def humanise(feature: str) -> str:
    if feature in GLOSSARY:
        return GLOSSARY[feature]
    if feature.startswith("driver_"):
        return f"driver identity ({feature.removeprefix('driver_').upper()})"
    if feature.startswith("team_"):
        return f"team identity ({feature.removeprefix('team_').replace('_', ' ').title()})"
    if feature.startswith("compound_"):
        return f"{feature.removeprefix('compound_')} compound"
    return feature.replace("_", " ")


def _magnitude(share: float) -> str:
    if share >= 0.45:
        return "the dominant"
    if share >= 0.25:
        return "a major"
    if share >= 0.12:
        return "a moderate"
    return "a minor"


def pit_decision_sentence(
    probability: float,
    shap_row: list[dict],
    feature_values: dict[str, float],
    trust_band: str,
) -> str:
    """One or two sentences explaining a pit recommendation."""
    recommend = "Recommend PITTING" if probability >= 0.5 else "Recommend STAYING OUT"
    # A probability of 0.0004 is not "0%" - rounding it away overstates the
    # model's certainty in exactly the direction a reader should not be misled.
    pct = probability * 100
    conf = f"{pct:.0f}%" if pct >= 1 else f"<1% ({pct:.2f}%)"
    total = sum(abs(r["shap_value"]) for r in shap_row) or 1.0

    clauses = []
    for r in shap_row[:3]:
        name = humanise(r["feature"])
        val = feature_values.get(r["feature"])
        share = abs(r["shap_value"]) / total
        pushes = "towards a stop" if r["shap_value"] > 0 else "against stopping"
        val_txt = f" ({val:.4g})" if val is not None else ""
        clauses.append(f"{name}{val_txt} is {_magnitude(share)} factor pushing {pushes}")

    body = "; ".join(clauses)
    return (
        f"{recommend} - model confidence {conf}. "
        f"{body.capitalize()}. "
        f"Trust in this recommendation: {trust_band}."
    )


def laptime_sentence(
    predicted_seconds: float,
    shap_row: list[dict],
    feature_values: dict[str, float],
    trust_band: str,
    base_value: float | None = None,
) -> str:
    """One or two sentences explaining a lap-time prediction."""
    total = sum(abs(r["shap_value"]) for r in shap_row) or 1.0
    clauses = []
    for r in shap_row[:3]:
        name = humanise(r["feature"])
        val = feature_values.get(r["feature"])
        share = abs(r["shap_value"]) / total
        effect = "slower" if r["shap_value"] > 0 else "faster"
        val_txt = f" ({val:.4g})" if val is not None else ""
        clauses.append(f"{name}{val_txt} is {_magnitude(share)} factor making this lap {effect}")

    vs_avg = ""
    if base_value is not None:
        delta = predicted_seconds - base_value
        vs_avg = f" - {abs(delta):.3f}s {'slower' if delta > 0 else 'faster'} than an average lap"

    return (
        f"Expected lap time {predicted_seconds:.3f}s{vs_avg}. "
        f"{'; '.join(clauses).capitalize()}. "
        f"Trust in this prediction: {trust_band}."
    )


def counterfactual_sentence(cf: dict) -> str:
    """Plain-English rendering of a single-feature perturbation scan."""
    name = humanise(cf["feature"])
    if not cf["reachable"]:
        lo, hi = cf["searched_range"]
        return (
            f"No value of {name} between {lo:.4g} and {hi:.4g} flips this recommendation "
            f"with the rest of the race state unchanged - the call is not sensitive to "
            f"{name} alone."
        )
    return (
        f"The recommendation flips when {name} reaches {cf['crossing_value']:.4g} "
        f"(currently {cf['original_value']:.4g}, a {cf['direction']} of "
        f"{abs(cf['delta_required']):.4g}), holding everything else fixed."
    )
