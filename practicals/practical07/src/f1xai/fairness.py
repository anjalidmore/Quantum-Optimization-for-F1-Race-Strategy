"""
f1xai.fairness
==============

Is the model predicting from **race state** or from **who is driving**?

This is the concrete fairness risk in this project, and it is not
hypothetical. Task 5's feature selection retained one-hot driver and team
dummies (``driver_ham``, ``team_mercedes``, ...). A lap-time model that leans
on ``driver_sai`` has learned "Sainz laps are like this" rather than "a tyre
this old on a track this hot laps like this". That model:

* cannot generalise to a driver it has not seen,
* silently encodes a driver's historical car performance as if it were the
  driver's own pace, and
* would give a different strategy call for two cars in an identical race state
  purely because of the name on the car.

The measurement is deliberately simple and hard to argue with: take the mean
|SHAP| attribution per feature, and report what **share of total attribution**
goes to identity features versus genuine race-state features. That share is
compared against the share you would expect if attribution were spread evenly
across features, so "3 of 45 features carry 40% of the attribution" reads as
the concentration it is.

No threshold here is a pass/fail verdict - the function reports the number and
a plain-language reading, and the report states it whichever way it falls.
"""
from __future__ import annotations

import numpy as np


def assess(shap_ranking: list[dict], identity_features: list[str]) -> dict:
    """``shap_ranking`` is the output of ``shap_analysis.global_ranking``."""
    total = sum(r["mean_abs_shap"] for r in shap_ranking)
    n_features = len(shap_ranking)
    identity = [r for r in shap_ranking if r["feature"] in set(identity_features)]
    race_state = [r for r in shap_ranking if r["feature"] not in set(identity_features)]

    identity_attr = sum(r["mean_abs_shap"] for r in identity)
    share = float(identity_attr / total) if total > 0 else 0.0
    expected = len(identity) / n_features if n_features else 0.0
    concentration = float(share / expected) if expected > 0 else None

    ranks = {r["feature"]: i + 1 for i, r in enumerate(shap_ranking)}
    top_identity = min((ranks[r["feature"]] for r in identity), default=None)

    return {
        "n_features": n_features,
        "n_identity_features": len(identity),
        "identity_features": [r["feature"] for r in identity],
        "identity_attribution_share": round(share, 4),
        "expected_share_if_uniform": round(float(expected), 4),
        "concentration_ratio": round(concentration, 3) if concentration is not None else None,
        "highest_ranked_identity_feature": (
            {"feature": shap_ranking[top_identity - 1]["feature"], "rank": top_identity}
            if top_identity else None
        ),
        "race_state_attribution_share": round(1.0 - share, 4),
        "top_race_state_features": [r["feature"] for r in race_state[:5]],
        "reading": _reading(share, expected, concentration, len(identity), n_features, top_identity),
    }


def _reading(share, expected, concentration, n_identity, n_features, top_rank) -> str:
    if n_identity == 0:
        return (
            "No driver or team identity features survived Task 5's selection funnel for this "
            "target, so this model cannot be leaning on identity - the risk does not arise here."
        )
    pct, exp_pct = share * 100, expected * 100
    lead = (
        f"{n_identity} of {n_features} selected features encode driver or team identity. "
        f"They carry **{pct:.1f}%** of the model's total attribution, against **{exp_pct:.1f}%** "
        f"if attribution were spread evenly"
    )
    if concentration is None:
        return lead + "."
    if concentration >= 2.0:
        return (
            lead + f" - a concentration of **{concentration:.2f}x**. Identity is doing "
            f"substantially more work than its share of the feature count, and the "
            f"highest-ranked identity feature sits at rank {top_rank}. This model is "
            f"partly predicting *who* rather than *what*, and it should not be expected "
            f"to transfer to an unseen driver."
        )
    if concentration >= 1.2:
        return (
            lead + f" - a concentration of **{concentration:.2f}x**. Identity is somewhat "
            f"over-weighted relative to its feature count. Worth monitoring, and worth "
            f"re-checking on a multi-race dataset before trusting cross-driver transfer."
        )
    return (
        lead + f" - a concentration of **{concentration:.2f}x**, at or below what an even "
        f"spread would give. Race-state features dominate this model's reasoning, which is "
        f"the desired outcome."
    )
