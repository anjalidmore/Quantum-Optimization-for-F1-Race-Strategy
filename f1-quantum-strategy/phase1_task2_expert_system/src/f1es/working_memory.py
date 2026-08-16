"""
f1es.working_memory
====================

Working memory (the "fact base") for the expert system, plus a registry of the
canonical **fact keys** the rule base is written against.

The :class:`WorkingMemory` is a thin, instrumented wrapper over a ``dict`` that:

* records the **provenance** of each fact (was it supplied by the user, or
  *derived* by a rule firing?), which the explanation subsystem uses to build a
  justification chain;
* tracks the **history** of assertions so inference can be replayed/audited;
* prevents silent type surprises by exposing typed getters.

The :data:`FACT_KEYS` registry documents every key a rule may read or write,
grouped by domain area. It is used by the rule validator to catch typos (a
condition referencing an unregistered key is flagged).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional


class Provenance(str, Enum):
    """Where a fact came from."""

    GIVEN = "given"      # supplied as input (race configuration / telemetry)
    DERIVED = "derived"  # asserted by a rule firing


@dataclass
class FactRecord:
    """A single fact together with its provenance and originating rule (if any)."""

    key: str
    value: Any
    provenance: Provenance
    rule_id: Optional[str] = None  # set when provenance is DERIVED


@dataclass
class WorkingMemory:
    """
    Mutable fact base for one inference run.

    Facts are stored as ``key -> value``; parallel metadata records provenance
    and an ordered assertion log.
    """

    _facts: Dict[str, Any] = field(default_factory=dict)
    _records: Dict[str, FactRecord] = field(default_factory=dict)
    _log: List[FactRecord] = field(default_factory=list)

    # ---- construction -----------------------------------------------------
    @classmethod
    def from_inputs(cls, inputs: Mapping[str, Any]) -> "WorkingMemory":
        """Create a working memory seeded with GIVEN facts."""
        wm = cls()
        for key, value in inputs.items():
            wm.assert_fact(key, value, Provenance.GIVEN)
        return wm

    # ---- core operations --------------------------------------------------
    def assert_fact(
        self,
        key: str,
        value: Any,
        provenance: Provenance = Provenance.DERIVED,
        rule_id: Optional[str] = None,
    ) -> bool:
        """
        Assert ``key = value``.

        Returns True if this changed working memory (new key, or new value),
        which the forward-chaining loop uses to detect when a fixed point is
        reached. Re-asserting an identical value returns False.
        """
        if self._facts.get(key, _MISSING) == value:
            return False
        record = FactRecord(key, value, provenance, rule_id)
        self._facts[key] = value
        self._records[key] = record
        self._log.append(record)
        return True

    def get(self, key: str, default: Any = None) -> Any:
        return self._facts.get(key, default)

    def as_mapping(self) -> Mapping[str, Any]:
        """Return a read-only view of the current facts (for condition tests)."""
        return dict(self._facts)

    def record_for(self, key: str) -> Optional[FactRecord]:
        return self._records.get(key)

    @property
    def log(self) -> List[FactRecord]:
        """Ordered list of every assertion made during the run."""
        return list(self._log)

    def derived_keys(self) -> List[str]:
        return [k for k, r in self._records.items() if r.provenance is Provenance.DERIVED]

    def __contains__(self, key: str) -> bool:
        return key in self._facts

    def __len__(self) -> int:
        return len(self._facts)


_MISSING = object()


# --------------------------------------------------------------------------- #
# Canonical fact-key registry
# --------------------------------------------------------------------------- #
# Grouped documentation of every key rules may reference. Values are short
# descriptions. The validator treats any condition/action key absent from here
# (after normalisation) as a potential typo.

FACT_KEYS: Dict[str, Dict[str, str]] = {
    "weather": {
        "rain_probability": "Probability of rain during the stint, percent (0-100).",
        "is_raining": "Boolean: is it currently raining.",
        "track_wet": "Boolean: is the track surface wet.",
        "air_temperature": "Ambient air temperature, °C.",
        "track_temperature": "Track surface temperature, °C.",
        "humidity": "Relative humidity, percent.",
        "wind_speed": "Wind speed, m/s.",
        "weather_severity": "Categorical severity: 'dry','damp','wet','extreme'.",
    },
    "tyre": {
        "current_compound": "Current tyre compound: SOFT/MEDIUM/HARD/INTER/WET.",
        "tyre_wear": "Estimated tyre wear, percent (0-100).",
        "tyre_age_laps": "Laps completed on the current set.",
        "tyre_deg_rate": "Estimated degradation rate, percent/lap.",
        "front_left_wear": "Front-left tyre wear, percent.",
        "graining_risk": "Categorical graining risk: 'low','medium','high'.",
    },
    "race_state": {
        "current_lap": "Current lap number.",
        "total_laps": "Total race distance in laps.",
        "laps_remaining": "Laps remaining (total_laps - current_lap).",
        "grid_position": "Starting grid position.",
        "current_position": "Current track position.",
        "gap_ahead": "Time gap to car ahead, seconds.",
        "gap_behind": "Time gap to car behind, seconds.",
        "fuel_load": "Estimated fuel on board, kg.",
        "fuel_margin": "Fuel margin vs. race requirement, kg (negative = short).",
    },
    "circuit": {
        "circuit": "Circuit identifier / name.",
        "overtaking_difficulty": "Categorical: 'low','medium','high'.",
        "pit_loss": "Time lost per pit stop at this circuit, seconds.",
        "safety_car_likelihood": "Prior SC likelihood: 'low','medium','high'.",
        "track_evolution": "Categorical grip evolution: 'low','medium','high'.",
    },
    "race_control": {
        "track_status": "Flag state: 'GREEN','YELLOW','SC','VSC','RED'.",
        "safety_car_probability": "Estimated near-term SC probability, percent.",
        "drs_enabled": "Boolean: is DRS currently enabled.",
    },
    "strategy_state": {
        "planned_stops": "Number of pit stops in the current plan.",
        "stops_made": "Number of stops already completed.",
        "in_pit_window": "Boolean: within the optimal pit window.",
        "undercut_threat": "Boolean: a rival is in undercut range.",
        "overcut_opportunity": "Boolean: an overcut is viable.",
    },
    # ---- derived decision variables (rule outputs) -----------------------
    "decisions": {
        "recommended_tyre": "Recommended next compound.",
        "pit_decision": "One of 'PIT_NOW','STAY_OUT','PIT_SOON','DELAY_PIT'.",
        "strategy_stops": "Recommended number of stops for the plan.",
        "tyre_deg_adjustment": "Qualitative deg adjustment: 'increase','decrease'.",
        "risk_level": "Overall strategic risk: 'low','medium','high'.",
        "engine_mode_advice": "Advice on engine/ERS mode.",
        "fuel_advice": "Fuel-management advice.",
        "push_advice": "Whether to push or conserve.",
        "defend_advice": "Defensive-driving advice.",
        "notes": "Free-text advisory note appended by rules.",
    },
}


def all_fact_keys() -> Dict[str, str]:
    """Flatten :data:`FACT_KEYS` into a single ``key -> description`` mapping."""
    flat: Dict[str, str] = {}
    for group in FACT_KEYS.values():
        flat.update(group)
    return flat
