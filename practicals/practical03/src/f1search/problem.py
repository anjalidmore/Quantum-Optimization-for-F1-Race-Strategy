"""
f1search.problem
================

Formal state-space formulation of the Formula 1 race-strategy problem.

We cast strategy selection as a **shortest-path search** over race states. The
objective is to complete the full race distance in **minimum estimated total
race time**, choosing when to pit and which tyre compound to fit.

State
-----
A :class:`RaceState` captures the strategically relevant variables requested for
Task 3:

* ``lap``            — laps completed so far (0 .. total_laps)
* ``compound``       — current tyre compound in use
* ``tyre_age``       — laps run on the current set
* ``stops_made``     — number of pit stops taken
* ``fuel_kg``        — remaining fuel (monotonically decreasing)

Track temperature, weather, safety-car status, gaps, ERS/DRS etc. are supplied
by the :class:`RaceProblem` *environment* (they parameterise the cost model)
rather than living in the state, keeping the state space tractable while still
honouring those factors in the transition cost.

Actions
-------
From any non-terminal state the agent may:

* ``RUN``  — complete one racing lap on the current tyres, or
* ``PIT(compound)`` — enter the pits, fit ``compound``, and complete the
  out-lap. Pitting incurs the circuit pit-loss and resets tyre age.

Cost
----
The cost of an action is the **estimated time** it adds to the race:

    lap_time(compound, tyre_age, fuel, track_temp) [+ pit_loss if pitting]

Lap time grows with tyre age (degradation), varies by compound (pace vs.
durability trade-off), improves slightly as fuel burns off, and worsens on a
hot track. The model is deliberately transparent and deterministic so that
search optimality can be reasoned about and unit-tested.

Goal
----
A state is terminal when ``lap == total_laps``. The goal test also enforces the
sporting rule that at least two distinct dry compounds must be used across a dry
race (a real F1 regulation), unless the race is wet.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Dict, FrozenSet, List, Optional, Tuple


class Compound(str, Enum):
    """Tyre compounds available to the strategy search."""

    SOFT = "SOFT"
    MEDIUM = "MEDIUM"
    HARD = "HARD"
    INTERMEDIATE = "INTERMEDIATE"
    WET = "WET"


DRY_COMPOUNDS: FrozenSet[Compound] = frozenset(
    {Compound.SOFT, Compound.MEDIUM, Compound.HARD}
)
WET_COMPOUNDS: FrozenSet[Compound] = frozenset(
    {Compound.INTERMEDIATE, Compound.WET}
)


class ActionType(str, Enum):
    RUN = "RUN"
    PIT = "PIT"


@dataclass(frozen=True)
class Action:
    """An action: run a lap, or pit and fit ``compound``."""

    type: ActionType
    compound: Optional[Compound] = None  # only set for PIT

    def __str__(self) -> str:
        if self.type is ActionType.PIT:
            return f"PIT->{self.compound.value}"
        return "RUN"


@dataclass(frozen=True)
class RaceState:
    """
    An immutable race-strategy state.

    Frozen + hashable so states can be used as dictionary keys / set members in
    the search frontier and explored set.
    """

    lap: int
    compound: Compound
    tyre_age: int
    stops_made: int
    fuel_kg: float
    compounds_used: FrozenSet[Compound] = field(default_factory=frozenset)

    def key(self) -> Tuple:
        """A hashable identity used for the explored/closed set.

        Fuel is bucketed to one decimal so that floating-point noise does not
        fragment otherwise-identical states.
        """
        return (self.lap, self.compound, self.tyre_age, self.stops_made,
                round(self.fuel_kg, 1), self.compounds_used)


@dataclass(frozen=True)
class TyreModel:
    """
    Per-compound pace and degradation parameters.

    ``base_pace`` is the fresh-tyre lap time (s) on a reference track; ``deg``
    is the per-lap time loss (s/lap) from wear; ``cliff_age`` is the age beyond
    which degradation accelerates (a simple piecewise model).
    """

    base_pace: float
    deg: float
    cliff_age: int
    cliff_multiplier: float = 2.5


# Reference tyre models — softer = faster but degrades quicker.
DEFAULT_TYRE_MODELS: Dict[Compound, TyreModel] = {
    Compound.SOFT: TyreModel(base_pace=90.0, deg=0.09, cliff_age=16),
    Compound.MEDIUM: TyreModel(base_pace=90.8, deg=0.055, cliff_age=26),
    Compound.HARD: TyreModel(base_pace=91.6, deg=0.035, cliff_age=40),
    Compound.INTERMEDIATE: TyreModel(base_pace=98.0, deg=0.06, cliff_age=30),
    Compound.WET: TyreModel(base_pace=105.0, deg=0.05, cliff_age=35),
}


@dataclass
class RaceProblem:
    """
    A concrete race-strategy search problem instance.

    Parameters
    ----------
    total_laps:
        Race distance in laps.
    start_compound:
        Compound fitted at the start (lap 0).
    pit_loss:
        Time lost for a pit stop (pit-lane delta + stationary time), seconds.
    track_temp:
        Track temperature, °C — scales degradation.
    is_wet:
        Whether the race is wet (relaxes the two-dry-compound rule and makes
        wet compounds the sensible choice).
    fuel_start_kg / fuel_per_lap:
        Fuel model; fuel burns linearly and lightens the car (each kg costs
        ``fuel_time_penalty`` seconds per lap).
    allowed_compounds:
        The compounds the search may fit (defaults to all dry, or all wet).
    tyre_models:
        Per-compound pace/deg parameters (defaults to :data:`DEFAULT_TYRE_MODELS`).
    max_stops:
        Upper bound on pit stops (bounds the branching / search depth).
    """

    total_laps: int
    start_compound: Compound
    pit_loss: float = 22.0
    track_temp: float = 35.0
    is_wet: bool = False
    fuel_start_kg: float = 100.0
    fuel_per_lap: float = 1.8
    fuel_time_penalty: float = 0.032  # s per kg per lap
    allowed_compounds: Tuple[Compound, ...] = ()
    tyre_models: Dict[Compound, TyreModel] = field(default_factory=lambda: dict(DEFAULT_TYRE_MODELS))
    max_stops: int = 3

    def __post_init__(self) -> None:
        if not self.allowed_compounds:
            pool = WET_COMPOUNDS if self.is_wet else DRY_COMPOUNDS
            # Preserve a deterministic order for reproducible search.
            order = [Compound.SOFT, Compound.MEDIUM, Compound.HARD,
                     Compound.INTERMEDIATE, Compound.WET]
            self.allowed_compounds = tuple(c for c in order if c in pool)

    # ------------------------------------------------------------------ #
    # Search-problem interface
    # ------------------------------------------------------------------ #
    def initial_state(self) -> RaceState:
        return RaceState(
            lap=0,
            compound=self.start_compound,
            tyre_age=0,
            stops_made=0,
            fuel_kg=self.fuel_start_kg,
            compounds_used=frozenset({self.start_compound}),
        )

    def is_goal(self, state: RaceState) -> bool:
        if state.lap != self.total_laps:
            return False
        if self.is_wet:
            return True
        # Dry race: at least two distinct dry compounds must have been used.
        dry_used = {c for c in state.compounds_used if c in DRY_COMPOUNDS}
        return len(dry_used) >= 2

    def actions(self, state: RaceState) -> List[Action]:
        """Return the legal actions from ``state``."""
        if state.lap >= self.total_laps:
            return []
        acts: List[Action] = [Action(ActionType.RUN)]
        # Pitting is legal if we have stops left and are not on the final lap
        # (pitting on the last lap is never beneficial and is disallowed).
        if state.stops_made < self.max_stops and state.lap < self.total_laps - 1:
            for c in self.allowed_compounds:
                # No value in fitting the identical compound you already run.
                if c != state.compound:
                    acts.append(Action(ActionType.PIT, c))
        return acts

    def lap_time(self, compound: Compound, tyre_age: int, fuel_kg: float) -> float:
        """Estimated time (s) for one lap under the given conditions."""
        model = self.tyre_models[compound]
        # Base pace + linear degradation, with an accelerated 'cliff'.
        deg_laps = tyre_age
        penalty = model.deg * deg_laps
        if tyre_age > model.cliff_age:
            over = tyre_age - model.cliff_age
            penalty += model.deg * model.cliff_multiplier * over
        # Track-temperature scaling: hotter track amplifies degradation.
        temp_factor = 1.0 + max(0.0, (self.track_temp - 30.0)) * 0.01
        penalty *= temp_factor
        # Fuel effect: heavier car is slower.
        fuel_penalty = fuel_kg * self.fuel_time_penalty
        return model.base_pace + penalty + fuel_penalty

    def step_cost(self, state: RaceState, action: Action) -> float:
        """Cost (added race time, s) of taking ``action`` in ``state``."""
        if action.type is ActionType.RUN:
            return self.lap_time(state.compound, state.tyre_age, state.fuel_kg)
        # PIT: the out-lap is run on the *new* compound at age 0, plus pit loss.
        assert action.compound is not None
        return self.pit_loss + self.lap_time(action.compound, 0, state.fuel_kg)

    def result(self, state: RaceState, action: Action) -> RaceState:
        """Return the successor state from applying ``action``."""
        new_fuel = max(0.0, state.fuel_kg - self.fuel_per_lap)
        if action.type is ActionType.RUN:
            return replace(
                state,
                lap=state.lap + 1,
                tyre_age=state.tyre_age + 1,
                fuel_kg=new_fuel,
            )
        assert action.compound is not None
        return replace(
            state,
            lap=state.lap + 1,          # the out-lap counts as a completed lap
            compound=action.compound,
            tyre_age=1,                 # one lap already run on the new set
            stops_made=state.stops_made + 1,
            fuel_kg=new_fuel,
            compounds_used=state.compounds_used | {action.compound},
        )

    # ------------------------------------------------------------------ #
    # Heuristics (for informed search)
    # ------------------------------------------------------------------ #
    def fastest_possible_lap(self) -> float:
        """The theoretical minimum lap time over all allowed compounds at age 0
        and zero fuel — an admissible lower bound on any lap's cost."""
        return min(
            self.tyre_models[c].base_pace for c in self.allowed_compounds
        )

    def heuristic(self, state: RaceState) -> float:
        """
        Admissible heuristic: laps remaining × fastest possible lap time.

        This never overestimates the true remaining cost because no lap can be
        faster than the fastest fresh-tyre pace at zero fuel, and it ignores the
        (non-negative) mandatory pit loss that a dry race may still require.
        """
        remaining = self.total_laps - state.lap
        return remaining * self.fastest_possible_lap()
