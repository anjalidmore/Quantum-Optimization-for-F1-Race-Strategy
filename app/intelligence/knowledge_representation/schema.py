"""
f1kr.schema
===========

Single source of truth for the Formula 1 knowledge-representation domain.

This module declares the domain model *declaratively* so that every downstream
artefact -- the ontology (OWL), the knowledge graph (NetworkX/RDF), the
validation report and the generated documentation tables -- is produced from
one consistent definition. Nothing in this project hard-codes an entity or a
relationship twice.

The model covers the entities and relationships requested for Phase 1 / Task 1,
spanning both the historical Kaggle dataset (Ergast-derived) and the FastF1
live-timing / telemetry domain.

Design notes
------------
* Entities are grouped into semantic categories to support ontology class
  hierarchies (e.g. all tyre-related concepts descend from ``TyreConcept``).
* Attributes carry a primitive data-type and a human-readable description so we
  can emit both an attribute table and OWL data properties.
* Relationships carry a domain, a range and cardinality metadata so we can emit
  OWL object properties with proper ``rdfs:domain`` / ``rdfs:range``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple


class DataType(str, Enum):
    """Primitive datatypes used for entity attributes (maps to XSD in OWL)."""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATETIME = "datetime"


class Cardinality(str, Enum):
    """Relationship cardinality between a domain and range entity."""

    ONE_TO_ONE = "1..1"
    ONE_TO_MANY = "1..*"
    MANY_TO_ONE = "*..1"
    MANY_TO_MANY = "*..*"


@dataclass(frozen=True)
class Attribute:
    """A single data property belonging to an entity."""

    name: str
    dtype: DataType
    description: str
    unit: str = ""


@dataclass(frozen=True)
class Entity:
    """A domain concept (OWL class)."""

    name: str
    category: str
    description: str
    source: str  # "kaggle", "fastf1", or "both"
    parent: str = ""  # name of parent entity for the class hierarchy, "" if a root
    attributes: Tuple[Attribute, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Relationship:
    """A directed, named relationship (OWL object property)."""

    name: str
    domain: str  # subject entity name
    range: str   # object entity name
    cardinality: Cardinality
    description: str


# ---------------------------------------------------------------------------
# ENTITIES
# ---------------------------------------------------------------------------
# The category strings double as parent classes in the ontology when an entity
# has no explicit ``parent``. Roots of each category are created automatically.

_A = Attribute  # local alias for compactness

ENTITIES: Tuple[Entity, ...] = (
    # ---- Competition structure -------------------------------------------
    Entity(
        "Season", "Competition", "A Formula One championship year.", "kaggle",
        attributes=(
            _A("year", DataType.INTEGER, "Calendar year of the championship."),
            _A("url", DataType.STRING, "Reference URL for the season."),
        ),
    ),
    Entity(
        "Race", "Competition", "A single championship round / event weekend.", "both",
        attributes=(
            _A("round", DataType.INTEGER, "Round number within the season."),
            _A("name", DataType.STRING, "Official race name."),
            _A("date", DataType.DATETIME, "Race date."),
        ),
    ),
    Entity(
        "GrandPrix", "Competition",
        "The recurring named event (e.g. 'Monaco Grand Prix') across seasons.",
        "kaggle",
        attributes=(_A("name", DataType.STRING, "Grand Prix name."),),
    ),
    Entity(
        "Circuit", "Competition", "A racing venue / track.", "both",
        attributes=(
            _A("name", DataType.STRING, "Circuit name."),
            _A("location", DataType.STRING, "City / locality."),
            _A("country", DataType.STRING, "Country."),
            _A("length_km", DataType.FLOAT, "Track length.", "km"),
            _A("lat", DataType.FLOAT, "Latitude.", "deg"),
            _A("lng", DataType.FLOAT, "Longitude.", "deg"),
        ),
    ),
    # ---- Competitors ------------------------------------------------------
    Entity(
        "Driver", "Competitor", "A racing driver.", "both",
        attributes=(
            _A("driver_ref", DataType.STRING, "Stable identifier / reference."),
            _A("code", DataType.STRING, "Three-letter driver code (e.g. VER)."),
            _A("number", DataType.INTEGER, "Permanent car number."),
            _A("forename", DataType.STRING, "First name."),
            _A("surname", DataType.STRING, "Last name."),
            _A("nationality", DataType.STRING, "Nationality."),
            _A("dob", DataType.DATETIME, "Date of birth."),
        ),
    ),
    Entity(
        "Constructor", "Competitor", "A constructor / entrant team.", "both",
        attributes=(
            _A("constructor_ref", DataType.STRING, "Stable identifier."),
            _A("name", DataType.STRING, "Constructor name."),
            _A("nationality", DataType.STRING, "Nationality."),
        ),
    ),
    Entity(
        "Team", "Competitor",
        "The operational team entity that fields cars for a constructor.",
        "fastf1",
        attributes=(_A("name", DataType.STRING, "Team name as reported by timing."),),
    ),
    Entity(
        "Car", "Competitor", "The chassis + power unit fielded by a team.", "fastf1",
        attributes=(
            _A("chassis", DataType.STRING, "Chassis designation."),
            _A("power_unit", DataType.STRING, "Power unit manufacturer."),
        ),
    ),
    # ---- Session hierarchy ------------------------------------------------
    Entity(
        "Session", "Session", "An on-track session within a race weekend.", "both",
        attributes=(
            _A("session_type", DataType.STRING, "FP/Q/S/R type code."),
            _A("start_time", DataType.DATETIME, "Session start."),
        ),
    ),
    Entity("PracticeSession", "Session", "A free-practice session.", "fastf1", parent="Session"),
    Entity("Qualifying", "Session", "A qualifying session.", "both", parent="Session",
           attributes=(
               _A("q1", DataType.FLOAT, "Q1 time.", "s"),
               _A("q2", DataType.FLOAT, "Q2 time.", "s"),
               _A("q3", DataType.FLOAT, "Q3 time.", "s"),
           )),
    Entity("Sprint", "Session", "A sprint race session.", "fastf1", parent="Session"),
    # ---- Lap & timing -----------------------------------------------------
    Entity(
        "Lap", "Timing", "A single completed lap by a driver in a session.", "both",
        attributes=(
            _A("lap_number", DataType.INTEGER, "Lap index within the session."),
            _A("lap_time", DataType.FLOAT, "Total lap time.", "s"),
            _A("is_personal_best", DataType.BOOLEAN, "Whether this is the driver PB."),
        ),
    ),
    Entity(
        "Sector", "Timing", "One of the three timing sectors of a lap.", "fastf1",
        attributes=(_A("sector_number", DataType.INTEGER, "Sector index 1-3."),),
    ),
    Entity("SectorTime", "Timing", "The measured time for a sector.", "fastf1",
           attributes=(_A("time", DataType.FLOAT, "Sector time.", "s"),)),
    Entity("LapTime", "Timing", "A recorded lap-time measurement.", "kaggle",
           attributes=(_A("milliseconds", DataType.INTEGER, "Lap time.", "ms"),)),
    Entity("FastestLap", "Timing", "The fastest lap of a driver in a race.", "kaggle",
           attributes=(_A("rank", DataType.INTEGER, "Rank of the fastest lap."),
                       _A("time", DataType.FLOAT, "Fastest lap time.", "s"))),
    Entity("Gap", "Timing", "Time gap to a reference car.", "fastf1",
           attributes=(_A("seconds", DataType.FLOAT, "Gap.", "s"),)),
    Entity("Interval", "Timing", "Interval to the car directly ahead.", "fastf1",
           attributes=(_A("seconds", DataType.FLOAT, "Interval.", "s"),)),
    Entity("Overtake", "Timing", "An on-track position change event.", "fastf1",
           attributes=(_A("lap", DataType.INTEGER, "Lap on which it occurred."),)),
    # ---- Pit & stint ------------------------------------------------------
    Entity(
        "PitStop", "PitStrategy", "A pit-lane stop performed by a driver.", "both",
        attributes=(
            _A("lap", DataType.INTEGER, "Lap on which the stop occurred."),
            _A("duration", DataType.FLOAT, "Stationary + pit-lane time.", "s"),
            _A("stop_number", DataType.INTEGER, "Ordinal stop for the driver."),
        ),
    ),
    Entity("PitWindow", "PitStrategy",
           "The lap range within which a pit stop is strategically optimal.", "fastf1",
           attributes=(_A("open_lap", DataType.INTEGER, "First viable lap."),
                       _A("close_lap", DataType.INTEGER, "Last viable lap."))),
    Entity("Stint", "PitStrategy",
           "A continuous run on one set of tyres between pit stops.", "fastf1",
           attributes=(_A("stint_number", DataType.INTEGER, "Ordinal stint."),
                       _A("start_lap", DataType.INTEGER, "First lap of the stint."),
                       _A("end_lap", DataType.INTEGER, "Last lap of the stint."))),
    # ---- Tyres ------------------------------------------------------------
    Entity("TyreConcept", "Tyre", "Abstract root for tyre-related concepts.", "fastf1"),
    Entity("TyreCompound", "Tyre", "A tyre compound (Soft/Medium/Hard/Inter/Wet).",
           "fastf1", parent="TyreConcept",
           attributes=(_A("compound", DataType.STRING, "Compound name."),
                       _A("colour", DataType.STRING, "Marking colour."))),
    Entity("TyreLife", "Tyre", "Accumulated age/wear of a tyre set.", "fastf1",
           parent="TyreConcept",
           attributes=(_A("laps", DataType.INTEGER, "Laps completed on the set."),
                       _A("wear_pct", DataType.FLOAT, "Estimated wear.", "%"))),
    # ---- Weather ----------------------------------------------------------
    Entity("Weather", "Weather", "Weather conditions during a session.", "fastf1"),
    Entity("TrackTemperature", "Weather", "Track surface temperature.", "fastf1",
           parent="Weather", attributes=(_A("value", DataType.FLOAT, "Temp.", "C"),)),
    Entity("AirTemperature", "Weather", "Ambient air temperature.", "fastf1",
           parent="Weather", attributes=(_A("value", DataType.FLOAT, "Temp.", "C"),)),
    Entity("Humidity", "Weather", "Relative humidity.", "fastf1",
           parent="Weather", attributes=(_A("value", DataType.FLOAT, "Humidity.", "%"),)),
    Entity("WindSpeed", "Weather", "Wind speed.", "fastf1",
           parent="Weather", attributes=(_A("value", DataType.FLOAT, "Speed.", "m/s"),)),
    Entity("RainProbability", "Weather", "Probability of rain.", "fastf1",
           parent="Weather", attributes=(_A("value", DataType.FLOAT, "Probability.", "%"),)),
    # ---- Car dynamics -----------------------------------------------------
    Entity("FuelLoad", "CarState", "Estimated fuel mass on board.", "fastf1",
           attributes=(_A("kg", DataType.FLOAT, "Fuel mass.", "kg"),)),
    Entity("EngineMode", "CarState", "Power-unit deployment mode.", "fastf1",
           attributes=(_A("mode", DataType.STRING, "Mode label."),)),
    Entity("ERS", "CarState", "Energy Recovery System state.", "fastf1",
           attributes=(_A("charge_pct", DataType.FLOAT, "State of charge.", "%"),)),
    Entity("DRS", "CarState", "Drag Reduction System state.", "fastf1",
           attributes=(_A("active", DataType.BOOLEAN, "Whether DRS is open."),)),
    # ---- Telemetry --------------------------------------------------------
    Entity("Telemetry", "Telemetry", "A telemetry sample stream for a lap.", "fastf1"),
    Entity("Speed", "Telemetry", "Instantaneous car speed.", "fastf1",
           parent="Telemetry", attributes=(_A("value", DataType.FLOAT, "Speed.", "km/h"),)),
    Entity("Throttle", "Telemetry", "Throttle application.", "fastf1",
           parent="Telemetry", attributes=(_A("value", DataType.FLOAT, "Throttle.", "%"),)),
    Entity("Brake", "Telemetry", "Brake application.", "fastf1",
           parent="Telemetry", attributes=(_A("value", DataType.BOOLEAN, "Brake on/off."),)),
    Entity("RPM", "Telemetry", "Engine revolutions per minute.", "fastf1",
           parent="Telemetry", attributes=(_A("value", DataType.INTEGER, "RPM."),)),
    Entity("Gear", "Telemetry", "Selected gear.", "fastf1",
           parent="Telemetry", attributes=(_A("value", DataType.INTEGER, "Gear index."),)),
    # ---- Track status & race control -------------------------------------
    Entity("TrackStatus", "RaceControl", "Overall track status flag state.", "fastf1"),
    Entity("SafetyCar", "RaceControl", "Full safety-car deployment.", "fastf1",
           parent="TrackStatus"),
    Entity("VirtualSafetyCar", "RaceControl", "Virtual safety-car period.", "fastf1",
           parent="TrackStatus"),
    Entity("YellowFlag", "RaceControl", "Yellow-flag caution.", "fastf1",
           parent="TrackStatus"),
    Entity("RedFlag", "RaceControl", "Session-stopping red flag.", "fastf1",
           parent="TrackStatus"),
    Entity("BlueFlag", "RaceControl", "Blue flag (yield to leaders).", "fastf1",
           parent="TrackStatus"),
    Entity("RaceControlMessage", "RaceControl", "An official race-control message.",
           "fastf1", attributes=(_A("message", DataType.STRING, "Message text."),
                                 _A("lap", DataType.INTEGER, "Lap of issue."))),
    Entity("Incident", "RaceControl", "An on-track incident under investigation.",
           "fastf1", attributes=(_A("description", DataType.STRING, "Incident detail."),)),
    Entity("Penalty", "RaceControl", "A sporting penalty applied to a driver.", "both",
           attributes=(_A("seconds", DataType.FLOAT, "Time penalty.", "s"),
                       _A("reason", DataType.STRING, "Reason."))),
    # ---- Results & standings ---------------------------------------------
    Entity("Result", "Result", "A driver's classified result in a race.", "kaggle",
           attributes=(_A("position", DataType.INTEGER, "Finishing position."),
                       _A("points", DataType.FLOAT, "Points scored."),
                       _A("status", DataType.STRING, "Finish status."))),
    Entity("Position", "Result", "A position/order value.", "both",
           attributes=(_A("value", DataType.INTEGER, "Position number."),)),
    Entity("Points", "Result", "Championship points value.", "kaggle",
           attributes=(_A("value", DataType.FLOAT, "Points."),)),
    Entity("Championship", "Result", "A championship classification context.", "kaggle"),
    Entity("DriverStanding", "Result", "A driver's championship standing snapshot.",
           "kaggle", parent="Championship",
           attributes=(_A("points", DataType.FLOAT, "Cumulative points."),
                       _A("position", DataType.INTEGER, "Standing position."),
                       _A("wins", DataType.INTEGER, "Wins to date."))),
    Entity("ConstructorStanding", "Result",
           "A constructor's championship standing snapshot.", "kaggle",
           parent="Championship",
           attributes=(_A("points", DataType.FLOAT, "Cumulative points."),
                       _A("position", DataType.INTEGER, "Standing position."),
                       _A("wins", DataType.INTEGER, "Wins to date."))),
    # ---- Decision / analytics layer --------------------------------------
    Entity("Strategy", "Decision",
           "A recommended race-strategy plan (pit/tyre/fuel decisions).", "derived",
           attributes=(_A("name", DataType.STRING, "Strategy label."),
                       _A("n_stops", DataType.INTEGER, "Planned pit stops."))),
    Entity("Prediction", "Decision", "A model prediction over race outcomes.", "derived",
           attributes=(_A("target", DataType.STRING, "Predicted quantity."),
                       _A("confidence", DataType.FLOAT, "Confidence.", "%"))),
    Entity("Recommendation", "Decision",
           "An actionable recommendation surfaced to the race engineer.", "derived",
           attributes=(_A("text", DataType.STRING, "Recommendation text."),)),
    Entity("Optimization", "Decision",
           "An optimisation result over the strategy search space.", "derived",
           attributes=(_A("objective", DataType.STRING, "Objective optimised."),
                       _A("value", DataType.FLOAT, "Objective value."))),
)


# ---------------------------------------------------------------------------
# RELATIONSHIPS
# ---------------------------------------------------------------------------
_C = Cardinality

RELATIONSHIPS: Tuple[Relationship, ...] = (
    Relationship("drives_for", "Driver", "Constructor", _C.MANY_TO_ONE,
                 "A driver competes on behalf of a constructor."),
    Relationship("owns", "Constructor", "Car", _C.ONE_TO_MANY,
                 "A constructor owns / fields cars."),
    Relationship("participates_in", "Driver", "Race", _C.MANY_TO_MANY,
                 "A driver takes part in a race."),
    Relationship("held_at", "Race", "Circuit", _C.MANY_TO_ONE,
                 "A race is held at a circuit."),
    Relationship("belongs_to_season", "Race", "Season", _C.MANY_TO_ONE,
                 "A race belongs to a season."),
    Relationship("instance_of_gp", "Race", "GrandPrix", _C.MANY_TO_ONE,
                 "A race is an instance of a recurring Grand Prix."),
    Relationship("contains_lap", "Race", "Lap", _C.ONE_TO_MANY,
                 "A race contains laps."),
    Relationship("lap_of_driver", "Lap", "Driver", _C.MANY_TO_ONE,
                 "A lap is driven by a driver."),
    Relationship("lap_in_session", "Lap", "Session", _C.MANY_TO_ONE,
                 "A lap belongs to a session."),
    Relationship("lap_has_telemetry", "Lap", "Telemetry", _C.ONE_TO_MANY,
                 "A lap carries telemetry samples."),
    Relationship("lap_has_sector", "Lap", "Sector", _C.ONE_TO_MANY,
                 "A lap is divided into sectors."),
    Relationship("sector_has_time", "Sector", "SectorTime", _C.ONE_TO_ONE,
                 "A sector has a measured sector time."),
    Relationship("performs_pitstop", "Driver", "PitStop", _C.ONE_TO_MANY,
                 "A driver performs pit stops."),
    Relationship("pitstop_changes_tyre", "PitStop", "TyreCompound", _C.MANY_TO_ONE,
                 "A pit stop changes to a tyre compound."),
    Relationship("pitstop_begins_stint", "PitStop", "Stint", _C.ONE_TO_ONE,
                 "A pit stop begins a new stint."),
    Relationship("stint_uses_compound", "Stint", "TyreCompound", _C.MANY_TO_ONE,
                 "A stint runs on a tyre compound."),
    Relationship("weather_affects_strategy", "Weather", "Strategy", _C.MANY_TO_MANY,
                 "Weather conditions influence strategy."),
    Relationship("safetycar_affects_pitwindow", "SafetyCar", "PitWindow", _C.MANY_TO_MANY,
                 "Safety-car deployment shifts the optimal pit window."),
    Relationship("trackstatus_affects_strategy", "TrackStatus", "Strategy", _C.MANY_TO_MANY,
                 "Track status influences strategy."),
    Relationship("telemetry_influences_prediction", "Telemetry", "Prediction",
                 _C.MANY_TO_MANY, "Telemetry features feed predictions."),
    Relationship("strategy_recommends_pitstop", "Strategy", "PitStop", _C.ONE_TO_MANY,
                 "A strategy recommends pit stops."),
    Relationship("strategy_recommends_tyre", "Strategy", "TyreCompound", _C.ONE_TO_MANY,
                 "A strategy recommends tyre compounds."),
    Relationship("strategy_predicts_position", "Strategy", "Position", _C.ONE_TO_ONE,
                 "A strategy predicts a finish position."),
    Relationship("prediction_generates_recommendation", "Prediction", "Recommendation",
                 _C.ONE_TO_MANY, "A prediction yields recommendations."),
    Relationship("driver_has_result", "Driver", "Result", _C.ONE_TO_MANY,
                 "A driver has classified results."),
    Relationship("result_in_race", "Result", "Race", _C.MANY_TO_ONE,
                 "A result is recorded for a race."),
    Relationship("driver_has_standing", "Driver", "DriverStanding", _C.ONE_TO_MANY,
                 "A driver has championship standings."),
    Relationship("constructor_has_standing", "Constructor", "ConstructorStanding",
                 _C.ONE_TO_MANY, "A constructor has championship standings."),
    Relationship("optimization_yields_strategy", "Optimization", "Strategy", _C.ONE_TO_MANY,
                 "Optimisation produces candidate strategies."),
)


# ---------------------------------------------------------------------------
# Derived indices & helpers
# ---------------------------------------------------------------------------

def entities_by_name() -> Dict[str, Entity]:
    """Return a name -> Entity lookup."""
    return {e.name: e for e in ENTITIES}


def categories() -> List[str]:
    """Return the sorted list of distinct entity categories."""
    return sorted({e.category for e in ENTITIES})


def relationship_names() -> List[str]:
    """Return the list of relationship (object-property) names."""
    return [r.name for r in RELATIONSHIPS]
