"""
f1es.rule_base
==============

The curated Formula 1 strategy rule base.

Rather than padding to an arbitrary count with near-duplicate rules, this module
provides a **substantial, non-redundant core** of expert production rules,
organised by strategic domain and authored declaratively against the fact-key
registry in :mod:`f1es.working_memory`.

Domains covered
---------------
* Tyre selection & compound management
* Pit-stop timing & pit-window logic
* Weather response (dry / damp / wet / changeover)
* Safety-car & VSC opportunism
* Tyre degradation & thermal management
* Fuel & energy management
* Track-position tactics (undercut / overcut / defend)
* Overall strategy shape (1-stop / 2-stop) & risk assessment

Each rule carries a stable id, a category, a salience (priority) and a
human-readable ``description`` used by the explanation subsystem. Salience
conventions:

    100  safety-critical / weather-critical overrides
     80  safety-car & VSC opportunities
     60  pit-window & degradation decisions
     40  tyre compound selection
     20  tactical (undercut/overcut/defend)
      0  advisory / informational
"""

from __future__ import annotations

from typing import List

from .rules_schema import Connective, Rule, act, cond, rule
from .rules_schema import Operator as Op

# Local aliases to keep rule definitions readable.
_ALL = Connective.ALL
_ANY = Connective.ANY


def build_rule_base() -> List[Rule]:
    """Return the full curated list of :class:`Rule` objects."""
    rules: List[Rule] = []

    # ===================================================================== #
    # 1. WEATHER-CRITICAL OVERRIDES (salience 100)
    # ===================================================================== #
    rules += [
        rule("R-WX-001", "Heavy rain forecast -> wet tyres", "weather",
             [cond("rain_probability", Op.GT, 70)],
             [act("recommended_tyre", "INTERMEDIATE"),
              act("risk_level", "high"),
              act("notes", "High rain probability: switch to intermediates.")],
             salience=100,
             description="If rain probability exceeds 70%, intermediates are the "
                         "safe baseline; grip on slicks collapses once the track wets."),
        rule("R-WX-002", "Track already wet -> intermediates minimum", "weather",
             [cond("track_wet", Op.IS_TRUE), cond("rain_probability", Op.LE, 90)],
             [act("recommended_tyre", "INTERMEDIATE"),
              act("push_advice", "conserve")],
             salience=100,
             description="A wet track with moderate rain calls for intermediates "
                         "and reduced pace until conditions clarify."),
        rule("R-WX-003", "Standing water / extreme -> full wets", "weather",
             [cond("weather_severity", Op.EQ, "extreme")],
             [act("recommended_tyre", "WET"),
              act("risk_level", "high"),
              act("push_advice", "conserve"),
              act("notes", "Extreme conditions: full wet tyres, expect SC/red flag.")],
             salience=100,
             description="Standing water demands full wet tyres for aquaplaning "
                         "resistance; expect race-control intervention."),
        rule("R-WX-004", "Drying track after rain -> prepare slick changeover", "weather",
             [cond("current_compound", Op.IN, ("INTERMEDIATE", "WET")),
              cond("track_wet", Op.IS_FALSE),
              cond("rain_probability", Op.LT, 30)],
             [act("pit_decision", "PIT_SOON"),
              act("recommended_tyre", "SOFT"),
              act("notes", "Track drying: crossover to slicks approaching.")],
             salience=100,
             description="On a drying line, the intermediate 'cliff' arrives fast; "
                         "plan the crossover to slicks before losing seconds/lap."),
    ]

    # ===================================================================== #
    # 2. SAFETY CAR & VSC OPPORTUNISM (salience 80)
    # ===================================================================== #
    rules += [
        rule("R-SC-001", "Safety car deployed + worn tyres -> pit now", "safety_car",
             [cond("track_status", Op.EQ, "SC"), cond("tyre_wear", Op.GE, 40)],
             [act("pit_decision", "PIT_NOW"),
              act("notes", "SC out: cheap pit stop, pit immediately.")],
             salience=80,
             description="Under a safety car the pit-loss shrinks dramatically; "
                         "stopping with worn tyres is near free time."),
        rule("R-SC-002", "VSC + in pit window -> pit now", "safety_car",
             [cond("track_status", Op.EQ, "VSC"), cond("in_pit_window", Op.IS_TRUE)],
             [act("pit_decision", "PIT_NOW"),
              act("notes", "VSC reduces pit loss: take the stop within the window.")],
             salience=80,
             description="A virtual safety car slows the whole field, cutting the "
                         "relative time lost in the pit lane."),
        rule("R-SC-003", "High SC probability + marginal tyres -> delay pit", "safety_car",
             [cond("safety_car_probability", Op.GE, 60),
              cond("tyre_wear", Op.LT, 55),
              cond("track_status", Op.EQ, "GREEN")],
             [act("pit_decision", "DELAY_PIT"),
              act("notes", "Bank on an imminent SC for a cheaper stop.")],
             salience=80,
             description="When a safety car looks likely and tyres are not yet "
                         "critical, delaying can convert a full-price stop into a "
                         "discounted one."),
        rule("R-SC-004", "SC in final laps + leading -> stay out to keep track position",
             "safety_car",
             [cond("track_status", Op.EQ, "SC"),
              cond("laps_remaining", Op.LE, 8),
              cond("current_position", Op.LE, 3)],
             [act("pit_decision", "STAY_OUT"),
              act("defend_advice", "control_restart"),
              act("notes", "Track position outweighs tyre delta this late.")],
             salience=80,
             description="Late in a race, losing track position to a pit stop is "
                         "rarely recoverable; hold position and manage the restart."),
    ]

    # ===================================================================== #
    # 3. PIT-WINDOW & DEGRADATION (salience 60)
    # ===================================================================== #
    rules += [
        rule("R-PIT-001", "Critical tyre wear -> pit now", "pit",
             [cond("tyre_wear", Op.GE, 80)],
             [act("pit_decision", "PIT_NOW"),
              act("risk_level", "high")],
             salience=60,
             description="Beyond ~80% wear, lap-time loss and failure risk rise "
                         "sharply; stop regardless of window."),
        rule("R-PIT-002", "In window + undercut threat -> pit now", "pit",
             [cond("in_pit_window", Op.IS_TRUE), cond("undercut_threat", Op.IS_TRUE)],
             [act("pit_decision", "PIT_NOW"),
              act("notes", "Cover the undercut before the rival gains free air.")],
             salience=60,
             description="Reacting to an undercut requires pitting on the same lap "
                         "or the fresh-tyre advantage flips positions."),
        rule("R-PIT-003", "Overcut opportunity + clear air -> extend stint", "pit",
             [cond("overcut_opportunity", Op.IS_TRUE), cond("gap_ahead", Op.GT, 2.0),
              cond("tyre_wear", Op.LT, 65)],
             [act("pit_decision", "STAY_OUT"),
              act("push_advice", "push"),
              act("notes", "Overcut: push in clear air, pit later than rival.")],
             salience=60,
             description="With serviceable tyres and clear air, staying out and "
                         "pushing can leapfrog a rival who stops early."),
        rule("R-PIT-004", "Approaching window edge -> prime pit crew", "pit",
             [cond("in_pit_window", Op.IS_TRUE), cond("tyre_wear", Op.GE, 55),
              cond("tyre_wear", Op.LT, 80)],
             [act("pit_decision", "PIT_SOON"),
              act("notes", "Within window and wearing: prepare to stop.")],
             salience=60,
             description="Signals the crew to be ready without committing, keeping "
                         "flexibility for a safety car."),
        rule("R-DEG-001", "High track temp -> increased degradation", "degradation",
             [cond("track_temperature", Op.GT, 40)],
             [act("tyre_deg_adjustment", "increase"),
              act("notes", "Hot track: expect elevated thermal degradation.")],
             salience=60,
             description="Track temperatures above ~40°C accelerate thermal "
                         "degradation, shortening viable stint length."),
        rule("R-DEG-002", "Cool track -> reduced degradation, extend stint",
             "degradation",
             [cond("track_temperature", Op.LT, 25), cond("tyre_wear", Op.LT, 60)],
             [act("tyre_deg_adjustment", "decrease"),
              act("pit_decision", "STAY_OUT"),
              act("notes", "Cool track favours longer stints.")],
             salience=60,
             description="Lower surface temperatures reduce degradation, so a "
                         "longer stint (fewer stops) becomes viable."),
        rule("R-DEG-003", "High graining risk + high wear -> pit soon", "degradation",
             [cond("graining_risk", Op.EQ, "high"), cond("tyre_wear", Op.GE, 50)],
             [act("pit_decision", "PIT_SOON"),
              act("push_advice", "conserve"),
              act("notes", "Manage graining: reduce sliding, plan an earlier stop.")],
             salience=60,
             description="Graining scrubs grip nonlinearly; conserving and stopping "
                         "earlier avoids a lap-time collapse."),
    ]

    # ===================================================================== #
    # 4. TYRE COMPOUND SELECTION (salience 40)
    # ===================================================================== #
    rules += [
        rule("R-TYRE-001", "Soft worn -> step to medium", "tyre",
             [cond("current_compound", Op.EQ, "SOFT"), cond("tyre_wear", Op.GE, 60)],
             [act("recommended_tyre", "MEDIUM")],
             salience=40,
             description="When softs degrade, mediums trade a little peak grip for "
                         "durability over the next stint."),
        rule("R-TYRE-002", "Long final stint -> hard compound", "tyre",
             [cond("laps_remaining", Op.GT, 30), cond("track_wet", Op.IS_FALSE)],
             [act("recommended_tyre", "HARD")],
             salience=40,
             description="A long run to the flag favours the hard compound's "
                         "durability, avoiding a further stop."),
        rule("R-TYRE-003", "Short final stint + dry -> soft for pace", "tyre",
             [cond("laps_remaining", Op.LE, 15), cond("track_wet", Op.IS_FALSE),
              cond("tyre_wear", Op.GE, 40)],
             [act("recommended_tyre", "SOFT"),
              act("push_advice", "push")],
             salience=40,
             description="For a short sprint to the flag, the soft's peak grip "
                         "outweighs its lower durability."),
        rule("R-TYRE-004", "Damp but not wet -> intermediates over slicks", "tyre",
             [cond("weather_severity", Op.EQ, "damp")],
             [act("recommended_tyre", "INTERMEDIATE"),
              act("push_advice", "conserve")],
             salience=40,
             description="A damp track lacks the grip for slicks but not the water "
                         "for full wets; intermediates bridge the gap."),
        rule("R-TYRE-005", "Medium mid-race baseline in the heat", "tyre",
             [cond("current_compound", Op.EQ, "SOFT"),
              cond("track_temperature", Op.GT, 45),
              cond("laps_remaining", Op.GT, 20)],
             [act("recommended_tyre", "MEDIUM"),
              act("notes", "Very hot track: avoid softs for a long middle stint.")],
             salience=40,
             description="In extreme heat, softs overheat quickly; mediums give a "
                         "more stable operating window for a long stint."),
    ]

    # ===================================================================== #
    # 5. STRATEGY SHAPE & CIRCUIT-SPECIFIC (salience 30-60)
    # ===================================================================== #
    rules += [
        rule("R-STRAT-001", "Monaco + front row -> one stop", "strategy",
             [cond("circuit", Op.EQ, "Monaco"), cond("grid_position", Op.LE, 3)],
             [act("strategy_stops", 1),
              act("notes", "Monaco: track position is king, minimise stops.")],
             salience=60,
             description="Overtaking at Monaco is exceptionally hard, so protecting "
                         "track position with a single stop is standard."),
        rule("R-STRAT-002", "High overtaking difficulty -> minimise stops", "strategy",
             [cond("overtaking_difficulty", Op.EQ, "high")],
             [act("strategy_stops", 1),
              act("defend_advice", "protect_position")],
             salience=40,
             description="Where passing is hard, every pit stop risks a position "
                         "that cannot be won back on track."),
        rule("R-STRAT-003", "High deg circuit + low pit loss -> two stops", "strategy",
             [cond("track_temperature", Op.GT, 40), cond("pit_loss", Op.LT, 20),
              cond("overtaking_difficulty", Op.NE, "high")],
             [act("strategy_stops", 2),
              act("push_advice", "push"),
              act("notes", "Cheap stops + high deg reward an aggressive 2-stop.")],
             salience=40,
             description="When pit loss is small and degradation high, a two-stop "
                         "on fresher tyres beats nursing one set."),
        rule("R-STRAT-004", "Low SC likelihood + long stint -> plan fixed strategy",
             "strategy",
             [cond("safety_car_likelihood", Op.EQ, "low"),
              cond("total_laps", Op.GT, 50)],
             [act("notes", "Low SC prior: commit to a planned strategy, fewer reactive stops.")],
             salience=30,
             description="With little chance of a race-altering SC, a pre-planned "
                         "strategy usually outperforms reactive gambling."),
    ]

    # ===================================================================== #
    # 6. FUEL & ENERGY MANAGEMENT (salience 30)
    # ===================================================================== #
    rules += [
        rule("R-FUEL-001", "Fuel short -> lift and coast", "fuel",
             [cond("fuel_margin", Op.LT, 0)],
             [act("fuel_advice", "lift_and_coast"),
              act("push_advice", "conserve"),
              act("risk_level", "medium"),
              act("notes", "Under fuel target: save via lift-and-coast zones.")],
             salience=30,
             description="A negative fuel margin means the car will not reach the "
                         "flag at full pace; lift-and-coast recovers margin."),
        rule("R-FUEL-002", "Comfortable fuel + soft tyres -> push", "fuel",
             [cond("fuel_margin", Op.GT, 1.5), cond("current_compound", Op.EQ, "SOFT")],
             [act("fuel_advice", "normal"),
              act("push_advice", "push")],
             salience=30,
             description="With fuel in hand and grippy tyres, converting the "
                         "surplus into pace is worthwhile."),
        rule("R-ERS-001", "Defending with DRS train -> deploy ERS on straights",
             "energy",
             [cond("gap_behind", Op.LT, 1.0), cond("drs_enabled", Op.IS_TRUE)],
             [act("engine_mode_advice", "deploy_on_straights"),
              act("defend_advice", "cover_inside_line"),
              act("notes", "Under DRS threat: deploy ERS to defend the straights.")],
             salience=30,
             description="When a rival is within DRS range, biasing ERS deployment "
                         "to the straights protects against a slipstream pass."),
    ]

    # ===================================================================== #
    # 7. TACTICAL TRACK-POSITION (salience 20)
    # ===================================================================== #
    rules += [
        rule("R-TAC-001", "Close behind + faster tyres -> attack", "tactics",
             [cond("gap_ahead", Op.LT, 1.0), cond("tyre_age_laps", Op.LT, 10),
              cond("drs_enabled", Op.IS_TRUE)],
             [act("push_advice", "push"),
              act("notes", "Fresh tyres + DRS: attack the car ahead.")],
             salience=20,
             description="A fresher-tyred car within DRS range should press the "
                         "advantage before the tyre delta erodes."),
        rule("R-TAC-002", "Big gap both ways -> manage tyres in clear air", "tactics",
             [cond("gap_ahead", Op.GT, 3.0), cond("gap_behind", Op.GT, 3.0)],
             [act("push_advice", "conserve"),
              act("notes", "Clear air: manage tyres, no need to push.")],
             salience=20,
             description="With no immediate threat or target, conserving tyres "
                         "extends strategic options later."),
        rule("R-TAC-003", "Under pressure + worn tyres -> defensive lines", "tactics",
             [cond("gap_behind", Op.LT, 1.5), cond("tyre_wear", Op.GE, 60)],
             [act("defend_advice", "defensive_lines"),
              act("push_advice", "conserve"),
              act("notes", "Worn tyres under pressure: defend, protect braking zones.")],
             salience=20,
             description="Worn tyres cannot match a fresher pursuer's pace, so "
                         "positioning and braking-zone control matter most."),
    ]

    # ===================================================================== #
    # 8. RISK / ADVISORY DEFAULTS (salience 0)
    # ===================================================================== #
    rules += [
        rule("R-RISK-001", "Multiple high-risk factors -> flag high risk", "risk",
             [cond("tyre_wear", Op.GE, 75), cond("rain_probability", Op.GT, 50)],
             [act("risk_level", "high"),
              act("notes", "Worn tyres and rain risk compound: high uncertainty.")],
             salience=0,
             description="Concurrent worn tyres and rain risk multiply strategic "
                         "uncertainty and warrant a conservative posture."),
        rule("R-RISK-002", "Stable dry conditions mid-race -> low risk baseline",
             "risk",
             [cond("weather_severity", Op.EQ, "dry"), cond("track_status", Op.EQ, "GREEN"),
              cond("tyre_wear", Op.LT, 50)],
             [act("risk_level", "low")],
             salience=0,
             description="Green-flag dry running on healthy tyres is the low-risk "
                         "baseline against which deviations are judged."),
    ]

    return rules


# Convenience: the built rule base as a module-level constant.
RULE_BASE: List[Rule] = build_rule_base()
