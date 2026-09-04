"""
Unit tests for the f1es rule-based expert system (Phase 1, Task 2).

Run with:  pytest -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from f1es import rule_validation as rv
from f1es.explanation import Explainer
from f1es.inference import ConflictResolution, InferenceEngine
from f1es.rule_base import build_rule_base
from f1es.rules_schema import (
    Action,
    Condition,
    Connective,
    Operator,
    Rule,
    act,
    cond,
    rule,
)
from f1es.working_memory import Provenance, WorkingMemory, all_fact_keys


# --------------------------------------------------------------------------- #
# Schema-level: conditions & operators
# --------------------------------------------------------------------------- #

def test_condition_numeric_operators():
    facts = {"tyre_wear": 82}
    assert Condition("tyre_wear", Operator.GE, 80).evaluate(facts)
    assert not Condition("tyre_wear", Operator.LT, 80).evaluate(facts)


def test_condition_membership_operators():
    facts = {"current_compound": "INTERMEDIATE"}
    assert Condition("current_compound", Operator.IN,
                     ("INTERMEDIATE", "WET")).evaluate(facts)
    assert Condition("current_compound", Operator.NOT_IN, ("SOFT",)).evaluate(facts)


def test_condition_boolean_operators():
    assert Condition("track_wet", Operator.IS_TRUE).evaluate({"track_wet": True})
    assert Condition("track_wet", Operator.IS_FALSE).evaluate({"track_wet": False})


def test_missing_key_is_false():
    assert not Condition("nope", Operator.GT, 1).evaluate({})


def test_incomparable_types_do_not_raise():
    # str vs float comparison must fail gracefully, not raise.
    assert not Condition("x", Operator.GT, 5).evaluate({"x": "abc"})


def test_rule_satisfaction_all_vs_any():
    r_all = rule("T1", "all", "t",
                 [cond("a", Operator.EQ, 1), cond("b", Operator.EQ, 2)],
                 [act("out", 1)], connective=Connective.ALL)
    r_any = rule("T2", "any", "t",
                 [cond("a", Operator.EQ, 1), cond("b", Operator.EQ, 99)],
                 [act("out", 1)], connective=Connective.ANY)
    facts = {"a": 1, "b": 2}
    assert r_all.is_satisfied(facts)
    assert r_any.is_satisfied(facts)
    assert not r_all.is_satisfied({"a": 1, "b": 0})


# --------------------------------------------------------------------------- #
# Working memory
# --------------------------------------------------------------------------- #

def test_working_memory_provenance_and_change_detection():
    wm = WorkingMemory.from_inputs({"a": 1})
    assert wm.record_for("a").provenance is Provenance.GIVEN
    assert wm.assert_fact("b", 2) is True          # new key -> changed
    assert wm.assert_fact("b", 2) is False         # same value -> no change
    assert wm.assert_fact("b", 3) is True          # new value -> changed
    assert wm.record_for("b").provenance is Provenance.DERIVED


# --------------------------------------------------------------------------- #
# Rule base & validation
# --------------------------------------------------------------------------- #

def test_rule_base_nonempty_and_multi_category():
    rules = build_rule_base()
    assert len(rules) >= 25
    assert len({r.category for r in rules}) >= 6


def test_rule_base_passes_static_validation():
    rules = build_rule_base()
    results = rv.validate_rule_base(rules)
    assert rv.all_passed(results), [
        (r.name, r.detail) for r in results if not r.passed
    ]


def test_all_rule_keys_are_registered():
    known = set(all_fact_keys())
    for r in build_rule_base():
        for c in r.conditions:
            assert c.key in known, f"{r.rule_id}: {c.key}"
        for a in r.actions:
            assert a.key in known, f"{r.rule_id}: {a.key}"


def test_validator_catches_unregistered_key():
    bad = [rule("X", "bad", "t",
                [cond("totally_unknown_key", Operator.EQ, 1)], [act("out", 1)])]
    results = rv.validate_rule_base(bad)
    assert any(not r.passed and r.name == "keys_registered" for r in results)


def test_validator_catches_contradiction():
    r1 = rule("C1", "c1", "t", [cond("recommended_tyre", Operator.EQ, "X")],
              [act("pit_decision", "PIT_NOW")], salience=5)
    r2 = rule("C2", "c2", "t", [cond("recommended_tyre", Operator.EQ, "X")],
              [act("pit_decision", "STAY_OUT")], salience=5)
    results = rv.validate_rule_base([r1, r2])
    assert any(not r.passed and r.name == "no_order_dependent_contradictions"
               for r in results)


# --------------------------------------------------------------------------- #
# Forward chaining
# --------------------------------------------------------------------------- #

def test_forward_chain_wet_weather():
    engine = InferenceEngine(build_rule_base())
    fr = engine.forward_chain({
        "rain_probability": 85, "track_wet": True, "current_compound": "SOFT",
        "weather_severity": "wet",
    })
    assert fr.conclusions.get("recommended_tyre") == "INTERMEDIATE"
    assert fr.reached_fixed_point


def test_forward_chain_safety_car_pits():
    engine = InferenceEngine(build_rule_base())
    fr = engine.forward_chain({"track_status": "SC", "tyre_wear": 60})
    assert fr.conclusions.get("pit_decision") == "PIT_NOW"


def test_forward_chain_reaches_fixed_point_and_terminates():
    engine = InferenceEngine(build_rule_base(), max_iterations=50)
    fr = engine.forward_chain({
        "track_temperature": 46, "in_pit_window": True, "undercut_threat": True,
        "pit_loss": 18, "overtaking_difficulty": "medium", "tyre_wear": 58,
        "current_compound": "SOFT", "weather_severity": "dry",
        "track_status": "GREEN", "laps_remaining": 25,
    })
    assert fr.reached_fixed_point
    assert fr.iterations <= 50


def test_salience_ordering_prioritises_critical_rules():
    engine = InferenceEngine(build_rule_base(),
                             strategy=ConflictResolution.SALIENCE)
    # Rain override (salience 100) must fire before generic tyre rules.
    fr = engine.forward_chain({
        "rain_probability": 90, "current_compound": "SOFT", "tyre_wear": 65,
        "laps_remaining": 40, "track_wet": True, "weather_severity": "wet",
    })
    first = fr.firings[0]
    fired_rule = next(r for r in build_rule_base() if r.rule_id == first.rule_id)
    assert fired_rule.salience >= 80


def test_empty_inputs_produce_no_firings():
    engine = InferenceEngine(build_rule_base())
    fr = engine.forward_chain({})
    assert fr.firings == []
    assert fr.conclusions == {}


# --------------------------------------------------------------------------- #
# Backward chaining
# --------------------------------------------------------------------------- #

def test_backward_chain_proves_reachable_goal():
    engine = InferenceEngine(build_rule_base())
    proof = engine.backward_chain(
        "pit_decision", "PIT_NOW",
        {"track_status": "SC", "tyre_wear": 60})
    assert proof.proven


def test_backward_chain_fails_unreachable_goal():
    engine = InferenceEngine(build_rule_base())
    proof = engine.backward_chain(
        "pit_decision", "PIT_NOW", {"weather_severity": "dry"})
    assert not proof.proven


# --------------------------------------------------------------------------- #
# Explanation
# --------------------------------------------------------------------------- #

def test_explanation_justifies_derived_fact():
    engine = InferenceEngine(build_rule_base())
    fr = engine.forward_chain({"track_status": "SC", "tyre_wear": 60})
    explainer = Explainer(fr, build_rule_base())
    j = explainer.justify("pit_decision")
    assert j is not None
    assert j.value == "PIT_NOW"
    assert j.rule_id is not None
    assert j.because  # at least one satisfied condition recorded


def test_explanation_narrative_and_audit_trail():
    engine = InferenceEngine(build_rule_base())
    fr = engine.forward_chain({"rain_probability": 85, "track_wet": True,
                               "weather_severity": "wet"})
    explainer = Explainer(fr, build_rule_base())
    assert "fired" in explainer.narrative()
    trail = explainer.audit_trail()
    assert any("input" in line for line in trail)
    assert any("rule" in line for line in trail)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
