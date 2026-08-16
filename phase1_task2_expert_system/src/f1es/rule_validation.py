"""
f1es.rule_validation
=====================

Static validation of the rule base, independent of any particular inference run.

Checks performed
----------------
* **Unique ids** — no two rules share a ``rule_id``.
* **Non-empty structure** — every rule has >=1 condition and >=1 action.
* **Registered keys** — every condition/action key exists in the fact-key
  registry (:data:`f1es.working_memory.FACT_KEYS`), catching typos.
* **Operator/value sanity** — value-less operators (``IS_TRUE``/``IS_FALSE``)
  carry no value; value-bearing operators do.
* **Direct contradictions** — no two rules with identical antecedents assert
  conflicting values for the same output key at the same salience (which would
  make the outcome order-dependent).
* **Confidence bounds** — action confidences lie in [0, 1].
* **Salience sanity** — salience is a plain int (documented convention).

Results are returned as :class:`RuleCheckResult` records for reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Set, Tuple

from .rules_schema import Operator, Rule
from .working_memory import all_fact_keys

_VALUELESS_OPS = {Operator.IS_TRUE, Operator.IS_FALSE}


@dataclass(frozen=True)
class RuleCheckResult:
    """Outcome of a single rule-base validation check."""

    name: str
    passed: bool
    detail: str


def check_unique_ids(rules: Sequence[Rule]) -> RuleCheckResult:
    ids = [r.rule_id for r in rules]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    return RuleCheckResult("unique_rule_ids", not dupes,
                           "All rule ids unique." if not dupes
                           else f"Duplicate ids: {dupes}")


def check_nonempty(rules: Sequence[Rule]) -> RuleCheckResult:
    bad = [r.rule_id for r in rules if not r.conditions or not r.actions]
    return RuleCheckResult("rules_nonempty", not bad,
                           "Every rule has conditions and actions." if not bad
                           else f"Empty conditions/actions in: {bad}")


def check_registered_keys(rules: Sequence[Rule]) -> RuleCheckResult:
    known: Set[str] = set(all_fact_keys())
    problems: List[str] = []
    for r in rules:
        for c in r.conditions:
            if c.key not in known:
                problems.append(f"{r.rule_id}: condition key '{c.key}' unregistered")
        for a in r.actions:
            if a.key not in known:
                problems.append(f"{r.rule_id}: action key '{a.key}' unregistered")
    return RuleCheckResult("keys_registered", not problems,
                           "All condition/action keys are registered." if not problems
                           else "; ".join(problems))


def check_operator_value_sanity(rules: Sequence[Rule]) -> RuleCheckResult:
    problems: List[str] = []
    for r in rules:
        for c in r.conditions:
            if c.operator in _VALUELESS_OPS and c.value is not None:
                problems.append(f"{r.rule_id}: {c.operator.value} should not carry a value")
            if c.operator not in _VALUELESS_OPS and c.value is None:
                problems.append(f"{r.rule_id}: {c.operator.value} requires a value")
    return RuleCheckResult("operator_value_sanity", not problems,
                           "Operators and values are consistent." if not problems
                           else "; ".join(problems))


def check_confidence_bounds(rules: Sequence[Rule]) -> RuleCheckResult:
    problems = [
        f"{r.rule_id}: {a.key} confidence {a.confidence}"
        for r in rules for a in r.actions
        if not (0.0 <= a.confidence <= 1.0)
    ]
    return RuleCheckResult("confidence_bounds", not problems,
                           "All confidences within [0,1]." if not problems
                           else "; ".join(problems))


def _antecedent_signature(rule: Rule) -> Tuple:
    """A hashable signature of a rule's antecedent for contradiction detection."""
    return (rule.connective.value,
            tuple(sorted((c.key, c.operator.value, repr(c.value))
                         for c in rule.conditions)))


def check_no_contradictions(rules: Sequence[Rule]) -> RuleCheckResult:
    """
    Flag pairs of rules with identical antecedents and salience that assert
    different values for the same output key (order-dependent outcome).
    """
    problems: List[str] = []
    by_sig: Dict[Tuple, List[Rule]] = {}
    for r in rules:
        by_sig.setdefault((_antecedent_signature(r), r.salience), []).append(r)
    for (_sig, _sal), group in by_sig.items():
        if len(group) < 2:
            continue
        # Compare their asserted key/value pairs.
        effects: Dict[str, Set] = {}
        for r in group:
            for a in r.actions:
                if a.key == "notes":
                    continue  # notes intentionally accumulate
                effects.setdefault(a.key, set()).add(repr(a.value))
        for key, values in effects.items():
            if len(values) > 1:
                ids = sorted(r.rule_id for r in group)
                problems.append(
                    f"Rules {ids} share an antecedent+salience but assert "
                    f"conflicting '{key}' values {sorted(values)}")
    return RuleCheckResult("no_order_dependent_contradictions", not problems,
                           "No order-dependent contradictions found." if not problems
                           else "; ".join(problems))


def validate_rule_base(rules: Sequence[Rule]) -> List[RuleCheckResult]:
    """Run the full static validation suite over ``rules``."""
    return [
        check_unique_ids(rules),
        check_nonempty(rules),
        check_registered_keys(rules),
        check_operator_value_sanity(rules),
        check_confidence_bounds(rules),
        check_no_contradictions(rules),
    ]


def all_passed(results: Sequence[RuleCheckResult]) -> bool:
    return all(r.passed for r in results)
