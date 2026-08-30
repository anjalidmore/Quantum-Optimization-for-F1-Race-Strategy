"""
f1es.rules_schema
=================

The data model for the Formula 1 rule-based expert system.

This module defines the *structured* representation of production rules used by
the inference engine. Rather than encoding rules as opaque strings, every rule
is a typed object:

    Rule := IF <all/any of Condition...> THEN <Action...>  [with metadata]

Design goals
------------
* **Declarative & serialisable.** Rules can be authored in Python (see
  ``f1es.rule_base``) or loaded from / dumped to JSON, and validated
  independently of the engine.
* **Typed conditions.** A :class:`Condition` compares a *fact key* against a
  value using a typed :class:`Operator`, so the engine never does string
  matching on hand-written predicates.
* **Explainable actions.** Each :class:`Action` records what it asserts into
  working memory, enabling the explanation subsystem to reconstruct *why* a
  conclusion was reached.
* **Conflict-resolution metadata.** ``salience`` (priority), ``specificity``
  (auto-derived from condition count) and rule ``category`` drive deterministic
  conflict resolution.

No machine learning and no quantum computing appear here — this is a classical
symbolic system.
"""

from __future__ import annotations

import operator as _op
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple


class Operator(str, Enum):
    """Comparison operators usable inside a :class:`Condition`."""

    EQ = "=="
    NE = "!="
    GT = ">"
    GE = ">="
    LT = "<"
    LE = "<="
    IN = "in"          # value is a container; fact must be a member
    NOT_IN = "not_in"
    CONTAINS = "contains"  # fact is a container; value must be a member
    IS_TRUE = "is_true"    # boolean fact is truthy (value ignored)
    IS_FALSE = "is_false"  # boolean fact is falsy (value ignored)


# Mapping from operator to a concrete comparison function ``f(fact, value)``.
_OPS: Dict[Operator, Callable[[Any, Any], bool]] = {
    Operator.EQ: _op.eq,
    Operator.NE: _op.ne,
    Operator.GT: _op.gt,
    Operator.GE: _op.ge,
    Operator.LT: _op.lt,
    Operator.LE: _op.le,
    Operator.IN: lambda fact, value: fact in value,
    Operator.NOT_IN: lambda fact, value: fact not in value,
    Operator.CONTAINS: lambda fact, value: value in fact,
    Operator.IS_TRUE: lambda fact, _value: bool(fact),
    Operator.IS_FALSE: lambda fact, _value: not bool(fact),
}


class Connective(str, Enum):
    """How the conditions of a rule are combined."""

    ALL = "all"  # logical AND (default)
    ANY = "any"  # logical OR


@dataclass(frozen=True)
class Condition:
    """
    A single antecedent test: ``fact[key] <operator> value``.

    Parameters
    ----------
    key:
        The working-memory fact key to test (e.g. ``"rain_probability"``).
    operator:
        The :class:`Operator` to apply.
    value:
        The right-hand-side literal (ignored for ``IS_TRUE`` / ``IS_FALSE``).
    """

    key: str
    operator: Operator
    value: Any = None

    def evaluate(self, facts: Mapping[str, Any]) -> bool:
        """
        Return True iff this condition holds given ``facts``.

        A missing key evaluates to False (closed-world assumption) except for
        ``IS_FALSE`` on an absent key, which is treated as False as well so that
        rules never fire on unknown data.
        """
        if self.key not in facts:
            return False
        fact = facts[self.key]
        if fact is None:
            return False
        try:
            return _OPS[self.operator](fact, self.value)
        except TypeError:
            # Incomparable types (e.g. str vs float) -> condition simply fails.
            return False

    def describe(self) -> str:
        """Human-readable rendering, used by the explanation subsystem."""
        if self.operator in (Operator.IS_TRUE, Operator.IS_FALSE):
            return f"{self.key} {self.operator.value}"
        return f"{self.key} {self.operator.value} {self.value!r}"


@dataclass(frozen=True)
class Action:
    """
    A consequent: assert ``fact[key] = value`` into working memory.

    ``key`` typically names a decision variable such as
    ``"recommended_tyre"`` or ``"pit_decision"``. The optional ``confidence``
    (0-1) lets downstream layers weight competing recommendations.
    """

    key: str
    value: Any
    confidence: float = 1.0

    def describe(self) -> str:
        conf = "" if self.confidence >= 1.0 else f" (confidence {self.confidence:.2f})"
        return f"set {self.key} = {self.value!r}{conf}"


@dataclass(frozen=True)
class Rule:
    """
    A production rule.

    Attributes
    ----------
    rule_id:
        Stable unique identifier (e.g. ``"R-TYRE-001"``).
    name:
        Short human-readable name.
    category:
        Grouping used for organisation and conflict resolution tie-breaks
        (e.g. ``"tyre"``, ``"pit"``, ``"weather"``, ``"safety_car"``).
    conditions:
        The antecedents.
    connective:
        Whether ``conditions`` are combined with ALL (AND) or ANY (OR).
    actions:
        The consequents asserted when the rule fires.
    salience:
        Priority for conflict resolution; higher fires first. Default 0.
    description:
        Free-text rationale, surfaced in explanations.
    """

    rule_id: str
    name: str
    category: str
    conditions: Tuple[Condition, ...]
    actions: Tuple[Action, ...]
    connective: Connective = Connective.ALL
    salience: int = 0
    description: str = ""

    @property
    def specificity(self) -> int:
        """Number of conditions — more specific rules win ties by default."""
        return len(self.conditions)

    def is_satisfied(self, facts: Mapping[str, Any]) -> bool:
        """Return True iff the rule's antecedent holds for ``facts``."""
        if not self.conditions:
            return False
        checker = all if self.connective is Connective.ALL else any
        return checker(c.evaluate(facts) for c in self.conditions)

    def fired_effect(self) -> Dict[str, Any]:
        """Return the mapping this rule would assert if it fired."""
        return {a.key: a.value for a in self.actions}

    def describe(self) -> str:
        """Render the whole rule as ``IF ... THEN ...`` text."""
        joiner = " AND " if self.connective is Connective.ALL else " OR "
        antecedent = joiner.join(c.describe() for c in self.conditions)
        consequent = "; ".join(a.describe() for a in self.actions)
        return f"IF {antecedent} THEN {consequent}"


# --------------------------------------------------------------------------- #
# Convenience constructors — make authoring rules terse and readable.
# --------------------------------------------------------------------------- #

def cond(key: str, operator: Operator, value: Any = None) -> Condition:
    """Shorthand for :class:`Condition`."""
    return Condition(key, operator, value)


def act(key: str, value: Any, confidence: float = 1.0) -> Action:
    """Shorthand for :class:`Action`."""
    return Action(key, value, confidence)


def rule(
    rule_id: str,
    name: str,
    category: str,
    conditions: Sequence[Condition],
    actions: Sequence[Action],
    *,
    connective: Connective = Connective.ALL,
    salience: int = 0,
    description: str = "",
) -> Rule:
    """Shorthand for :class:`Rule` that freezes the condition/action sequences."""
    return Rule(
        rule_id=rule_id,
        name=name,
        category=category,
        conditions=tuple(conditions),
        actions=tuple(actions),
        connective=connective,
        salience=salience,
        description=description,
    )
