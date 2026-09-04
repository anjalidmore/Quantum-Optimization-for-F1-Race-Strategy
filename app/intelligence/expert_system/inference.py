"""
f1es.inference
==============

The inference engine: forward chaining, backward chaining, and pluggable
conflict-resolution strategies.

Forward chaining
----------------
Data-driven. Starting from the GIVEN facts, repeatedly finds all satisfied
rules (the *conflict set*), resolves conflicts to pick a firing order, fires
the winning rule(s), and iterates until working memory reaches a fixed point
(no rule changes anything) or a safety iteration cap is hit. Every firing is
recorded so the run can be explained and audited.

Backward chaining
-----------------
Goal-driven. Given a target ``(key, value)`` goal, searches for a rule whose
action asserts it, then recursively tries to satisfy that rule's conditions —
either because they are already GIVEN facts or because another rule can derive
them. Returns a proof tree when the goal is provable.

Conflict resolution
-------------------
When several rules are simultaneously satisfied, a
:class:`ConflictResolution` strategy orders them. The default is
salience → specificity → recency (rule-id), which is deterministic and
matches classic production-system behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

from .rules_schema import Rule
from .working_memory import Provenance, WorkingMemory


class ConflictResolution(str, Enum):
    """Available conflict-resolution strategies."""

    SALIENCE = "salience"          # salience, then specificity, then id
    SPECIFICITY = "specificity"    # specificity, then salience, then id
    FIFO = "fifo"                  # first-declared wins


def _order_conflict_set(
    conflict_set: Sequence[Rule], strategy: ConflictResolution
) -> List[Rule]:
    """Return ``conflict_set`` ordered so index 0 is the winning rule."""
    if strategy is ConflictResolution.SALIENCE:
        return sorted(conflict_set,
                      key=lambda r: (-r.salience, -r.specificity, r.rule_id))
    if strategy is ConflictResolution.SPECIFICITY:
        return sorted(conflict_set,
                      key=lambda r: (-r.specificity, -r.salience, r.rule_id))
    # FIFO: preserve declaration order (already the natural list order).
    return list(conflict_set)


@dataclass
class FiringRecord:
    """A record of a single rule firing during forward chaining."""

    iteration: int
    rule_id: str
    rule_name: str
    matched_conditions: List[str]
    asserted: Dict[str, object]


@dataclass
class ForwardResult:
    """The outcome of a forward-chaining run."""

    working_memory: WorkingMemory
    firings: List[FiringRecord] = field(default_factory=list)
    iterations: int = 0
    reached_fixed_point: bool = True

    @property
    def conclusions(self) -> Dict[str, object]:
        """All derived decision facts, keyed by fact key."""
        wm = self.working_memory
        return {k: wm.get(k) for k in wm.derived_keys()}


class InferenceEngine:
    """
    A production-rule inference engine over a fixed rule base.

    Parameters
    ----------
    rules:
        The rule base to reason with.
    strategy:
        Conflict-resolution strategy for forward chaining.
    max_iterations:
        Safety cap on forward-chaining iterations (guards against cycles).
    fire_all_in_iteration:
        If True, every rule in the (ordered) conflict set that still changes
        working memory fires within an iteration (classic "fire all" agenda).
        If False, only the single highest-priority rule fires per iteration.
    """

    def __init__(
        self,
        rules: Sequence[Rule],
        strategy: ConflictResolution = ConflictResolution.SALIENCE,
        max_iterations: int = 200,
        fire_all_in_iteration: bool = True,
    ) -> None:
        self.rules: List[Rule] = list(rules)
        self.strategy = strategy
        self.max_iterations = max_iterations
        self.fire_all_in_iteration = fire_all_in_iteration
        self._by_id: Dict[str, Rule] = {r.rule_id: r for r in self.rules}

    # ------------------------------------------------------------------ #
    # Forward chaining
    # ------------------------------------------------------------------ #
    def forward_chain(self, inputs: Mapping[str, object]) -> ForwardResult:
        """Run data-driven forward chaining from ``inputs``."""
        wm = WorkingMemory.from_inputs(inputs)
        result = ForwardResult(working_memory=wm)
        # Track (rule_id -> value-signature) already fired to avoid re-firing an
        # identical conclusion, which would otherwise stall the fixed point.
        fired_signatures: Set[Tuple[str, Tuple]] = set()

        for iteration in range(1, self.max_iterations + 1):
            facts = wm.as_mapping()
            conflict_set = [r for r in self.rules if r.is_satisfied(facts)]
            ordered = _order_conflict_set(conflict_set, self.strategy)

            changed_this_iter = False
            for r in ordered:
                signature = (r.rule_id, tuple(sorted(r.fired_effect().items(),
                                                     key=lambda kv: kv[0])))
                if signature in fired_signatures:
                    continue
                # Attempt to assert this rule's actions.
                asserted: Dict[str, object] = {}
                for a in r.actions:
                    if a.key == "notes":
                        # Notes accumulate rather than overwrite.
                        existing = wm.get("notes")
                        new_val = (f"{existing} | {a.value}"
                                   if existing else a.value)
                        if wm.assert_fact("notes", new_val, Provenance.DERIVED, r.rule_id):
                            asserted["notes"] = a.value
                    else:
                        if wm.assert_fact(a.key, a.value, Provenance.DERIVED, r.rule_id):
                            asserted[a.key] = a.value
                if asserted:
                    changed_this_iter = True
                    fired_signatures.add(signature)
                    result.firings.append(FiringRecord(
                        iteration=iteration,
                        rule_id=r.rule_id,
                        rule_name=r.name,
                        matched_conditions=[c.describe() for c in r.conditions],
                        asserted=asserted,
                    ))
                    if not self.fire_all_in_iteration:
                        break

            result.iterations = iteration
            if not changed_this_iter:
                result.reached_fixed_point = True
                break
        else:
            result.reached_fixed_point = False

        return result

    # ------------------------------------------------------------------ #
    # Backward chaining
    # ------------------------------------------------------------------ #
    def backward_chain(
        self,
        goal_key: str,
        goal_value: object,
        inputs: Mapping[str, object],
    ) -> "ProofNode":
        """
        Attempt to prove ``goal_key == goal_value`` from ``inputs``.

        Returns a :class:`ProofNode` whose ``proven`` flag indicates success and
        whose ``children`` reconstruct the supporting sub-goals.
        """
        facts = dict(inputs)
        return self._prove(goal_key, goal_value, facts, visited=set())

    def _prove(
        self,
        key: str,
        value: object,
        facts: Dict[str, object],
        visited: Set[str],
    ) -> "ProofNode":
        goal = f"{key} == {value!r}"
        # 1. Already a known fact?
        if facts.get(key) == value:
            return ProofNode(goal, proven=True, via="known fact", children=[])

        # Guard against infinite recursion on cyclic rule dependencies.
        if goal in visited:
            return ProofNode(goal, proven=False, via="cycle detected", children=[])
        visited = visited | {goal}

        # 2. Find rules whose action asserts (key, value).
        candidates = [
            r for r in self.rules
            if any(a.key == key and a.value == value for a in r.actions)
        ]
        for r in candidates:
            child_nodes: List[ProofNode] = []
            all_ok = True
            for c in r.conditions:
                # A condition is satisfiable if it already holds, or if some
                # rule can derive a fact making it hold. We only recurse on the
                # simple equality case; other operators are checked against
                # facts directly (closed-world).
                if c.evaluate(facts):
                    child_nodes.append(
                        ProofNode(c.describe(), proven=True, via="given", children=[]))
                    continue
                # Try to derive the exact value the condition needs (EQ only).
                from .rules_schema import Operator as _Op
                if c.operator is _Op.EQ:
                    sub = self._prove(c.key, c.value, facts, visited)
                    child_nodes.append(sub)
                    if not sub.proven:
                        all_ok = False
                        break
                else:
                    child_nodes.append(
                        ProofNode(c.describe(), proven=False,
                                  via="unprovable non-equality condition", children=[]))
                    all_ok = False
                    break
            if all_ok:
                return ProofNode(goal, proven=True, via=f"rule {r.rule_id}",
                                 children=child_nodes)

        return ProofNode(goal, proven=False, via="no supporting rule", children=[])

    def rule(self, rule_id: str) -> Optional[Rule]:
        """Look up a rule by id."""
        return self._by_id.get(rule_id)


@dataclass
class ProofNode:
    """A node in a backward-chaining proof tree."""

    goal: str
    proven: bool
    via: str
    children: List["ProofNode"]

    def render(self, indent: int = 0) -> str:
        """Return a pretty indented text rendering of the proof tree."""
        mark = "✓" if self.proven else "✗"
        line = f"{'  ' * indent}{mark} {self.goal}  [{self.via}]"
        lines = [line]
        for child in self.children:
            lines.append(child.render(indent + 1))
        return "\n".join(lines)
