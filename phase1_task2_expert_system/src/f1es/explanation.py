"""
f1es.explanation
================

The explanation subsystem. It turns an inference run into human-readable
justifications, answering the two classic expert-system questions:

* **HOW** was a conclusion reached? — a justification chain tracing a derived
  fact back through the rule that asserted it and the facts that satisfied that
  rule's conditions (recursively, down to GIVEN inputs).
* **WHY** did the system recommend this? — a natural-language summary of the
  firing sequence and the final decision set.

The explanations are reconstructed purely from the recorded
:class:`~f1es.inference.FiringRecord` log and working-memory provenance, so they
faithfully reflect what actually happened during inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .inference import ForwardResult
from .rules_schema import Rule
from .working_memory import Provenance


@dataclass
class Justification:
    """A HOW-explanation for a single derived fact."""

    fact: str
    value: object
    rule_id: Optional[str]
    rule_name: str
    because: List[str]  # the condition descriptions that were satisfied
    supporting_facts: Dict[str, object]  # the input facts that mattered


class Explainer:
    """Builds explanations from a :class:`ForwardResult` and the rule base."""

    def __init__(self, result: ForwardResult, rules: List[Rule]) -> None:
        self.result = result
        self._by_id: Dict[str, Rule] = {r.rule_id: r for r in rules}

    # ------------------------------------------------------------------ #
    # HOW: justify a specific derived fact
    # ------------------------------------------------------------------ #
    def justify(self, fact_key: str) -> Optional[Justification]:
        """Explain HOW ``fact_key`` obtained its current value."""
        wm = self.result.working_memory
        record = wm.record_for(fact_key)
        if record is None or record.provenance is not Provenance.DERIVED:
            return None
        rule = self._by_id.get(record.rule_id or "")
        if rule is None:
            return None
        facts = wm.as_mapping()
        satisfied = [c.describe() for c in rule.conditions if c.evaluate(facts)]
        supporting = {c.key: wm.get(c.key) for c in rule.conditions}
        return Justification(
            fact=fact_key,
            value=record.value,
            rule_id=rule.rule_id,
            rule_name=rule.name,
            because=satisfied,
            supporting_facts=supporting,
        )

    def justify_all(self) -> List[Justification]:
        """Justify every derived decision fact."""
        out: List[Justification] = []
        for key in self.result.working_memory.derived_keys():
            j = self.justify(key)
            if j is not None:
                out.append(j)
        return out

    # ------------------------------------------------------------------ #
    # WHY: narrative summary of the whole run
    # ------------------------------------------------------------------ #
    def narrative(self) -> str:
        """Return a natural-language summary of the inference run."""
        lines: List[str] = []
        n = len(self.result.firings)
        lines.append(
            f"The engine fired {n} rule(s) over {self.result.iterations} "
            f"iteration(s) and "
            f"{'reached a stable conclusion' if self.result.reached_fixed_point else 'hit the iteration cap'}."
        )
        for f in self.result.firings:
            rule = self._by_id.get(f.rule_id)
            rationale = f" — {rule.description}" if rule and rule.description else ""
            asserted = ", ".join(f"{k}={v!r}" for k, v in f.asserted.items())
            lines.append(
                f"  [iter {f.iteration}] {f.rule_id} «{f.rule_name}» fired, "
                f"asserting {asserted}{rationale}"
            )
        conclusions = self.result.conclusions
        if conclusions:
            lines.append("Final decisions:")
            for k, v in conclusions.items():
                lines.append(f"  • {k} = {v!r}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Audit trail
    # ------------------------------------------------------------------ #
    def audit_trail(self) -> List[str]:
        """Return an ordered, machine-checkable audit trail of all assertions."""
        trail: List[str] = []
        for rec in self.result.working_memory.log:
            origin = (f"rule {rec.rule_id}" if rec.provenance is Provenance.DERIVED
                      else "input")
            trail.append(f"{rec.key} := {rec.value!r}  ({origin})")
        return trail
