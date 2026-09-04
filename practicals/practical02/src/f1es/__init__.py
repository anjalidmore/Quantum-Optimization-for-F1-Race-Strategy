"""
f1es — Formula 1 Rule-Based Expert System (Phase 1, Task 2)
===========================================================

A classical production-rule expert system for F1 race strategy:

* ``rules_schema``   — typed conditions/actions/rules (structured, serialisable).
* ``working_memory`` — instrumented fact base + canonical fact-key registry.
* ``rule_base``      — curated, non-redundant expert rule set.
* ``inference``      — forward chaining, backward chaining, conflict resolution.
* ``rule_validation``— static rule-base integrity checks.
* ``explanation``    — HOW/WHY justifications and audit trails.

No machine learning, no quantum computing — this is a symbolic reasoning layer.
"""
from __future__ import annotations

__version__ = "1.0.0"
__all__ = [
    "rules_schema", "working_memory", "rule_base",
    "inference", "rule_validation", "explanation",
]
