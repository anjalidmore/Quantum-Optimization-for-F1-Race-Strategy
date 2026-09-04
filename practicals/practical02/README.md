# Phase 1 · Task 2 — Rule-Based Expert System

Part of **Quantum Optimization for Formula 1 Race Strategy**.

> **Scope of this module:** a classical, symbolic production-rule expert system.
> No machine learning, no quantum computing. It reasons over the vocabulary
> established by Task 1 (Knowledge Representation) to recommend race-strategy
> decisions with full, auditable explanations.

---

## 1. What this module does

Given a snapshot of the race state (weather, tyres, position, track status,
fuel, …), the expert system infers strategy recommendations — which tyre to
fit, whether and when to pit, how many stops to plan, how to manage fuel/energy,
and how to race the cars around you — and **explains every conclusion**.

It is a complete production-rule system:

| Component | File | Responsibility |
|-----------|------|----------------|
| Rule schema | `rules_schema.py` | Typed `Condition` / `Action` / `Rule` objects with operators, salience, connectives |
| Working memory | `working_memory.py` | Instrumented fact base with provenance + a canonical fact-key registry |
| Rule base | `rule_base.py` | Curated, non-redundant expert rules across 10 strategic domains |
| Inference engine | `inference.py` | Forward chaining, backward chaining, conflict resolution |
| Rule validator | `rule_validation.py` | Static integrity checks (typos, contradictions, bounds) |
| Explanation | `explanation.py` | HOW / WHY justifications and audit trails |
| Reports | `reports.py` | Markdown deliverable generators |

---

## 2. Why a curated rule set

A large count of near-duplicate rules inflates a number without adding
knowledge, and it actively *harms* a production system: overlapping rules create
order-dependent conflicts and make the knowledge base unmaintainable. This
module instead provides a **substantial, well-organised, non-redundant core**
that is:

* **Structured** — each rule is a typed object, serialisable to/from JSON
  (`outputs/rules/rule_base.json`), not an opaque string.
* **Validated** — a static validator proves the base is internally consistent
  (unique ids, registered keys, no order-dependent contradictions, sane
  operators and confidences).
* **Explainable** — every rule carries a human rationale surfaced in
  explanations.
* **Extensible** — the schema and fact registry make adding rules safe; the
  validator catches mistakes immediately.

The rule base spans ten categories: `weather`, `safety_car`, `pit`,
`degradation`, `tyre`, `strategy`, `fuel`, `energy`, `tactics`, `risk`. It is
designed to *grow* — the architecture, not the raw count, is the deliverable.

---

## 3. Architecture

```
race-state inputs
        │
        ▼
 WorkingMemory (GIVEN facts)
        │
        ▼
 InferenceEngine ──► forward_chain()  ─┐   conflict resolution:
        │                              │   salience → specificity → id
        │                              ▼
        │                     ForwardResult (firings + derived facts)
        │                              │
        │                              ▼
        │                        Explainer ──► HOW / WHY / audit trail
        │
        └──► backward_chain(goal) ──► ProofNode tree
```

* **Forward chaining** is data-driven: it repeatedly evaluates the conflict set,
  fires rules in resolved order, and iterates to a fixed point. A firing
  signature guard prevents a rule re-asserting an identical conclusion, so the
  loop provably terminates.
* **Backward chaining** is goal-driven: given a target `(key, value)`, it finds
  supporting rules and recursively proves their conditions, returning a proof
  tree.
* **Conflict resolution** is pluggable (`SALIENCE`, `SPECIFICITY`, `FIFO`); the
  default is salience → specificity → rule-id, which is deterministic.

---

## 4. Installation

Requires **Python 3.10+**. The runtime has **no third-party dependencies**.

```bash
cd practical02
pip3 install -r requirements.txt   # only pytest, for the tests
```

---

## 5. Execution — one command

```bash
python3 practical02.py
```

This validates the rule base, serialises it to JSON, runs five representative
scenarios through forward chaining, demonstrates backward chaining, and writes
every report to `outputs/`. It exits non-zero if validation fails (CI gate).

Expected tail:

```
Built rule base: 32 rules across 10 categories.
[PASS] unique_rule_ids ... [PASS] no_order_dependent_contradictions
Scenario 'Wet weather changeover': ... recommended_tyre='INTERMEDIATE'
Scenario 'Safety car opportunity': ... pit_decision='PIT_NOW'
Backward-chaining 'pit_decision == PIT_NOW': proven=True
All Task-2 deliverables generated successfully in .../outputs
```

### Using the engine programmatically

```python
from f1es.rule_base import build_rule_base
from f1es.inference import InferenceEngine
from f1es.explanation import Explainer

engine = InferenceEngine(build_rule_base())
result = engine.forward_chain({"track_status": "SC", "tyre_wear": 60})
print(result.conclusions)                      # {'pit_decision': 'PIT_NOW', ...}
print(Explainer(result, build_rule_base()).narrative())
```

---

## 6. Tests

```bash
pytest -q
```

21 tests cover the operator semantics, working-memory provenance, rule-base
validation (including negative tests that the validator *catches* unregistered
keys and contradictions), forward-chaining correctness and termination, salience
ordering, backward-chaining success/failure, and explanation generation.

---

## 7. Deliverables (written to `outputs/`)

| Deliverable | Path |
|-------------|------|
| Serialised rule base (JSON) | `outputs/rules/rule_base.json` |
| Rule catalogue (all rules, grouped) | `outputs/reports/rule_catalogue.md` |
| Rule-base validation report | `outputs/reports/rule_validation_report.md` |
| Per-scenario inference reports (×5, with HOW/WHY) | `outputs/reports/inference_0*.md` |
| Backward-chaining proof demonstration | `outputs/reports/backward_chaining_demo.md` |

---

## 8. Folder structure

```
practical02/
├── practical02.py        # end-to-end driver / CI gate
├── requirements.txt
├── pytest.ini
├── README.md
├── src/f1es/
│   ├── __init__.py
│   ├── rules_schema.py         # typed Condition/Action/Rule
│   ├── working_memory.py       # fact base + fact-key registry
│   ├── rule_base.py            # curated expert rules
│   ├── inference.py            # forward/backward chaining + conflict resolution
│   ├── rule_validation.py      # static integrity checks
│   ├── explanation.py          # HOW/WHY explanations
│   └── reports.py              # Markdown generators
├── tests/
│   └── test_expert_system.py
└── outputs/                    # generated on run
    ├── rules/rule_base.json
    └── reports/*.md
```

---

## 9. Relationship to other tasks

* **Consumes Task 1.** The fact keys and decision variables are drawn from the
  entities and relationships formalised in the knowledge representation.
* **Feeds later phases.** These symbolic rules provide interpretable baselines
  and guardrails that the machine-learning (Phase 2) and quantum-optimisation
  (Phase 3) components are compared against and constrained by.

## 10. Next task

**Task 3 — State-Space Search** models race strategy as a search problem
(BFS, DFS, UCS, Greedy, A\*) over strategy states, reusing this task's state
vocabulary and the expert rules as heuristic guidance.
