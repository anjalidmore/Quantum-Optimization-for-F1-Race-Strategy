# Expert System Architecture

## Inference control flow

```mermaid
flowchart TD
    A[Race-state inputs] --> B[WorkingMemory: GIVEN facts]
    B --> C{Forward chaining loop}
    C --> D[Build conflict set:<br/>rules whose antecedent holds]
    D --> E[Conflict resolution:<br/>salience → specificity → id]
    E --> F[Fire winning rule/s:<br/>assert actions into WM]
    F --> G{Any fact changed?}
    G -- yes --> C
    G -- no --> H[Fixed point reached]
    H --> I[ForwardResult:<br/>firings + derived decisions]
    I --> J[Explainer]
    J --> K[HOW: per-decision justification]
    J --> L[WHY: narrative summary]
    J --> M[Audit trail with provenance]

    B --> N{Backward chaining<br/>goal query}
    N --> O[Find rules asserting goal]
    O --> P[Recursively prove conditions]
    P --> Q[ProofNode tree]
```

## Rule anatomy

```mermaid
classDiagram
    class Rule {
        +rule_id: str
        +name: str
        +category: str
        +salience: int
        +connective: Connective
        +conditions: Condition[]
        +actions: Action[]
        +specificity() int
        +is_satisfied(facts) bool
    }
    class Condition {
        +key: str
        +operator: Operator
        +value: Any
        +evaluate(facts) bool
    }
    class Action {
        +key: str
        +value: Any
        +confidence: float
    }
    Rule "1" *-- "many" Condition
    Rule "1" *-- "many" Action
```

## Conflict-resolution semantics

When multiple rules are satisfied simultaneously, the conflict set is ordered by
the active strategy:

| Strategy | Primary | Secondary | Tie-break |
|----------|---------|-----------|-----------|
| `SALIENCE` (default) | higher salience | higher specificity | rule-id |
| `SPECIFICITY` | higher specificity | higher salience | rule-id |
| `FIFO` | declaration order | — | — |

Salience bands encode strategic priority so that, e.g., a weather-critical
override (salience 100) always precedes a generic tyre-selection rule
(salience 40), guaranteeing that safety-relevant conclusions dominate.

## Termination guarantee

Forward chaining terminates because:

1. Each rule firing either changes working memory or is skipped.
2. A per-rule *firing signature* (rule-id + asserted key/value pairs) is
   recorded; a rule cannot re-assert an identical conclusion.
3. The loop halts as soon as an iteration changes nothing (fixed point), and a
   `max_iterations` cap bounds pathological cases.
