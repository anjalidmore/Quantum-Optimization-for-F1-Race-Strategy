# Phase 1 · Task 1 — Knowledge Representation

Part of **Quantum Optimization for Formula 1 Race Strategy** — a computational
intelligence framework for Formula One race-strategy decision support.

> **Scope of this module:** purely classical knowledge representation. No
> machine learning, no quantum computing. This task establishes the *knowledge
> foundation* on which every later phase builds.

---

## 1. What this module does

It formalises the Formula 1 racing domain as a machine-readable knowledge base
and produces every Task-1 deliverable **from a single declarative source of
truth** (`src/f1kr/schema.py`), so documentation can never drift from code.

| Deliverable | Artifact produced |
|-------------|-------------------|
| Entity table | `outputs/reports/entity_table.md` |
| Attribute table | `outputs/reports/attribute_table.md` |
| Relationship table | `outputs/reports/relationship_table.md` |
| Ontology (OWL 2) | `outputs/ontology/formula1.owl` |
| Ontology documentation | `outputs/reports/ontology_documentation.md` |
| Knowledge graph (schema + instance) | `outputs/graph/*.graphml`, `outputs/graph/instance_graph.ttl` |
| Knowledge graph visualisation | `outputs/diagrams/schema_graph.png`, `instance_graph.png`, `schema_graph.mmd` |
| Validation report | `outputs/reports/validation_report.md` |

The domain covers **61 entities** and **29 relationships** spanning both the
historical Kaggle dataset (Ergast-derived: seasons, races, circuits, drivers,
constructors, results, standings, pit stops, lap times, qualifying) and the
FastF1 live-timing / telemetry domain (sessions, stints, tyre compounds,
weather, track status, race-control messages, car data, telemetry channels).

---

## 2. Architecture

```
schema.py  ──►  ontology.py       ──►  formula1.owl        (OWL 2 classes/properties)
    │      ──►  knowledge_graph.py ──►  schema/instance graphs + RDF (Turtle)
    │      ──►  validation.py      ──►  CheckResult records  (schema/ontology/instance)
    │      ──►  visualize.py       ──►  PNG + Mermaid diagrams
    └──────►  reports.py           ──►  Markdown tables + reports

build_knowledge_base.py orchestrates all of the above end-to-end.
```

* **`schema.py`** — declarative model: `Entity`, `Attribute`, `Relationship`
  dataclasses with datatypes, cardinalities, categories, and source provenance
  (`kaggle` / `fastf1` / `both` / `derived`).
* **`ontology.py`** — maps the schema to an OWL 2 ontology via `owlready2`:
  entities → classes (with a category/subclass hierarchy), attributes → data
  properties, relationships → object properties. Cardinalities are encoded as
  OWL property characteristics (`FunctionalProperty`, `InverseFunctionalProperty`).
* **`knowledge_graph.py`** — builds a NetworkX **schema graph** (TBox) and a
  populated **instance graph** (ABox — a realistic Monaco GP 2023 scenario), and
  exports the instance graph to RDF/Turtle via `rdflib`.
* **`validation.py`** — 8 integrity checks across schema, ontology, and instance
  layers (type-compatibility of every instance edge respects the subclass
  hierarchy).
* **`visualize.py`** — headless Matplotlib PNG rendering + Mermaid source.
* **`reports.py`** — Markdown deliverable generators.

---

## 3. Installation

Requires **Python 3.10+**.

```bash
cd phase1_knowledge_representation
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

---

## 4. Execution — one command

```bash
python build_knowledge_base.py
```

This regenerates every deliverable into `outputs/` and **exits non-zero if any
validation check fails**, so it doubles as a CI gate. Optional custom output
directory:

```bash
python build_knowledge_base.py --output-dir /path/to/outputs
```

Expected tail of the log:

```
Ontology built: 73 classes, 29 object props, 90 data props
Schema graph:  {'nodes': 61, 'edges': 29, ...}
Instance graph:{'nodes': 17, 'edges': 20, 'weakly_connected': 1}
...
Summary: 8/8 checks passed — ✅ ALL CHECKS PASSED
All Task-1 deliverables generated successfully in .../outputs
```

---

## 5. Tests

```bash
pytest -q
```

15 unit tests cover schema integrity, ontology construction (including the
category-vs-entity name-collision regression), graph structure, RDF round-trip,
the validation suite (including a negative test that a malformed edge is
*caught*), report generation, and visualisation rendering.

---

## 6. Folder structure

```
phase1_knowledge_representation/
├── build_knowledge_base.py     # end-to-end driver / CI gate
├── requirements.txt
├── pytest.ini
├── README.md
├── src/f1kr/
│   ├── __init__.py
│   ├── schema.py               # single source of truth
│   ├── ontology.py             # OWL 2 builder (owlready2)
│   ├── knowledge_graph.py      # NetworkX + RDF graphs
│   ├── validation.py           # integrity checks
│   ├── visualize.py            # PNG + Mermaid
│   └── reports.py              # Markdown deliverables
├── tests/
│   └── test_knowledge_representation.py
├── docs/
│   └── ER_DIAGRAM.md           # Mermaid entity-relationship diagram
└── outputs/                    # generated deliverables (created on run)
    ├── ontology/formula1.owl
    ├── graph/{schema,instance}_graph.graphml, instance_graph.ttl
    ├── diagrams/{schema,instance}_graph.png, schema_graph.mmd
    └── reports/*.md
```

---

## 7. Design decisions & rationale

* **Single source of truth.** Every table, the ontology, and both graphs derive
  from `schema.py`. Adding an entity or relationship in one place updates all
  deliverables on the next run — no manual synchronisation.
* **Cardinality → OWL characteristics.** `*..1` and `1..1` relationships become
  `FunctionalProperty`; `1..1` additionally becomes `InverseFunctionalProperty`.
  This lets an OWL reasoner enforce the intended multiplicity.
* **Subclass-aware validation.** The instance validator resolves the subclass
  closure, so an edge whose declared range is `TrackStatus` legitimately accepts
  a `SafetyCar` individual.
* **Provenance on every entity.** Each entity records whether it originates from
  the Kaggle historical dataset, FastF1, both, or is derived — this feeds the
  data-engineering task later in Phase 1.

---

## 8. Next task

**Task 2 — Rule-Based Expert System** consumes this knowledge base: the
entities and relationships defined here become the vocabulary over which the
production rules (forward/backward chaining, conflict resolution, explanation)
operate.
