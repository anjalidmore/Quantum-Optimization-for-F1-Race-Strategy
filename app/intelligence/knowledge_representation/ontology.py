"""
f1kr.ontology
=============

Builds an OWL 2 ontology for the Formula 1 domain directly from
:mod:`f1kr.schema`, using ``owlready2``.

The builder maps:

* Each :class:`~f1kr.schema.Entity` -> an OWL class.
* Each entity ``category`` -> a top-level OWL class, with entities placed
  underneath their category (or under an explicit ``parent`` entity).
* Each :class:`~f1kr.schema.Attribute` -> an OWL *data property* whose domain is
  the owning class and whose range is the mapped XSD datatype.
* Each :class:`~f1kr.schema.Relationship` -> an OWL *object property* with the
  appropriate ``rdfs:domain`` / ``rdfs:range`` and cardinality-derived
  functional / inverse-functional characteristics.

The resulting ontology is serialised to RDF/XML (``.owl``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import owlready2
from owlready2 import (
    DataProperty,
    FunctionalProperty,
    InverseFunctionalProperty,
    ObjectProperty,
    Thing,
    get_ontology,
)

from .schema import (
    ENTITIES,
    RELATIONSHIPS,
    Cardinality,
    DataType,
    categories,
)

BASE_IRI = "http://f1kr.org/ontology/formula1#"

# Map our primitive datatypes to Python types understood by owlready2 (-> XSD).
_XSD: Dict[DataType, type] = {
    DataType.STRING: str,
    DataType.INTEGER: int,
    DataType.FLOAT: float,
    DataType.BOOLEAN: bool,
    DataType.DATETIME: str,  # stored as ISO-8601 lexical string for portability
}


def _valid_name(name: str) -> str:
    """Return a Python/OWL-safe class or property identifier."""
    return name.replace(" ", "").replace("-", "_")


def build_ontology(output_path: str | Path) -> "owlready2.Ontology":
    """
    Construct and persist the F1 OWL ontology.

    Parameters
    ----------
    output_path:
        File path to which the ontology is serialised (RDF/XML).

    Returns
    -------
    owlready2.Ontology
        The populated ontology object (also written to ``output_path``).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    onto = get_ontology(BASE_IRI)
    classes: Dict[str, type] = {}          # entity name -> OWL class
    category_roots: Dict[str, type] = {}   # category string -> OWL root class

    with onto:
        # --- 1. Category root classes -------------------------------------
        # Category roots are suffixed with "Category" so they never collide
        # with an entity of the same name (e.g. the "Weather" category vs. the
        # "Weather" entity). Kept in a separate registry from entity classes.
        for cat in categories():
            category_roots[cat] = types_new_class(f"{_valid_name(cat)}Category", (Thing,))

        # --- 2. Entity classes, resolving parents -------------------------
        # Roots (no explicit parent) are created first so that a child whose
        # parent is another entity always resolves against an existing class.
        ordered = [e for e in ENTITIES if not e.parent] + [e for e in ENTITIES if e.parent]
        for ent in ordered:
            parent_cls = classes[ent.parent] if ent.parent else category_roots[ent.category]
            cls = types_new_class(_valid_name(ent.name), (parent_cls,))
            cls.comment = [ent.description]
            classes[ent.name] = cls

        # --- 3. Data properties from attributes ---------------------------
        for ent in ENTITIES:
            owner = classes[ent.name]
            for attr in ent.attributes:
                prop_name = f"{_valid_name(ent.name)}_{attr.name}"
                prop = types_new_class(prop_name, (DataProperty,))
                prop.domain = [owner]
                prop.range = [_XSD[attr.dtype]]
                label = f"{attr.description}"
                if attr.unit:
                    label += f" [{attr.unit}]"
                prop.comment = [label]

        # --- 4. Object properties from relationships ----------------------
        for rel in RELATIONSHIPS:
            dom = classes[rel.domain]
            rng = classes[rel.range]
            bases: tuple = (ObjectProperty,)
            # Encode cardinality as OWL property characteristics.
            if rel.cardinality in (Cardinality.MANY_TO_ONE, Cardinality.ONE_TO_ONE):
                bases = (ObjectProperty, FunctionalProperty)
            prop = types_new_class(_valid_name(rel.name), bases)
            prop.domain = [dom]
            prop.range = [rng]
            prop.comment = [rel.description]
            if rel.cardinality == Cardinality.ONE_TO_ONE:
                # 1..1 is both functional and inverse-functional.
                prop.is_a.append(InverseFunctionalProperty)

    onto.save(file=str(output_path), format="rdfxml")
    return onto


def types_new_class(name: str, bases: tuple) -> type:
    """
    Create a new owlready2 class dynamically.

    ``owlready2`` classes are ordinary Python classes created inside an
    ``with onto:`` block. ``type(name, bases, {})`` is the supported idiom for
    programmatic creation.
    """
    return type(name, bases, {})


def ontology_statistics(onto) -> Dict[str, int]:
    """Return simple counts describing the built ontology."""
    return {
        "classes": len(list(onto.classes())),
        "object_properties": len(list(onto.object_properties())),
        "data_properties": len(list(onto.data_properties())),
    }
