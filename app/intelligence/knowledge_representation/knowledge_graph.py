"""
f1kr.knowledge_graph
====================

Constructs the Formula 1 knowledge graph in two complementary forms:

1. **Schema graph** -- a :class:`networkx.MultiDiGraph` whose nodes are the
   domain *entities* and whose edges are the *relationships* between them. This
   is the conceptual / TBox-level graph used for visualisation and validation.

2. **Instance graph** -- a small, fully-worked *ABox* populated with a realistic
   sample (e.g. Max Verstappen driving for Red Bull at the 2023 Monaco Grand
   Prix) so that the knowledge representation is demonstrably instantiable, not
   merely declarative. This instance graph is also exported to RDF triples via
   ``rdflib`` for interoperability.

Both graphs are derived from :mod:`f1kr.schema`, keeping a single source of
truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import networkx as nx
from rdflib import RDF, RDFS, Literal, Namespace, URIRef
from rdflib import Graph as RDFGraph

from .schema import ENTITIES, RELATIONSHIPS

F1 = Namespace("http://f1kr.org/resource/")
F1O = Namespace("http://f1kr.org/ontology/formula1#")


def build_schema_graph() -> nx.MultiDiGraph:
    """
    Build the entity/relationship *schema* graph.

    Nodes carry ``category``, ``source`` and ``description`` attributes; edges
    carry the relationship ``name``, ``cardinality`` and ``description``.
    """
    g = nx.MultiDiGraph(name="F1 Knowledge Schema Graph")
    for ent in ENTITIES:
        g.add_node(
            ent.name,
            category=ent.category,
            source=ent.source,
            description=ent.description,
            n_attributes=len(ent.attributes),
        )
    for rel in RELATIONSHIPS:
        g.add_edge(
            rel.domain,
            rel.range,
            key=rel.name,
            name=rel.name,
            cardinality=rel.cardinality.value,
            description=rel.description,
        )
    return g


@dataclass(frozen=True)
class Triple:
    """A subject-predicate-object triple in the instance graph."""

    subject: str
    predicate: str
    obj: str


# A realistic, self-consistent sample scenario used to demonstrate the ABox.
# Each tuple is (node_id, entity_type, {attribute: value}).
_SAMPLE_INDIVIDUALS: Tuple[Tuple[str, str, Dict[str, object]], ...] = (
    ("season_2023", "Season", {"year": 2023}),
    ("gp_monaco", "GrandPrix", {"name": "Monaco Grand Prix"}),
    ("circuit_monaco", "Circuit",
     {"name": "Circuit de Monaco", "country": "Monaco", "length_km": 3.337}),
    ("race_monaco_2023", "Race",
     {"round": 7, "name": "Monaco Grand Prix", "date": "2023-05-28"}),
    ("driver_verstappen", "Driver",
     {"code": "VER", "number": 1, "forename": "Max", "surname": "Verstappen",
      "nationality": "Dutch"}),
    ("driver_alonso", "Driver",
     {"code": "ALO", "number": 14, "forename": "Fernando", "surname": "Alonso",
      "nationality": "Spanish"}),
    ("constructor_redbull", "Constructor",
     {"name": "Red Bull", "nationality": "Austrian"}),
    ("constructor_aston", "Constructor",
     {"name": "Aston Martin", "nationality": "British"}),
    ("session_race", "Session", {"session_type": "R"}),
    ("lap_ver_1", "Lap", {"lap_number": 1, "lap_time": 78.421}),
    ("pitstop_ver_1", "PitStop", {"lap": 54, "duration": 22.9, "stop_number": 1}),
    ("stint_ver_1", "Stint", {"stint_number": 1, "start_lap": 1, "end_lap": 54}),
    ("compound_medium", "TyreCompound", {"compound": "MEDIUM", "colour": "yellow"}),
    ("weather_dry", "Weather", {}),
    ("strategy_onestop", "Strategy", {"name": "One-stop (M->H)", "n_stops": 1}),
    ("position_p1", "Position", {"value": 1}),
    ("result_ver_monaco", "Result", {"position": 1, "points": 25.0, "status": "Finished"}),
)

# (subject_id, relationship_name, object_id)
_SAMPLE_TRIPLES: Tuple[Triple, ...] = (
    Triple("driver_verstappen", "drives_for", "constructor_redbull"),
    Triple("driver_alonso", "drives_for", "constructor_aston"),
    Triple("driver_verstappen", "participates_in", "race_monaco_2023"),
    Triple("driver_alonso", "participates_in", "race_monaco_2023"),
    Triple("race_monaco_2023", "held_at", "circuit_monaco"),
    Triple("race_monaco_2023", "belongs_to_season", "season_2023"),
    Triple("race_monaco_2023", "instance_of_gp", "gp_monaco"),
    Triple("race_monaco_2023", "contains_lap", "lap_ver_1"),
    Triple("lap_ver_1", "lap_of_driver", "driver_verstappen"),
    Triple("lap_ver_1", "lap_in_session", "session_race"),
    Triple("driver_verstappen", "performs_pitstop", "pitstop_ver_1"),
    Triple("pitstop_ver_1", "pitstop_begins_stint", "stint_ver_1"),
    Triple("pitstop_ver_1", "pitstop_changes_tyre", "compound_medium"),
    Triple("stint_ver_1", "stint_uses_compound", "compound_medium"),
    Triple("weather_dry", "weather_affects_strategy", "strategy_onestop"),
    Triple("strategy_onestop", "strategy_recommends_pitstop", "pitstop_ver_1"),
    Triple("strategy_onestop", "strategy_recommends_tyre", "compound_medium"),
    Triple("strategy_onestop", "strategy_predicts_position", "position_p1"),
    Triple("driver_verstappen", "driver_has_result", "result_ver_monaco"),
    Triple("result_ver_monaco", "result_in_race", "race_monaco_2023"),
)


def build_instance_graph() -> nx.MultiDiGraph:
    """
    Build a populated *instance* (ABox) graph from the sample scenario.

    Node ids are individual identifiers; each node carries its ``entity_type``
    plus concrete attribute values. Edges carry the relationship ``name``.
    """
    g = nx.MultiDiGraph(name="F1 Knowledge Instance Graph")
    for node_id, etype, attrs in _SAMPLE_INDIVIDUALS:
        g.add_node(node_id, entity_type=etype, **attrs)
    for t in _SAMPLE_TRIPLES:
        g.add_edge(t.subject, t.obj, key=t.predicate, name=t.predicate)
    return g


def instance_graph_to_rdf() -> RDFGraph:
    """
    Serialise the sample instance graph to an ``rdflib`` RDF graph.

    Individuals are typed with ``rdf:type`` against the ontology namespace and
    connected with object properties; literal attributes become datatype
    properties.
    """
    rdf = RDFGraph()
    rdf.bind("f1", F1)
    rdf.bind("f1o", F1O)

    for node_id, etype, attrs in _SAMPLE_INDIVIDUALS:
        subj = URIRef(F1[node_id])
        rdf.add((subj, RDF.type, URIRef(F1O[etype])))
        rdf.add((subj, RDFS.label, Literal(node_id)))
        for key, value in attrs.items():
            rdf.add((subj, URIRef(F1O[f"{etype}_{key}"]), Literal(value)))

    for t in _SAMPLE_TRIPLES:
        rdf.add((URIRef(F1[t.subject]), URIRef(F1O[t.predicate]), URIRef(F1[t.obj])))

    return rdf


def export_rdf(path: str | Path, fmt: str = "turtle") -> Path:
    """Write the instance graph RDF to ``path`` in the given ``fmt``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    instance_graph_to_rdf().serialize(destination=str(path), format=fmt)
    return path


def graph_summary(g: nx.MultiDiGraph) -> Dict[str, object]:
    """Return summary statistics for a knowledge graph."""
    return {
        "name": g.graph.get("name", ""),
        "nodes": g.number_of_nodes(),
        "edges": g.number_of_edges(),
        "is_dag": nx.is_directed_acyclic_graph(g),
        "weakly_connected": nx.number_weakly_connected_components(g),
    }
