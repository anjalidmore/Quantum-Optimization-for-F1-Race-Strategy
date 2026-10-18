"""
scripts.study_guide.content
===========================

The hand-written half of the study guide: the "Understand the Code" walkthrough
of the ten practicals and the Showcase slides.

What is written here and what is not:

* **Prose is written here.** It explains code that exists in this repository;
  every claim was checked against the module it describes.
* **Numbers are not.** A ``[[name]]`` placeholder is filled from the code or a
  committed artifact by ``build_study_guide.collect_facts`` at build time, so a
  retrain changes the guide on the next build instead of leaving it wrong. An
  unknown placeholder fails the build.
* **Code is not.** A snippet or function reference names a file and a
  qualified name (``RaceProblem.heuristic``); the builder pulls the source and
  its line numbers through :mod:`ast`. A renamed function fails the build
  instead of shipping a stale excerpt.

Prose fields are Markdown.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# The ten practicals. "Practical N" is Task N of the lab specification; all ten
# live in one application (app/), not ten folders.
# ---------------------------------------------------------------------------
PRACTICALS: list[dict] = [
    # ------------------------------------------------------------------ 1
    {
        "number": 1,
        "title": "Knowledge Representation",
        "tagline": "An F1 ontology and knowledge graph built from one declarative schema.",
        "package": "app/intelligence/knowledge_representation/",
        "run": "python scripts/build_knowledge_base.py",
        "what": [
            "Practical 1 models the Formula 1 domain explicitly: **[[kr_entities]] entities** "
            "(Season, Race, Driver, Stint, PitStop, TyreCompound, ...) grouped into "
            "**[[kr_categories]] categories**, their typed attributes, and **[[kr_relationships]] "
            "relationships** with cardinalities.",
            "**Why:** a shared, machine-readable vocabulary makes the domain inspectable "
            "before any model is trained, and gives every later artefact (OWL file, graphs, "
            "tables, validation report) one definition to agree with. The design rule in "
            "`schema.py` is that nothing hard-codes an entity or relationship twice — the "
            "ontology, both graphs and all five reports are *generated* from the same tuples.",
        ],
        "steps": [
            ("Declare the domain as data",
             "`schema.py` defines `Entity`, `Attribute` and `Relationship` as frozen dataclasses "
             "and lists the whole domain in two tuples, `ENTITIES` and `RELATIONSHIPS`. Each "
             "attribute carries a `DataType` (mapped to XSD later) and each relationship a "
             "`Cardinality`."),
            ("Build the OWL ontology",
             "`build_ontology` creates one root class per category (suffixed `Category` so the "
             "`Weather` category cannot collide with the `Weather` entity), then entity classes "
             "— roots first, so a child's parent class always exists. Attributes become "
             "`DataProperty`s named `Entity_attribute` with an XSD range; relationships become "
             "`ObjectProperty`s. Cardinality is encoded as OWL characteristics: many-to-one and "
             "one-to-one are `FunctionalProperty`, one-to-one is also `InverseFunctionalProperty`. "
             "The result is saved as RDF/XML."),
            ("Build two graphs",
             "`build_schema_graph` turns the schema into a NetworkX `MultiDiGraph` (a multigraph, "
             "because two entities can be linked by more than one named relationship). "
             "`build_instance_graph` populates a small sample race (the ABox), and "
             "`instance_graph_to_rdf` serialises it as RDF triples with `rdf:type` and "
             "`rdfs:label` for Turtle export."),
            ("Validate the model",
             "`run_all_checks` runs the suite in `validation.py`: unique entity and relationship "
             "names, every relationship endpoint exists, every declared parent exists, no orphan "
             "entities, and the instance graph conforms to the schema. The driver script exits "
             "non-zero on any failure, so it doubles as a CI gate."),
            ("Render and report",
             "`visualize.py` draws the schema and instance graphs as PNGs and emits a Mermaid "
             "source file; `reports.py` writes the entity, attribute, relationship, ontology and "
             "validation tables as Markdown."),
        ],
        "tech": [
            ("owlready2", "Creates OWL classes and properties as Python objects and writes RDF/XML "
                          "that Protégé or any OWL reasoner can open. Chosen because the ontology "
                          "is generated from code, not drawn by hand."),
            ("rdflib", "Builds the instance graph as RDF triples and serialises Turtle (`.ttl`), "
                       "the format SPARQL tools expect."),
            ("NetworkX", "Graph structure, summary statistics and GraphML export of both graphs."),
            ("matplotlib", "Static PNG renders of the graphs, coloured by category."),
            ("dataclasses + Enum", "Frozen, typed domain declarations — the single source of truth."),
        ],
        "functions": [
            ("app/intelligence/knowledge_representation/schema.py", "Relationship",
             "A named, directed relationship with domain, range and cardinality."),
            ("app/intelligence/knowledge_representation/ontology.py", "build_ontology",
             "Generates and saves the OWL ontology from the schema."),
            ("app/intelligence/knowledge_representation/knowledge_graph.py", "build_schema_graph",
             "Schema as a NetworkX multigraph."),
            ("app/intelligence/knowledge_representation/knowledge_graph.py", "instance_graph_to_rdf",
             "Sample race individuals as RDF triples."),
            ("app/intelligence/knowledge_representation/validation.py", "run_all_checks",
             "Runs every structural check and returns the results."),
            ("app/intelligence/knowledge_representation/visualize.py", "render_schema_mermaid",
             "Mermaid `graph LR` source for the schema."),
        ],
        "snippets": [
            ("app/intelligence/knowledge_representation/ontology.py", "build_ontology",
             "The whole ontology is generated in four passes over the schema."),
            ("app/intelligence/knowledge_representation/validation.py", "check_relationship_endpoints",
             "A typical validation check: every relationship must point at real entities."),
        ],
        "outputs": [
            ("artifacts/knowledge_representation/ontology/formula1.owl", "The OWL ontology (RDF/XML)."),
            ("artifacts/knowledge_representation/graph/schema_graph.graphml", "Schema graph, GraphML."),
            ("artifacts/knowledge_representation/graph/instance_graph.ttl", "Sample race as RDF Turtle."),
            ("artifacts/knowledge_representation/diagrams/schema_graph.mmd", "Mermaid diagram source."),
            ("artifacts/knowledge_representation/reports/validation_report.md", "Validation results."),
        ],
        "figure": ("artifacts/knowledge_representation/diagrams/schema_graph.png",
                   "The generated schema graph, coloured by category."),
        "example": ("file", "artifacts/knowledge_representation/reports/validation_report.md", 16),
        "viva": [
            ("Why generate every deliverable from one schema?",
             "Single source of truth: the tables, the ontology and both graphs all derive from `schema.py`, so "
             "adding an entity or relationship in one place updates every deliverable on the next run."),
            ("How is cardinality represented in OWL?",
             "Many-to-one and one-to-one relationships become `FunctionalProperty`; one-to-one is also "
             "`InverseFunctionalProperty`, so an OWL reasoner can enforce the intended multiplicity."),
            ("Why is the instance validation subclass-aware?",
             "`_subclass_closure` resolves each entity's descendants, so an edge whose declared range is "
             "`TrackStatus` legitimately accepts a `SafetyCar` individual."),
            ("What does an entity's `source` field record?",
             "Whether it comes from the Kaggle historical schema, FastF1, or both — provenance carried into the "
             "data-engineering practical."),
        ],
        "applications": [
            "Open `formula1.owl` in Protégé to browse the class hierarchy or run a reasoner.",
            "Query `instance_graph.ttl` with SPARQL (e.g. every stint a driver ran on a compound).",
            "Use the entity/attribute tables as the data dictionary for Practicals 4 and 5.",
        ],
    },
    # ------------------------------------------------------------------ 2
    {
        "number": 2,
        "title": "Rule-Based Expert System",
        "tagline": "[[es_rules]] production rules with forward and backward chaining, and an explanation for every firing.",
        "package": "app/intelligence/expert_system/",
        "run": "python scripts/run_expert_system.py",
        "what": [
            "Practical 2 encodes race-strategy expertise as **[[es_rules]] IF–THEN rules** in "
            "[[es_categories]] categories (weather, safety car, tyre, pit, strategy, ...), and an "
            "inference engine that derives decisions such as `pit_decision`, `recommended_tyre` "
            "and `risk_level` from a race state.",
            "**Why:** some strategy knowledge is categorical and well understood — a safety car "
            "makes a stop cheap, standing water needs full wets — and a statistical model trained "
            "on one dry race could never learn it. Rules also explain themselves: every derived "
            "fact records which rule produced it and which conditions held. In Practical 9 the "
            "rule verdict takes precedence over the ML models for exactly this reason.",
        ],
        "steps": [
            ("Rules are data, not if-statements",
             "`rules_schema.py` defines `Condition` (`fact <operator> value`), `Action` "
             "(`set key = value`, with an optional confidence) and `Rule` (conditions, connective "
             "ALL/ANY, actions, salience). `rule_base.py` lists the rules with the `rule()`, "
             "`cond()` and `act()` shorthands."),
            ("Closed-world condition evaluation",
             "`Condition.evaluate` returns False for a missing or `None` fact and for incomparable "
             "types, so a rule never fires on data it does not have."),
            ("Working memory with provenance",
             "`WorkingMemory` stores each fact with its provenance (input or derived) and the rule "
             "that asserted it. `assert_fact` returns whether memory actually changed — the "
             "signal the engine uses to detect a fixed point."),
            ("Forward chaining (data-driven)",
             "Each iteration, `forward_chain` collects every satisfied rule (the conflict set), "
             "orders it by the chosen strategy — by default salience, then specificity (more "
             "conditions wins), then rule id — and fires the rules that still change memory. A "
             "rule/value signature stops identical re-firings. It stops at a fixed point or after "
             "200 iterations. `notes` accumulate instead of overwriting."),
            ("Backward chaining (goal-driven)",
             "`backward_chain` proves a goal such as `pit_decision == 'PIT_NOW'`: it finds rules "
             "that assert it, checks each condition against the facts, recurses on equality "
             "conditions it cannot yet satisfy, and returns a `ProofNode` tree. A visited set "
             "stops cycles."),
            ("Validate and explain",
             "`validate_rule_base` checks unique ids, non-empty rules, registered fact keys, "
             "operator/value sanity, confidence bounds, and contradictions (identical antecedents "
             "and salience asserting different values). `Explainer.justify` answers *how* a fact "
             "was derived: the rule, the conditions that held and the supporting facts."),
        ],
        "tech": [
            ("Pure Python", "No rule-engine library: the engine is ~270 lines, so its conflict "
                            "resolution and fixed-point logic are visible and unit-tested."),
            ("dataclasses (frozen) + Enum", "Immutable, hashable rules and type-safe operators."),
            ("operator module", "Maps each `Operator` to a comparison function (`_OPS`)."),
            ("JSON export", "`rule_base.json` makes the rule base readable outside Python."),
        ],
        "functions": [
            ("app/intelligence/expert_system/rules_schema.py", "Condition.evaluate",
             "Closed-world test of one antecedent."),
            ("app/intelligence/expert_system/rules_schema.py", "Rule.is_satisfied",
             "ALL / ANY over a rule's conditions."),
            ("app/intelligence/expert_system/inference.py", "_order_conflict_set",
             "Conflict resolution: salience, specificity, FIFO."),
            ("app/intelligence/expert_system/inference.py", "InferenceEngine.forward_chain",
             "Data-driven inference to a fixed point."),
            ("app/intelligence/expert_system/inference.py", "InferenceEngine._prove",
             "Recursive goal proof for backward chaining."),
            ("app/intelligence/expert_system/explanation.py", "Explainer.justify",
             "HOW-explanation for a derived fact."),
            ("app/intelligence/expert_system/rule_validation.py", "check_no_contradictions",
             "Detects order-dependent conflicting rules."),
        ],
        "snippets": [
            ("app/intelligence/expert_system/rule_base.py", "build_rule_base",
             "The first rule group, as declared (excerpt).",
             ("rules: List[Rule] = []", "# 2. SAFETY CAR")),
            ("app/intelligence/expert_system/inference.py", "_order_conflict_set",
             "Conflict resolution is a sort key."),
            ("app/intelligence/expert_system/inference.py", "InferenceEngine.forward_chain",
             "The forward-chaining loop."),
        ],
        "outputs": [
            ("artifacts/expert_system/rules/rule_base.json", "The rule base as JSON."),
            ("artifacts/expert_system/reports/rule_catalogue.md", "Every rule, in IF–THEN form."),
            ("artifacts/expert_system/reports/decision_table.md", "Condition/action matrix."),
            ("artifacts/expert_system/reports/inference_02_safety_car_opportunity.md",
             "One of five worked scenarios with its firing trace."),
            ("artifacts/expert_system/reports/backward_chaining_demo.md", "A backward-chaining proof."),
            ("artifacts/expert_system/reports/rule_validation_report.md", "Static validation results."),
        ],
        "example": ("file", "artifacts/expert_system/reports/inference_02_safety_car_opportunity.md", 40),
        "viva": [
            ("Why [[es_rules]] curated rules rather than hundreds?",
             "Near-duplicate rules inflate a count without adding knowledge and create order-dependent "
             "conflicts. A smaller, non-redundant base can be validated: `validate_rule_base` checks unique ids, "
             "registered fact keys, operator/value sanity, confidence bounds and contradictions."),
            ("Forward or backward chaining — when is each used?",
             "Forward chaining derives every conclusion from the race state (the live pipeline uses it); "
             "backward chaining starts from a goal such as `pit_decision == 'PIT_NOW'` and returns a proof tree."),
            ("How is a conflict between satisfied rules resolved?",
             "By default salience first, then specificity (more conditions), then rule id — a deterministic sort key."),
        ],
        "applications": [
            "Encoding pit-wall policy (safety-car stops, weather changeovers) that must hold "
            "regardless of what the data says.",
            "Auditable decisions: every recommendation carries the rules that produced it.",
            "The live strategy pipeline (Practical 9) runs `forward_chain` on every request.",
        ],
        "caveat": "The live pipeline populates [[es_populated_keys]] input keys. The evaluation run "
                  "of [[eval_date]] found **[[es_unreachable]] of [[es_rules]] rules** condition on "
                  "keys it never fills, so they are correct in isolation but cannot fire through "
                  "`/api/strategy/predict` (`artifacts/evaluation/domain_validation.json`).",
    },
    # ------------------------------------------------------------------ 3
    {
        "number": 3,
        "title": "State-Space Search",
        "tagline": "Pit strategy as a shortest-path problem, solved by BFS, DFS, UCS, Greedy and A*.",
        "package": "app/intelligence/search/",
        "run": "python scripts/run_search.py",
        "what": [
            "Practical 3 casts strategy as **search over race states**: from lap 0 on the start "
            "tyre, choose at every lap whether to run on or pit for a compound, so that the full "
            "distance takes the least estimated time.",
            "**Why:** the cost structure is a genuine trade-off — softer tyres are faster but "
            "degrade sooner, each stop costs [[search_pit_loss]] s — and search finds the "
            "provably optimal plan under an explicit model rather than a heuristic guess. "
            "Comparing five algorithms on the same problem shows the optimality/efficiency "
            "trade-off with real counts.",
        ],
        "steps": [
            ("State",
             "`RaceState` is a frozen dataclass: lap, compound, tyre age, stops made, fuel, and the "
             "set of compounds used. `key()` buckets fuel to one decimal so floating-point noise "
             "does not split identical states in the explored set."),
            ("Actions and transitions",
             "`actions` returns `RUN`, or `PIT(compound)` for each different allowed compound while "
             "stops remain and it is not the last lap. `result` applies one: a pit stop resets tyre "
             "age, adds the compound to the used set and counts the out-lap as a lap."),
            ("Cost model",
             "`step_cost` is the time the action adds: `lap_time` = compound base pace + linear "
             "degradation (plus an extra 2.5× the degradation rate for every lap past the "
             "compound's cliff), scaled up on a track hotter than 30 °C, + a fuel-weight penalty; "
             "a pit adds the pit loss to an out-lap on fresh tyres."),
            ("Goal",
             "`is_goal` requires the full distance and, in a dry race, at least two distinct dry "
             "compounds — the real F1 regulation."),
            ("Algorithms",
             "BFS (FIFO) and DFS (LIFO with a depth bound) are uninformed. UCS, Greedy and A* share "
             "one `_best_first` core that differs only in the priority: g(n), h(n) or g(n)+h(n). "
             "It uses a heap with a counter tie-breaker for determinism and skips stale entries "
             "when a cheaper route to a state was found."),
            ("Admissible heuristic",
             "`heuristic` = laps remaining × fastest fresh-tyre base pace. No lap can be faster than "
             "that, and it ignores degradation, fuel and any mandatory stop, so it never "
             "overestimates — which is what makes A* optimal."),
            ("Measure",
             "`_instrument` wraps each run with `time.perf_counter` and `tracemalloc` for wall time "
             "and peak memory; `comparison.compare` tabulates cost gap, nodes expanded and frontier "
             "size."),
        ],
        "tech": [
            ("heapq / collections.deque", "Priority queue for UCS/Greedy/A*, FIFO queue for BFS."),
            ("dataclasses (frozen)", "Hashable states for the explored set."),
            ("tracemalloc + perf_counter", "Measured, not estimated, time and memory per algorithm."),
            ("NetworkX + matplotlib", "Search-tree slice, stint bar and metric charts."),
        ],
        "functions": [
            ("app/intelligence/search/problem.py", "RaceProblem.lap_time", "The lap-time cost model."),
            ("app/intelligence/search/problem.py", "RaceProblem.actions", "Legal actions from a state."),
            ("app/intelligence/search/problem.py", "RaceProblem.is_goal", "Distance + two-compound rule."),
            ("app/intelligence/search/problem.py", "RaceProblem.heuristic", "Admissible lower bound."),
            ("app/intelligence/search/algorithms.py", "_best_first", "Shared core for UCS / Greedy / A*."),
            ("app/intelligence/search/algorithms.py", "astar_search", "A* as a one-line priority."),
        ],
        "snippets": [
            ("app/intelligence/search/problem.py", "RaceProblem.lap_time", "How a lap is costed."),
            ("app/intelligence/search/problem.py", "RaceProblem.heuristic", "Why A* stays optimal."),
            ("app/intelligence/search/algorithms.py", "_best_first", "One search loop, three algorithms."),
        ],
        "outputs": [
            ("artifacts/search/reports/comparison.json", "Every algorithm's metrics."),
            ("artifacts/search/reports/comparison_report.md", "The comparison, written up."),
            ("artifacts/search/reports/heuristic_table.md", "Admissibility check with real numbers."),
            ("artifacts/search/diagrams/optimal_strategy_path.png", "The optimal stint plan."),
        ],
        "figure": ("artifacts/search/diagrams/optimal_strategy_path.png",
                   "The optimal plan found by A* and UCS."),
        "example": ("search_table",),
        "applications": [
            "Re-planning the rest of a race: Practical 9 runs A* over the remaining laps on every "
            "strategy request.",
            "What-if analysis: change `pit_loss` or `track_temp` and re-run to see the plan change.",
        ],
    },
    # ------------------------------------------------------------------ 4
    {
        "number": 4,
        "title": "Data Engineering & EDA",
        "tagline": "Load, clean, audit and analyse eight tables, then publish one clean lap table.",
        "package": "app/intelligence/data/",
        "run": "python scripts/run_eda.py",
        "what": [
            "Practical 4 turns raw CSVs into an audited, analysed dataset. The lap table "
            "`fastf1_laps.csv` is **real**: [[data_n_laps]] laps from [[data_n_drivers]] drivers at "
            "the [[data_year]] [[data_event]] GP, fetched with FastF1 by "
            "`scripts/fetch_real_session.py`. The seven Kaggle-style tables (results, drivers, pit "
            "stops, ...) are **synthetic**, generated by `synthetic.py` with deliberately injected "
            "problems so the cleaning has something to fix.",
            "**Why:** every later practical trains on this output, so cleaning has to be "
            "reproducible and every change visible. Each step appends a `StepReport` to a "
            "`CleaningReport` — an audit trail of rows before/after and what was changed.",
        ],
        "steps": [
            ("Load against a schema",
             "`schemas.py` declares every table's columns and their kind (ID, numeric, time, "
             "date, categorical, ...). `load_csv` validates required columns and turns Ergast's "
             "`\\N` sentinels into NaN."),
            ("Clean, in five audited steps",
             "`clean_table`: (1) drop exact duplicates; (2) coerce types and parse `m:ss.mmm` lap "
             "times to seconds with `parse_laptime_to_seconds`; (3) trim and upper-case "
             "categoricals so spelling variants merge; (4) impute numerics by median and "
             "categoricals by mode, leaving IDs and dates as NaN; (5) detect IQR outliers "
             "(1.5 × IQR) and, as `run_eda.py` calls it, cap them."),
            ("Score quality and summarise",
             "`data_quality` produces a 0–100 scorecard; `statistical_summary` adds skew and "
             "kurtosis to `describe()`; `correlation_analysis` lists the strongest pairs."),
            ("Nine domain analyses",
             "Driver, constructor, circuit, pit stop, tyre (pace and degradation per compound), "
             "lap time, weather, season and safety car (track-status distribution)."),
            ("Encode and scale (demo)",
             "`encode_and_scale` one-hot encodes low-cardinality categoricals, label-encodes the "
             "rest and standardises numerics — written out as a demonstration only. Practical 5 "
             "deliberately does **not** reuse scaled data (see its leakage rule)."),
            ("Publish the hand-off",
             "The cleaned lap table is published to `data/processed/fastf1_laps_clean.csv`, and the "
             "real-vs-synthetic marker is copied to `data_source.json` so no later stage has to "
             "guess the provenance."),
        ],
        "tech": [
            ("pandas", "Tabular cleaning, grouping and analyses."),
            ("scikit-learn StandardScaler", "Standardisation in the encoding demo."),
            ("FastF1", "Downloads real timing data from F1's live-timing archive."),
            ("matplotlib / seaborn", "Figures and a composite dashboard image."),
        ],
        "functions": [
            ("app/intelligence/data/pipeline.py", "parse_laptime_to_seconds", "`1:31.196` → 91.196."),
            ("app/intelligence/data/pipeline.py", "load_csv", "Schema-validated load."),
            ("app/intelligence/data/pipeline.py", "clean_table", "The five audited cleaning steps."),
            ("app/intelligence/data/pipeline.py", "encode_and_scale", "Encoding/scaling demo."),
            ("app/intelligence/data/eda.py", "data_quality", "Quality scorecard."),
            ("app/intelligence/data/eda.py", "tyre_analysis", "Per-compound pace and degradation."),
            ("app/intelligence/data/synthetic.py", "_inject_issues", "Where the dirty data comes from."),
        ],
        "snippets": [
            ("app/intelligence/data/pipeline.py", "parse_laptime_to_seconds", "Lap-time parsing."),
            ("app/intelligence/data/pipeline.py", "clean_table", "Cleaning with an audit trail."),
        ],
        "outputs": [
            ("data/processed/fastf1_laps_clean.csv", "The canonical clean lap table (Practical 5's input)."),
            ("data/processed/data_source.json", "Real-vs-synthetic provenance."),
            ("artifacts/data_engineering/reports/cleaning_audit.md", "Every cleaning step, every table."),
            ("artifacts/data_engineering/reports/eda_report.md", "The nine analyses."),
            ("artifacts/data_engineering/figures/dashboard.png", "Composite EDA dashboard."),
        ],
        "figure": ("artifacts/data_engineering/figures/tyre_degradation.png",
                   "Tyre degradation by compound, from the real laps."),
        "example": ("file", "artifacts/data_engineering/reports/cleaning_audit.md", 30),
        "applications": [
            "Retrain on another race: `python scripts/fetch_real_session.py --year 2023 --event Monza "
            "--session R`, then rebuild.",
            "The audit trail answers \"what happened to my data?\" for any table.",
        ],
        "caveat": "Outlier capping has a visible side effect: **[[data_capped_laps]] laps** sit "
                  "exactly at the cap of [[data_cap_value]] s in the modelling table (mostly in- "
                  "and out-laps).",
    },
    # ------------------------------------------------------------------ 5
    {
        "number": 5,
        "title": "Feature Engineering & Selection",
        "tagline": "Candidate features in six blocks, a four-stage selection funnel, and a contract every model reads.",
        "package": "app/intelligence/features/",
        "run": "python scripts/build_features.py",
        "what": [
            "Practical 5 builds candidate features from the clean laps, defines the three targets, "
            "and selects **[[feat_reg]] features for lap time** and **[[feat_clf]] for the pit "
            "decision** over [[feat_rows]] modelling rows. The result is written as a *contract* "
            "— `f1_features_selected.csv` plus `feature_metadata.json` — that Practicals 6–10 "
            "read instead of hard-coding feature lists.",
            "**Why:** two rules protect every later result. **No leakage:** sector times sum to "
            "the lap time (measured by `leakage_evidence`), and speed traps and `IsPersonalBest` "
            "are only known after the lap, so they are excluded before anything is built. **No "
            "scaling here:** scaling on the full dataset would leak its statistics into every "
            "later cross-validation fold, so it is left to Practical 6, inside each fold.",
        ],
        "steps": [
            ("Targets",
             "`add_targets` adds `target_laptime`, `target_pit_next_lap` (1 when the stint index "
             "rises on the next lap — the lap *before* the stop, which is the decision point) and "
             "a fuel-corrected lap time from a fitted linear fuel coefficient."),
            ("Six feature blocks",
             "`engineer_features`, with a recorded reason for each feature: **A** tyre/stint "
             "(tyre life, its square, compound × tyre-life interactions); **B** race progress; "
             "**C** the driver's own recent pace as gap-to-field, strictly causal via `shift(1)` "
             "before rolling; **D** field pace (previous-lap median and trend); **E** environment "
             "(air temp, track–air delta, humidity, wind, a mean-centred temperature × tyre-life "
             "interaction); **F** one-hot compound, team and driver with one reference level dropped."),
            ("Trim the warm-up",
             "`trim_warmup` drops opening laps where history features are undefined, keeping the "
             "two frames aligned row for row."),
            ("Funnel stage 1–3",
             "Near-zero variance (`near_zero_variance`), then correlation above 0.95 — mutual "
             "information decides which of a pair survives (`prune_correlated`) — then VIF ≤ 10 "
             "on continuous columns only (`prune_by_vif`; a rare dummy always looks collinear with "
             "its group). On this data the funnel dropped: [[feat_funnel]]."),
            ("Funnel stage 4 — rank and choose K",
             "`rank_features` averages three notions of importance by rank: mutual information, "
             "random-forest importance and the L1 (Lasso / L1-logistic) coefficient. "
             "`rank_with_stability` records how many lap-forward folds put each feature in their "
             "top 12. `choose_regression_features` sweeps K and keeps the **smallest K within 2% of "
             "the best CV MAE**; classification fixes K = 8 because ~5% positives make a sweep noise."),
            ("Write the contract",
             "`build_metadata` records the selected features per target, each feature's "
             "provenance, which features need scaling, the leakage exclusions and the validation "
             "strategy; `FeatureContract` gives later code a typed view over it."),
        ],
        "tech": [
            ("scikit-learn", "Mutual information, RandomForest, LassoCV / L1 logistic regression, "
                             "LinearRegression (VIF and fuel coefficient)."),
            ("pandas groupby / shift / rolling", "Per-driver, strictly causal history features."),
            ("Lap-forward expanding splits", "The same time-ordered validation shape Practical 6 uses."),
            ("Jupyter notebook", "`docs/notebooks/task5_feature_engineering.ipynb` imports this "
                                 "module as a step-by-step walkthrough."),
        ],
        "functions": [
            ("app/intelligence/features/build.py", "leakage_evidence", "Proves sector times are leakage."),
            ("app/intelligence/features/build.py", "add_targets", "The three targets."),
            ("app/intelligence/features/build.py", "engineer_features", "Blocks A–F with provenance."),
            ("app/intelligence/features/build.py", "prune_correlated", "Correlation pruning decided by MI."),
            ("app/intelligence/features/build.py", "prune_by_vif", "Iterative VIF elimination."),
            ("app/intelligence/features/build.py", "rank_features", "Three importance measures, rank-averaged."),
            ("app/intelligence/features/build.py", "choose_regression_features", "Smallest K within 2%."),
            ("app/intelligence/features/contract.py", "FeatureContract", "Typed view of the contract."),
        ],
        "snippets": [
            ("app/intelligence/features/build.py", "add_targets", "Defining the pit target at the decision point."),
            ("app/intelligence/features/build.py", "engineer_features",
             "Block C — causal pace features (excerpt).", ("# Block C", "# Block D")),
            ("app/intelligence/features/build.py", "prune_by_vif", "VIF elimination."),
        ],
        "outputs": [
            ("data/processed/f1_features_selected.csv", "Identifiers + selected features + targets."),
            ("data/processed/feature_metadata.json", "The contract later practicals read."),
            ("artifacts/feature_engineering/feature_engineering_report.md", "The funnel, written up."),
            ("artifacts/feature_engineering/feature_importance_report.md", "Rankings and fold stability."),
            ("artifacts/feature_engineering/correlation_matrix.png", "Post-funnel correlation heatmap."),
        ],
        "figure": ("artifacts/feature_engineering/correlation_matrix.png",
                   "Correlation between the features that survived the funnel."),
        "example": ("json", "data/processed/feature_metadata.json", "selection_funnel"),
        "applications": [
            "Any model can be added later by reading `feature_metadata.json` — no feature list to copy.",
            "The provenance strings double as a data dictionary for the dashboard's feature inputs.",
        ],
    },
    # ------------------------------------------------------------------ 6
    {
        "number": 6,
        "title": "Machine Learning",
        "tagline": "[[ml_n_models]] classical models, time-aware validation, tuned thresholds and a generalisation guard.",
        "package": "app/intelligence/ml/",
        "run": "python scripts/build_all.py   # or: python -c \"from app.intelligence.ml import pipeline; pipeline.train_all()\"",
        "what": [
            "Practical 6 trains and compares classical models for both targets: linear/logistic "
            "regression, decision tree, random forest, SVR/SVM and XGBoost. The selected lap-time "
            "model is **`[[ml_reg_best]]`** (test MAE [[ml_reg_mae]] s, R² [[ml_reg_r2]]) and the "
            "selected pit model is **`[[ml_clf_best]]`** (test PR-AUC [[ml_clf_prauc]] at a tuned "
            "threshold of [[ml_clf_threshold]]).",
            "**Why it is built the way it is:** the data is a time-ordered panel, so a shuffled "
            "K-fold would train on lap 50 and validate on lap 10. Every model is validated on "
            "**expanding-window, lap-forward folds** and tested once on a **chronological holdout** "
            "(the last 20% of laps). Scalers are fitted inside each fold's pipeline. Pit stops are "
            "rare, so ROC-AUC alone would flatter the classifiers; PR-AUC, F1 and a tuned "
            "threshold are reported alongside it.",
        ],
        "steps": [
            ("Load and police the contract",
             "`data_contract.load_and_validate` checks that the selected columns exist, that no "
             "leakage column is present, and that there are no missing values; `build_task_frame` "
             "makes one frame per target."),
            ("Split by time",
             "`chronological_holdout` reserves the last 20% of unique laps, keeping whole laps on "
             "one side. `expanding_window_folds` cuts the remaining laps into 5 blocks: fold *i* "
             "trains on blocks 0..i−1 and validates on block *i*."),
            ("Preprocess inside the pipeline",
             "`build_preprocessor` is a `ColumnTransformer`: continuous features are median-imputed "
             "and — only for models that need it (linear, SVM) — standard-scaled; binary indicators "
             "pass through unscaled. Because it sits inside the sklearn `Pipeline`, it is re-fitted "
             "on each fold's training rows."),
            ("Tune",
             "`run_expanding_window_search` is a custom grid search rather than `GridSearchCV`, so a "
             "fold that fails (e.g. a single class in training) is recorded as an error instead of "
             "aborting, and a fold where a metric is undefined is excluded from the mean, not "
             "counted as zero. The winner is refitted on the whole development set."),
            ("Tune the decision threshold",
             "`tune_threshold` picks the probability cut-off on **pooled out-of-fold** predictions "
             "(never the test set), maximising F1 by default; an expected-cost objective with a "
             "missed pit costing 5× an early stop is also supported. 0.5 is only optimal for "
             "balanced classes with equal error costs — neither holds here."),
            ("Select, with a guard",
             "`rank_regression_models` ranks by CV MAE; `_apply_generalisation_guard` overrides the "
             "CV winner only if its holdout R² is negative while another model's is positive — and "
             "says so in the report."),
            ("Persist and register",
             "Pipelines are saved with joblib; `write_registry` records every model in "
             "`model_registry.json`, which the API reads to know which models exist."),
        ],
        "tech": [
            ("scikit-learn", "Estimators, `Pipeline`, `ColumnTransformer`, metrics, `clone`."),
            ("XGBoost", "Gradient boosting; optional at import time, with `scale_pos_weight` "
                        "standing in for `class_weight`."),
            ("joblib", "scikit-learn's recommended persistence format for fitted pipelines."),
            ("matplotlib", "ROC / PR curves, confusion matrix, residuals, importance — each with a "
                           "JSON sidecar of the numbers that drew it."),
        ],
        "functions": [
            ("app/intelligence/ml/splits.py", "chronological_holdout", "Last 20% of laps as test."),
            ("app/intelligence/ml/splits.py", "expanding_window_folds", "Lap-forward CV folds."),
            ("app/intelligence/ml/preprocessing.py", "build_preprocessor", "Fold-safe scaling."),
            ("app/intelligence/ml/tuning.py", "run_expanding_window_search", "Custom grid search."),
            ("app/intelligence/ml/threshold.py", "tune_threshold", "Out-of-fold threshold tuning."),
            ("app/intelligence/ml/selection.py", "_apply_generalisation_guard", "CV-vs-holdout guard."),
            ("app/intelligence/ml/pipeline.py", "train_all", "The orchestrator."),
        ],
        "snippets": [
            ("app/intelligence/ml/splits.py", "expanding_window_folds", "Time-aware cross-validation."),
            ("app/intelligence/ml/threshold.py", "tune_threshold", "Choosing the cut-off on out-of-fold data."),
            ("app/intelligence/ml/selection.py", "_apply_generalisation_guard", "When CV and holdout disagree."),
        ],
        "outputs": [
            ("artifacts/metrics/model_comparison.json", "All models ranked, CV and test."),
            ("artifacts/metadata/model_registry.json", "Which models exist — read by the API."),
            ("artifacts/reports/model_selection_report.md", "The selection rule and result."),
            ("artifacts/figures/precision_recall_curves.png", "PR curves (pit events are rare)."),
            ("artifacts/figures/confusion_matrix.png", "At the tuned threshold, not 0.5."),
        ],
        "figure": ("artifacts/figures/precision_recall_curves.png",
                   "Precision–recall curves for the pit-decision models."),
        "example": ("ml_table",),
        "viva": [
            ("Why expanding-window validation instead of K-fold?",
             "The data is a time-ordered driver × lap panel. Random K-fold would train on later laps and "
             "validate on earlier ones, so every score would be optimistic. Lap-forward folds put the model in "
             "the situation a real strategy model is in."),
            ("What does a negative test R² mean here?",
             "The model does worse on the closing laps than predicting the training mean — a different fuel and "
             "tyre regime from training, not a bug. The generalisation guard stops such a model being selected "
             "when another one generalises."),
            ("Why tune the threshold instead of using 0.5?",
             "0.5 is only right for balanced classes with equal error costs. Pit stops are [[data_pit_events]] "
             "of [[feat_rows]] laps, so the cut-off is chosen on pooled out-of-fold predictions."),
            ("Extension — what do the quantum models show?",
             "Three PennyLane models on the noiseless `[[qml_simulator]]` simulator, on the same split and "
             "metrics: features PCA-reduced to [[qml_qubits]] qubits ([[qml_variance]]% of variance kept). The "
             "only defensible claim is a narrow comparison with classical models given the same inputs — not a "
             "quantum speed-up, not better models, and nothing about real (noisy) hardware."),
        ],
        "applications": [
            "`POST /api/ml/predict/laptime` and `/api/ml/predict/pit` serve the registered pipelines.",
            "The same holdout is reused by Practicals 7, 8 and 10, so results are directly comparable.",
        ],
        "caveat": "The holdout contains **[[dl_test_pos]] labelled pit stop**, so test precision, "
                  "recall and F1 for the pit decision are noise; the cross-validated figures are the "
                  "ones to trust.",
    },
    # ------------------------------------------------------------------ 7
    {
        "number": 7,
        "title": "Deep Learning",
        "tagline": "Two Keras networks, tuned on the same folds and tested on the same holdout as Practical 6.",
        "package": "app/intelligence/dl/",
        "run": "python -c \"from app.intelligence.dl import pipeline; pipeline.train_all()\"",
        "what": [
            "Practical 7 trains a multilayer perceptron per target and tests — rather than "
            "assumes — whether it beats Practical 6. The lap-time network ([[dl_reg_params]] "
            "parameters) reaches **test MAE [[dl_reg_mae]] s, R² [[dl_reg_r2]]**; the pit network "
            "([[dl_clf_params]] parameters) has test ROC-AUC [[dl_clf_roc]] at a tuned threshold "
            "of [[dl_clf_threshold]].",
            "**Why small networks:** there are only ~800 development rows and [[data_pit_events]] "
            "pit events in the whole dataset. Dropout on every hidden layer, L2 weight decay and "
            "early stopping are used together, and the overfitting verdict is computed from the "
            "training curves, not asserted.",
        ],
        "steps": [
            ("Runtime",
             "`prepare_dl_runtime` selects Keras 3's PyTorch backend before `keras` is imported — "
             "TensorFlow publishes no wheel for the Python version this project targets."),
            ("Build",
             "`_hidden_stack` stacks Dense(ReLU) + Dropout layers with an L2 regulariser. "
             "`build_regression_mlp` adds a linear head (MSE or Huber loss); "
             "`build_classification_mlp` adds a sigmoid head (binary cross-entropy)."),
            ("Class weights inside the loss",
             "On this Keras/PyTorch install, `fit(class_weight=...)` was measured *not* to weight "
             "the loss. `weighted_binary_crossentropy` applies the weights inside the loss instead, "
             "and a test asserts it moves the predictions."),
            ("Fit one fold",
             "`fit_fold` fits the feature scaler on the training rows only, optionally standardises "
             "the lap-time target, and trains with `EarlyStopping(patience=20, "
             "restore_best_weights=True)` on validation loss."),
            ("One-factor-at-a-time search",
             "`tuning.search` starts from a base network and, stage by stage, varies one "
             "hyperparameter (architecture, learning rate, batch size, dropout, optimizer, L2, loss "
             "— and, for the pit model, whether to class-weight at all) across the CV folds, keeping "
             "each stage's winner. Every configuration is logged to the hyperparameter report."),
            ("Diagnose the fit",
             "`_diagnose` applies stated rules to the saved model's curves: overfit if training MAE "
             "is below 60% of validation MAE (regression) or training PR-AUC exceeds validation by "
             "0.25 (classification); underfit if it has not learned the training data. Verdicts: "
             "lap time **[[dl_reg_verdict]]**, pit decision **[[dl_clf_verdict]]**."),
            ("Save and verify",
             "`persistence.save` recompiles with a standard loss (the custom loss would break the "
             "reload), writes the `.h5`, reloads it and refuses to finish unless the reloaded model "
             "reproduces the predictions."),
        ],
        "tech": [
            ("Keras 3 (PyTorch backend)", "High-level layers and training loop; backend-agnostic API."),
            ("PyTorch", "The tensor backend Keras runs on here."),
            ("scikit-learn StandardScaler", "Fold-local feature and target scaling."),
            ("HDF5 (.h5)", "The model format the task sheet requires; scalers saved with joblib."),
        ],
        "functions": [
            ("app/intelligence/dl/models.py", "weighted_binary_crossentropy", "Class weights in the loss."),
            ("app/intelligence/dl/models.py", "build_regression_mlp", "Lap-time network."),
            ("app/intelligence/dl/models.py", "build_classification_mlp", "Pit-decision network."),
            ("app/intelligence/dl/training.py", "fit_fold", "Scale, train, early-stop."),
            ("app/intelligence/dl/tuning.py", "search", "One-factor-at-a-time search."),
            ("app/intelligence/dl/pipeline.py", "_diagnose", "Over/underfitting verdict."),
            ("app/intelligence/dl/persistence.py", "save", "Save, reload, verify."),
        ],
        "snippets": [
            ("app/intelligence/dl/models.py", "weighted_binary_crossentropy", "A measured library pitfall, worked around."),
            ("app/intelligence/dl/training.py", "fit_fold", "Training one fold (excerpt).", ("set_seeds(seed)", "history = {")),
            ("app/intelligence/dl/pipeline.py", "_diagnose", "The fit verdict, as code."),
        ],
        "outputs": [
            ("artifacts/models/deep_learning/laptime/f1_dnn_model.h5", "Lap-time network."),
            ("artifacts/models/deep_learning/pit_decision/f1_dnn_model.h5", "Pit-decision network."),
            ("artifacts/deep_learning/evaluation_report.md", "Train/validation/test for both."),
            ("artifacts/deep_learning/hyperparameter_report.md", "Every configuration tried."),
            ("artifacts/deep_learning/laptime/loss_curve.png", "Loss curves, restored epoch marked."),
        ],
        "figure": ("artifacts/deep_learning/laptime/loss_curve.png",
                   "Lap-time network: training vs validation loss; the dashed line is the restored epoch."),
        "example": ("dl_table",),
        "viva": [
            ("Why is the lap-time target standardised but the binary features are not?",
             "The target sits around [[data_lap_mean]] s with a standard deviation of [[data_lap_sd]] s; a linear "
             "head initialised near zero, under L2 decay, struggles to reach that offset. The target is "
             "standardised on training rows and inverted before every metric. Scaling a 0/1 dummy would only "
             "cost interpretability."),
            ("How do you know the saved `.h5` is the model you trained?",
             "`persistence.save` recompiles with a standard loss (the custom class-weighted loss would break the "
             "reload), saves, reloads with `compile=False`, and refuses to continue unless the reloaded model "
             "reproduces the predictions within 1e-6. The result is recorded as `reload_verified`."),
            ("What would you change with more data?",
             "Hold out whole races rather than the tail of one; try sequence models over stint history; and drop "
             "the driver/team identity features to test whether the model generalises to new drivers."),
        ],
        "applications": [
            "`/api/dl/*` serves both networks; the strategy pipeline runs them on every request.",
            "Model comparison against Practical 6 on identical laps (`model_comparison.csv`).",
        ],
    },
    # ------------------------------------------------------------------ 8
    {
        "number": 8,
        "title": "Explainable AI",
        "tagline": "SHAP, LIME, counterfactuals, a trust score and per-group performance — for both model families.",
        "package": "app/intelligence/xai/",
        "run": "python -c \"from app.intelligence.xai import pipeline; pipeline.run_all()\"",
        "what": [
            "Practical 8 explains *why* the Practical 7 networks and Practical 6's selected models "
            "predict what they do, globally (which features matter overall) and locally (why "
            "this lap). It adds a trust score per prediction and a performance breakdown by "
            "driver, team and compound.",
            "**Why several methods:** each has a blind spot. SHAP's exact tree explainer is "
            "available only for tree models, so the network uses sampled KernelSHAP; LIME is a "
            "local linear surrogate that can disagree with SHAP; counterfactuals answer the "
            "engineer's actual question (\"how many more laps until it says pit?\"). Agreement "
            "between them becomes one component of the trust score.",
        ],
        "steps": [
            ("Load what to explain",
             "`loading.load_target` loads the network, the classical model and the same holdout. "
             "If a model is missing it raises `ExplainerUnavailableError` — it never substitutes a "
             "stand-in. `pick_representative_rows` chooses laps worth explaining (fastest, slowest, "
             "the actual pit lap, the one closest to the decision boundary, ...)."),
            ("SHAP",
             "`tree_shap` gives exact Shapley values for tree ensembles; `kernel_shap` samples them "
             "for the network over a k-means background of [[xai_bg_k]] points with "
             "nsamples=[[xai_nsamples]]. `global_ranking` is mean |SHAP|."),
            ("LIME and permutation importance",
             "`lime_analysis.explain_row` fits a local surrogate; `top_features` maps LIME's "
             "discretised conditions (`tyre_life > 12.00`) back to feature names so its top-3 can be "
             "compared with SHAP's. `permutation_importance` measures the score lost when a column "
             "is shuffled."),
            ("Physically consistent counterfactuals",
             "`tyre_age_scan` sweeps tyre age across the training range in 0.5-lap steps. "
             "`consistent_tyre_age_rows` recomputes every feature derived from tyre age (the "
             "compound interactions and the temperature interaction) so each probe is a lap that "
             "could exist — moving `tyre_life` alone would not be. DiCE runs as a time-bounded "
             "supplementary check."),
            ("Trust score",
             "`trust.compute` combines confidence (distance from the *tuned* threshold), agreement "
             "between the network and the classical model, SHAP–LIME top-3 Jaccard overlap, and the "
             "share of inputs inside the training 1st–99th percentile. Weights are "
             "[[xai_weights]]; a missing component is dropped and the rest renormalised, never "
             "scored as zero."),
            ("Stratify and narrate",
             "`stratification.stratify` and `fairness.assess` report error or recall per driver, "
             "team and compound, flagging small groups; `narrative.py` writes the plain-English "
             "sentence an engineer would read."),
        ],
        "tech": [
            ("SHAP", "TreeExplainer (exact) and KernelExplainer (model-agnostic, sampled)."),
            ("LIME", "Local linear surrogates, used as an independent second opinion."),
            ("DiCE (dice-ml)", "Diverse counterfactuals, under a wall-clock budget."),
            ("NumPy / pandas / matplotlib", "Scans, tables, waterfalls and stratification plots."),
        ],
        "functions": [
            ("app/intelligence/xai/shap_analysis.py", "tree_shap", "Exact SHAP for trees."),
            ("app/intelligence/xai/shap_analysis.py", "kernel_shap", "Sampled SHAP for the network."),
            ("app/intelligence/xai/lime_analysis.py", "top_features", "LIME conditions → feature names."),
            ("app/intelligence/xai/counterfactual.py", "consistent_tyre_age_rows", "Valid what-if laps."),
            ("app/intelligence/xai/counterfactual.py", "tyre_age_scan", "Where the decision flips."),
            ("app/intelligence/xai/trust.py", "compute", "The trust score."),
            ("app/intelligence/xai/live.py", "explain_feature_row", "Explain one live API request."),
        ],
        "snippets": [
            ("app/intelligence/xai/counterfactual.py", "consistent_tyre_age_rows",
             "Keeping a counterfactual physically possible."),
            ("app/intelligence/xai/trust.py", "compute", "Trust score with renormalisation."),
        ],
        "outputs": [
            ("artifacts/xai/SHAP_Report.md", "Global and local SHAP for both model families."),
            ("artifacts/xai/LIME_Report.md", "LIME explanations compared with SHAP."),
            ("artifacts/xai/Counterfactual_Report.md", "Tyre-age scans and DiCE."),
            ("artifacts/xai/Trust_Score_Report.md", "Formula, weights and every component."),
            ("artifacts/xai/Fairness_Report.md", "Performance by driver, team, compound."),
            ("artifacts/xai/shap/target_laptime_shap_summary.png", "SHAP summary, lap time."),
        ],
        "figure": ("artifacts/xai/shap/target_laptime_shap_summary.png",
                   "SHAP summary for the lap-time network."),
        "example": ("trust_table",),
        "viva": [
            ("How do you know a SHAP explanation is not misleading?",
             "The tree model uses exact `TreeExplainer`; the network's sampled `KernelExplainer` values are "
             "labelled approximate (compare rank and sign, not magnitude); LIME runs on the same rows as a second "
             "opinion and its `local_r2` is reported. SHAP explains what the model does, not what is true — which "
             "is why the fairness assessment exists."),
            ("Why SHAP and LIME, and why two SHAP explainers?",
             "Exact when possible, model-agnostic when necessary. SHAP–LIME disagreement is a signal, and the "
             "trust score uses it."),
            ("Is the trust score principled or arbitrary?",
             "Its components are real failure modes — distance from the tuned threshold, model agreement, "
             "explanation agreement, familiar inputs. Its weights ([[xai_weights]]) are a documented judgement, "
             "not a fit, and are exposed in `trust.WEIGHTS` so they can be challenged."),
            ("Why does DiCE have a time budget?",
             "Its random search can sample for a long time without finding an opposite-class row when pit laps "
             "are rare. It is capped at 60 s, starts from the row closest to the boundary, and stays inside each "
             "feature's training range; running out of budget is reported, not hidden."),
            ("A counterfactual says \"not reachable\" — is that a failure?",
             "It is a result: sweeping tyre age across its whole training range never crosses the threshold, so "
             "the decision depends on combinations of features, not tyre age alone."),
        ],
        "applications": [
            "`POST /api/strategy/predict?explain=true` attaches SHAP factors and a trust score to a live recommendation.",
            "The Explainability page of the dashboard reads only `artifacts/xai/`.",
        ],
        "caveat": "Honest findings kept in the reports: LIME ranks `track_status` — constant in this "
                  "session — first on [[xai_lime_ts]] of [[xai_lime_n]] explained laps, a known "
                  "artefact of LIME's discretisation. The explainability validation of [[eval_date]] found the per-row top feature "
                  "`[[xai_faith_top]]` moved the prediction by [[xai_faith_top_delta]] s on ablation "
                  "versus [[xai_faith_rand_delta]] s for a random feature.",
    },
    # ------------------------------------------------------------------ 9
    {
        "number": 9,
        "title": "System Integration & Deployment",
        "tagline": "One strategy pipeline that chains every practical, behind an API, a dashboard and Docker.",
        "package": "app/services/ · app/api/ · frontend/",
        "run": "./run.sh        # or: docker compose build && docker compose up",
        "what": [
            "Practical 9 turns eight separate capabilities into one product. A race state goes in "
            "to `POST /api/strategy/predict`; a single recommendation (STAY OUT / PIT NOW / PIT IN "
            "N LAPS) comes out, with the evidence from every stage attached.",
            "**Why an API and a separate frontend** (instead of the lab's suggested Streamlit): the "
            "same endpoints back the dashboard, the tests and `scripts/demo_predict.py`, and a page "
            "that invents a number instead of fetching one is obvious.",
        ],
        "steps": [
            ("1 · Validate",
             "`RaceStateRequest` (Pydantic) rejects inconsistent states — `current_lap ≤ "
             "total_laps`, `tyre_age ≤ current_lap` — and the router checks driver, team, compound "
             "and model choices against the live dataset and registry. Failure is a 422."),
            ("2 · Construct features",
             "`build_feature_row` maps the race state onto the exact selected features: tyre life, "
             "race progress, compound one-hots and interactions, driver/team one-hots; anything it "
             "cannot derive is median-filled and *listed* as approximated. `_flag_out_of_range` "
             "warns when a derived value is outside the training range."),
            ("3–5 · ML, rules, DL",
             "`_run_ml` predicts with the registered Practical 6 pipelines; `_run_expert_system` "
             "runs Practical 2's forward chaining on a 16-key input dict; `_run_dl` runs the "
             "Practical 7 networks at their tuned threshold (never 0.5)."),
            ("6 · Search",
             "`_run_search` runs A* (Practical 3) over the remaining laps from the current compound."),
            ("7 · Recommend",
             "`_combine_recommendation`: a fired expert-system `pit_decision` wins outright; "
             "otherwise ML and DL are compared — agreement is used, disagreement is reported "
             "(`disagreement: true`) and falls back to ML. A pit verdict is refined into PIT NOW / "
             "PIT IN N LAPS from the A* plan's first pit action."),
            ("8 · Explain (opt-in) and report",
             "With `explain=true`, Practical 8's `explain_feature_row` attaches SHAP factors and a "
             "trust score. `POST /api/strategy/report` renders the same analysis as a Markdown or "
             "HTML briefing."),
            ("Serve and deploy",
             "`app/api/main.py` registers the routers, restricts CORS to an allow-list, and mounts "
             "only the artifact folders the frontend reads (model files are never served). The "
             "Next.js dashboard has a page per area; `Dockerfile`, `frontend/Dockerfile` and "
             "`docker-compose.yml` build code-only images with `artifacts/` and `data/` bind-mounted."),
        ],
        "tech": [
            ("FastAPI + Pydantic", "Typed endpoints and request validation with automatic 422s."),
            ("Uvicorn", "ASGI server."),
            ("Next.js + TypeScript + Tailwind", "The dashboard; every value is fetched from the API."),
            ("Docker / Compose", "Reproducible backend and frontend images."),
        ],
        "functions": [
            ("app/api/schemas.py", "RaceStateRequest", "Validated race-state input."),
            ("app/services/feature_approximation.py", "build_feature_row", "Race state → feature row."),
            ("app/services/strategy_service.py", "_run_expert_system", "Practical 2 on a live request."),
            ("app/services/strategy_service.py", "_run_search", "Practical 3 on the remaining laps."),
            ("app/services/strategy_service.py", "_combine_recommendation", "The recommendation engine."),
            ("app/services/strategy_service.py", "run_strategy_analysis", "The 8-stage pipeline."),
            ("app/services/strategy_report.py", "render_markdown", "Downloadable briefing."),
        ],
        "snippets": [
            ("app/services/strategy_service.py", "run_strategy_analysis",
             "The pipeline order (excerpt).",
             ("def run_strategy_analysis", "pit_decision = expert")),
            ("app/services/strategy_service.py", "_combine_recommendation",
             "How rules, ML and DL are combined (excerpt).",
             ("ml_pit, dl_pit =", "if pit_now:")),
        ],
        "outputs": [
            ("app/api/main.py", "FastAPI application."),
            ("frontend/app/strategy/page.tsx", "Race Strategy page."),
            ("docker-compose.yml", "Two-service deployment."),
            ("docs/source/deployment_guide.html", "Install, run, Docker, troubleshooting (also in Extras)."),
        ],
        "example": ("strategy_example",),
        "applications": [
            "A pit-wall assistant: one call returns a decision plus the evidence behind it.",
            "`scripts/demo_predict.py` and the tests drive the same endpoints the dashboard uses.",
        ],
    },
    # ------------------------------------------------------------------ 10
    {
        "number": 10,
        "title": "Evaluation, Responsible AI & Documentation",
        "tagline": "Seven generated evaluation sections, a responsible-AI view, and the project documents.",
        "package": "scripts/evaluate_system.py · app/services/rai_service.py · docs/",
        "run": "python scripts/evaluate_system.py   # --skip-pytest for a quick run",
        "what": [
            "Practical 10 evaluates the integrated system and writes every result to "
            "`artifacts/evaluation/*.json`, so the final report reads numbers from files, never "
            "from text typed by hand. The latest run ([[eval_date]]) passed **[[eval_passed]] of "
            "[[eval_total]] tests** at [[eval_coverage]]% line coverage; the strategy endpoint "
            "answered in [[eval_p50]] ms median, [[eval_p95]] ms at p95.",
            "**Why a script instead of a write-up:** a script can be re-run after every change and "
            "can fail loudly. Each section is independent; one that cannot run reports "
            "`\"available\": false` with a reason instead of an invented number.",
        ],
        "steps": [
            ("1 · Functional testing",
             "`functional_testing` runs the real pytest suite with coverage and counts results from "
             "its `--junitxml` output — the plain summary line disappears when coverage reports are "
             "requested, so stdout is not scraped."),
            ("2 · Performance",
             "`performance_evaluation` confirms empirically that Practicals 6 and 7 use the same "
             "holdout (`_verify_shared_holdout`), consolidates every model's metrics, and times "
             "[[eval_n_requests]] strategy requests with and without explanations."),
            ("3 · Domain validation",
             "`domain_validation` replays the real race lap by lap through the API and compares pit "
             "calls with the stops drivers actually made: [[eval_dom_laps]] laps replayed, pit-call "
             "precision [[eval_dom_precision]], recall [[eval_dom_recall]]. It also checks which "
             "rules can fire through the live pipeline (`_rule_base_reachability`)."),
            ("4 · Explainability validation",
             "`explainability_validation` checks SHAP top-feature stability, a faithfulness "
             "ablation (does removing the top feature move the prediction more than a random one?) "
             "and traceability."),
            ("5 · Responsible AI",
             "`responsible_ai_assessment` measures robustness (how often small perturbations flip "
             "the recommendation), per-team/driver/compound performance, transparency and human "
             "oversight. `rai_service.py` serves a live version for the dashboard's Responsible-AI "
             "page, with confidence shown only where a model natively produces one (probabilities, "
             "tree spread, Monte-Carlo dropout)."),
            ("6–7 · Deployment readiness and future work",
             "`deployment_readiness` checks Dockerfiles, the health endpoint, the test result and "
             "scans for hard-coded metrics; `future_enhancements` records the plan."),
            ("Documentation",
             "User manual, SRS, SDD, testing report and deployment guide live in `docs/` and are "
             "consolidated in this study guide's first tab."),
        ],
        "tech": [
            ("pytest + pytest-cov", "The test suite and coverage; JUnit XML for exact counts."),
            ("FastAPI TestClient", "In-process requests for latency and backtests."),
            ("resource / time", "Peak memory and request timing."),
            ("matplotlib", "Latency, robustness and fairness figures."),
        ],
        "functions": [
            ("scripts/evaluate_system.py", "functional_testing", "Runs and counts the test suite."),
            ("scripts/evaluate_system.py", "_verify_shared_holdout", "Same holdout, verified."),
            ("scripts/evaluate_system.py", "domain_validation", "Backtest against the real race."),
            ("scripts/evaluate_system.py", "_rule_base_reachability", "Which rules can actually fire."),
            ("scripts/evaluate_system.py", "responsible_ai_assessment", "Robustness and fairness."),
            ("scripts/evaluate_system.py", "_sanitize_nan", "NaN → null for valid JSON."),
            ("app/services/rai_service.py", "summary", "Live responsible-AI payload."),
        ],
        "snippets": [
            ("scripts/evaluate_system.py", "main", "Seven independent sections, seven files."),
            ("scripts/evaluate_system.py", "_sanitize_nan", "A real bug the evaluation found."),
        ],
        "outputs": [
            ("artifacts/evaluation/test_results.json", "Test counts and coverage."),
            ("artifacts/evaluation/performance.json", "Latency, holdout check, model table."),
            ("artifacts/evaluation/domain_validation.json", "The race backtest."),
            ("artifacts/evaluation/responsible_ai.json", "Robustness, fairness, oversight."),
            ("artifacts/evaluation/latency_distribution.png", "Request latency histogram."),
        ],
        "figure": ("artifacts/evaluation/latency_distribution.png",
                   "Latency of [[eval_n_requests]] strategy requests."),
        "example": ("eval_table",),
        "applications": [
            "Re-run after any change to get a fresh, comparable evaluation.",
            "`deployment_readiness.json` is a pre-release checklist a machine can verify.",
        ],
    },
]


# ---------------------------------------------------------------------------
# Showcase slides. ``kind`` picks a layout in build_study_guide.render_slide.
# ---------------------------------------------------------------------------
SLIDES: list[dict] = [
    {"kind": "title",
     "eyebrow": "Computational Intelligence Lab · 10 practicals",
     "title": "F1 Race Strategy Intelligence",
     "lead": "When should a Formula 1 driver pit, and what will the next lap cost?",
     "foot": "One real race · [[data_year]] [[data_event]] GP · [[data_n_laps]] laps · [[data_n_drivers]] drivers"},
    {"kind": "bullets",
     "title": "The problem",
     "lead": "Pit strategy is a trade-off under uncertainty.",
     "bullets": [
         "Softer tyres are faster but degrade sooner; every stop costs ~20 s in the pit lane.",
         "Safety cars, weather and rivals change the right answer mid-race.",
         "Two predictions run through the whole project: **lap time** (regression) and "
         "**does this driver pit at the end of this lap?** (classification).",
     ]},
    {"kind": "bullets",
     "title": "Objectives",
     "bullets": [
         "Attack one question three ways: **symbolic** reasoning, **machine learning**, and "
         "**explainability**.",
         "Build all ten practicals as **one application**, not ten folders.",
         "**No leakage, no random splits, no made-up numbers** — every value comes from a "
         "generated artifact.",
         "Integrate everything into a single recommendation, served by an API and a dashboard.",
     ]},
    {"kind": "pipeline", "title": "Architecture"},
    {"kind": "grid", "title": "Ten practicals, one platform"},
    {"kind": "tech", "title": "Technologies"},
    {"kind": "stats", "title": "Symbolic AI — Practicals 1–3",
     "stats": [
         ("[[kr_entities]]", "ontology entities", "[[kr_relationships]] relationships, OWL + RDF"),
         ("[[es_rules]]", "expert-system rules", "forward & backward chaining"),
         ("[[search_astar_cost]] s", "optimal race time", "A* = UCS, [[search_astar_nodes]] vs [[search_ucs_nodes]] nodes expanded"),
     ]},
    {"kind": "stats", "title": "Data & features — Practicals 4–5",
     "stats": [
         ("[[data_n_laps]]", "real laps fetched", "FastF1, [[data_year]] [[data_event]]"),
         ("[[feat_rows]]", "modelling rows", "after the warm-up trim"),
         ("[[feat_reg]] / [[feat_clf]]", "selected features", "lap time / pit decision"),
     ]},
    {"kind": "bullets",
     "title": "Implementation — the rules we held to",
     "bullets": [
         "**No leakage.** Sector times sum to the lap time — excluded, and a test fails if one reaches a training matrix.",
         "**Time-aware validation.** Expanding-window lap-forward folds and a chronological holdout; never a shuffled K-fold.",
         "**Scalers inside each fold**, never fitted on the whole dataset.",
         "**Tuned thresholds.** Pit cut-offs chosen on out-of-fold predictions, because 0.5 is wrong for 4.8% positives.",
         "**Honest gaps.** When a metric is undefined the UI and reports say so, and why.",
     ]},
    {"kind": "workflow", "title": "Workflow — one strategy request"},
    {"kind": "results_laptime", "title": "Results — lap time (same [[dl_test_n]] held-out laps)"},
    {"kind": "results_pit", "title": "Results — pit decision"},
    {"kind": "results_search", "title": "Results — search algorithms"},
    {"kind": "stats", "title": "Explainability & evaluation — Practicals 8–10",
     "stats": [
         ("[[eval_passed]]/[[eval_total]]", "tests passing", "[[eval_coverage]]% line coverage"),
         ("[[eval_p50]] ms", "median strategy request", "p95 [[eval_p95]] ms over [[eval_n_requests]] calls"),
         ("[[xai_trust_reg]] / [[xai_trust_clf]]", "mean trust score", "lap time / pit decision, all test laps"),
     ]},
    {"kind": "bullets",
     "title": "Limitations — stated, not hidden",
     "bullets": [
         "One race: the holdout holds **[[dl_test_pos]] labelled pit stop**, so pit precision/recall on test are noise.",
         "Backtest pit-call precision is **[[eval_dom_precision]]** (recall [[eval_dom_recall]]) — the engine over-calls stops.",
         "**[[es_unreachable]] of [[es_rules]] rules** cannot fire through the live pipeline's inputs.",
         "The lap-time network is flagged **overfitting** by its own curve rules.",
         "Kaggle-style history tables are synthetic; only the lap data is real.",
     ],
     "note": "Every figure on this slide is read from artifacts/ when the guide is built; the reports discuss each one."},
    {"kind": "title",
     "eyebrow": "Try it",
     "title": "./run.sh",
     "lead": "Builds what is missing, starts the API and the dashboard, and opens it.",
     "foot": "Study guide: docs/study_guide.html · Rebuild it: python scripts/build_study_guide.py"},
]
