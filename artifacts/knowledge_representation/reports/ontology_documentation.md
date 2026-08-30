# Ontology Documentation

_Generated 2026-08-30 11:23 UTC._

## Overview

The Formula 1 ontology is an OWL 2 ontology generated programmatically from the declarative domain schema (`f1kr.schema`). It formalises the racing domain as classes (entities), object properties (relationships) and data properties (attributes).

* **Base IRI:** `http://f1kr.org/ontology/formula1#`
* **Classes:** 73
* **Object properties:** 29
* **Data properties:** 90

## Class hierarchy by category

### CarState
- **FuelLoad** — Estimated fuel mass on board.
- **EngineMode** — Power-unit deployment mode.
- **ERS** — Energy Recovery System state.
- **DRS** — Drag Reduction System state.

### Competition
- **Season** — A Formula One championship year.
- **Race** — A single championship round / event weekend.
- **GrandPrix** — The recurring named event (e.g. 'Monaco Grand Prix') across seasons.
- **Circuit** — A racing venue / track.

### Competitor
- **Driver** — A racing driver.
- **Constructor** — A constructor / entrant team.
- **Team** — The operational team entity that fields cars for a constructor.
- **Car** — The chassis + power unit fielded by a team.

### Decision
- **Strategy** — A recommended race-strategy plan (pit/tyre/fuel decisions).
- **Prediction** — A model prediction over race outcomes.
- **Recommendation** — An actionable recommendation surfaced to the race engineer.
- **Optimization** — An optimisation result over the strategy search space.

### PitStrategy
- **PitStop** — A pit-lane stop performed by a driver.
- **PitWindow** — The lap range within which a pit stop is strategically optimal.
- **Stint** — A continuous run on one set of tyres between pit stops.

### RaceControl
- **TrackStatus** — Overall track status flag state.
- **SafetyCar** _(subclass of TrackStatus)_ — Full safety-car deployment.
- **VirtualSafetyCar** _(subclass of TrackStatus)_ — Virtual safety-car period.
- **YellowFlag** _(subclass of TrackStatus)_ — Yellow-flag caution.
- **RedFlag** _(subclass of TrackStatus)_ — Session-stopping red flag.
- **BlueFlag** _(subclass of TrackStatus)_ — Blue flag (yield to leaders).
- **RaceControlMessage** — An official race-control message.
- **Incident** — An on-track incident under investigation.
- **Penalty** — A sporting penalty applied to a driver.

### Result
- **Result** — A driver's classified result in a race.
- **Position** — A position/order value.
- **Points** — Championship points value.
- **Championship** — A championship classification context.
- **DriverStanding** _(subclass of Championship)_ — A driver's championship standing snapshot.
- **ConstructorStanding** _(subclass of Championship)_ — A constructor's championship standing snapshot.

### Session
- **Session** — An on-track session within a race weekend.
- **PracticeSession** _(subclass of Session)_ — A free-practice session.
- **Qualifying** _(subclass of Session)_ — A qualifying session.
- **Sprint** _(subclass of Session)_ — A sprint race session.

### Telemetry
- **Telemetry** — A telemetry sample stream for a lap.
- **Speed** _(subclass of Telemetry)_ — Instantaneous car speed.
- **Throttle** _(subclass of Telemetry)_ — Throttle application.
- **Brake** _(subclass of Telemetry)_ — Brake application.
- **RPM** _(subclass of Telemetry)_ — Engine revolutions per minute.
- **Gear** _(subclass of Telemetry)_ — Selected gear.

### Timing
- **Lap** — A single completed lap by a driver in a session.
- **Sector** — One of the three timing sectors of a lap.
- **SectorTime** — The measured time for a sector.
- **LapTime** — A recorded lap-time measurement.
- **FastestLap** — The fastest lap of a driver in a race.
- **Gap** — Time gap to a reference car.
- **Interval** — Interval to the car directly ahead.
- **Overtake** — An on-track position change event.

### Tyre
- **TyreConcept** — Abstract root for tyre-related concepts.
- **TyreCompound** _(subclass of TyreConcept)_ — A tyre compound (Soft/Medium/Hard/Inter/Wet).
- **TyreLife** _(subclass of TyreConcept)_ — Accumulated age/wear of a tyre set.

### Weather
- **Weather** — Weather conditions during a session.
- **TrackTemperature** _(subclass of Weather)_ — Track surface temperature.
- **AirTemperature** _(subclass of Weather)_ — Ambient air temperature.
- **Humidity** _(subclass of Weather)_ — Relative humidity.
- **WindSpeed** _(subclass of Weather)_ — Wind speed.
- **RainProbability** _(subclass of Weather)_ — Probability of rain.

## Cardinality encoding

Relationship cardinalities are encoded as OWL property characteristics:

| Cardinality | OWL characteristic |
|-------------|--------------------|
| `*..1` (many-to-one) | FunctionalProperty |
| `1..1` (one-to-one) | FunctionalProperty + InverseFunctionalProperty |
| `1..*` (one-to-many) | (plain object property) |
| `*..*` (many-to-many) | (plain object property) |

