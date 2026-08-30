# Entity Table

_Generated 2026-08-30 11:23 UTC — 61 entities._

| # | Entity | Category | Source | Parent | #Attrs | Description |
|---|--------|----------|--------|--------|--------|-------------|
| 1 | `Season` | Competition | kaggle | — | 2 | A Formula One championship year. |
| 2 | `Race` | Competition | both | — | 3 | A single championship round / event weekend. |
| 3 | `GrandPrix` | Competition | kaggle | — | 1 | The recurring named event (e.g. 'Monaco Grand Prix') across seasons. |
| 4 | `Circuit` | Competition | both | — | 6 | A racing venue / track. |
| 5 | `Driver` | Competitor | both | — | 7 | A racing driver. |
| 6 | `Constructor` | Competitor | both | — | 3 | A constructor / entrant team. |
| 7 | `Team` | Competitor | fastf1 | — | 1 | The operational team entity that fields cars for a constructor. |
| 8 | `Car` | Competitor | fastf1 | — | 2 | The chassis + power unit fielded by a team. |
| 9 | `Session` | Session | both | — | 2 | An on-track session within a race weekend. |
| 10 | `PracticeSession` | Session | fastf1 | Session | 0 | A free-practice session. |
| 11 | `Qualifying` | Session | both | Session | 3 | A qualifying session. |
| 12 | `Sprint` | Session | fastf1 | Session | 0 | A sprint race session. |
| 13 | `Lap` | Timing | both | — | 3 | A single completed lap by a driver in a session. |
| 14 | `Sector` | Timing | fastf1 | — | 1 | One of the three timing sectors of a lap. |
| 15 | `SectorTime` | Timing | fastf1 | — | 1 | The measured time for a sector. |
| 16 | `LapTime` | Timing | kaggle | — | 1 | A recorded lap-time measurement. |
| 17 | `FastestLap` | Timing | kaggle | — | 2 | The fastest lap of a driver in a race. |
| 18 | `Gap` | Timing | fastf1 | — | 1 | Time gap to a reference car. |
| 19 | `Interval` | Timing | fastf1 | — | 1 | Interval to the car directly ahead. |
| 20 | `Overtake` | Timing | fastf1 | — | 1 | An on-track position change event. |
| 21 | `PitStop` | PitStrategy | both | — | 3 | A pit-lane stop performed by a driver. |
| 22 | `PitWindow` | PitStrategy | fastf1 | — | 2 | The lap range within which a pit stop is strategically optimal. |
| 23 | `Stint` | PitStrategy | fastf1 | — | 3 | A continuous run on one set of tyres between pit stops. |
| 24 | `TyreConcept` | Tyre | fastf1 | — | 0 | Abstract root for tyre-related concepts. |
| 25 | `TyreCompound` | Tyre | fastf1 | TyreConcept | 2 | A tyre compound (Soft/Medium/Hard/Inter/Wet). |
| 26 | `TyreLife` | Tyre | fastf1 | TyreConcept | 2 | Accumulated age/wear of a tyre set. |
| 27 | `Weather` | Weather | fastf1 | — | 0 | Weather conditions during a session. |
| 28 | `TrackTemperature` | Weather | fastf1 | Weather | 1 | Track surface temperature. |
| 29 | `AirTemperature` | Weather | fastf1 | Weather | 1 | Ambient air temperature. |
| 30 | `Humidity` | Weather | fastf1 | Weather | 1 | Relative humidity. |
| 31 | `WindSpeed` | Weather | fastf1 | Weather | 1 | Wind speed. |
| 32 | `RainProbability` | Weather | fastf1 | Weather | 1 | Probability of rain. |
| 33 | `FuelLoad` | CarState | fastf1 | — | 1 | Estimated fuel mass on board. |
| 34 | `EngineMode` | CarState | fastf1 | — | 1 | Power-unit deployment mode. |
| 35 | `ERS` | CarState | fastf1 | — | 1 | Energy Recovery System state. |
| 36 | `DRS` | CarState | fastf1 | — | 1 | Drag Reduction System state. |
| 37 | `Telemetry` | Telemetry | fastf1 | — | 0 | A telemetry sample stream for a lap. |
| 38 | `Speed` | Telemetry | fastf1 | Telemetry | 1 | Instantaneous car speed. |
| 39 | `Throttle` | Telemetry | fastf1 | Telemetry | 1 | Throttle application. |
| 40 | `Brake` | Telemetry | fastf1 | Telemetry | 1 | Brake application. |
| 41 | `RPM` | Telemetry | fastf1 | Telemetry | 1 | Engine revolutions per minute. |
| 42 | `Gear` | Telemetry | fastf1 | Telemetry | 1 | Selected gear. |
| 43 | `TrackStatus` | RaceControl | fastf1 | — | 0 | Overall track status flag state. |
| 44 | `SafetyCar` | RaceControl | fastf1 | TrackStatus | 0 | Full safety-car deployment. |
| 45 | `VirtualSafetyCar` | RaceControl | fastf1 | TrackStatus | 0 | Virtual safety-car period. |
| 46 | `YellowFlag` | RaceControl | fastf1 | TrackStatus | 0 | Yellow-flag caution. |
| 47 | `RedFlag` | RaceControl | fastf1 | TrackStatus | 0 | Session-stopping red flag. |
| 48 | `BlueFlag` | RaceControl | fastf1 | TrackStatus | 0 | Blue flag (yield to leaders). |
| 49 | `RaceControlMessage` | RaceControl | fastf1 | — | 2 | An official race-control message. |
| 50 | `Incident` | RaceControl | fastf1 | — | 1 | An on-track incident under investigation. |
| 51 | `Penalty` | RaceControl | both | — | 2 | A sporting penalty applied to a driver. |
| 52 | `Result` | Result | kaggle | — | 3 | A driver's classified result in a race. |
| 53 | `Position` | Result | both | — | 1 | A position/order value. |
| 54 | `Points` | Result | kaggle | — | 1 | Championship points value. |
| 55 | `Championship` | Result | kaggle | — | 0 | A championship classification context. |
| 56 | `DriverStanding` | Result | kaggle | Championship | 3 | A driver's championship standing snapshot. |
| 57 | `ConstructorStanding` | Result | kaggle | Championship | 3 | A constructor's championship standing snapshot. |
| 58 | `Strategy` | Decision | derived | — | 2 | A recommended race-strategy plan (pit/tyre/fuel decisions). |
| 59 | `Prediction` | Decision | derived | — | 2 | A model prediction over race outcomes. |
| 60 | `Recommendation` | Decision | derived | — | 1 | An actionable recommendation surfaced to the race engineer. |
| 61 | `Optimization` | Decision | derived | — | 2 | An optimisation result over the strategy search space. |
