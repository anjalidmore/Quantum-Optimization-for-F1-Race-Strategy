# Attribute Table

_Generated 2026-08-03 04:22 UTC — 90 attributes across 61 entities._

| Entity | Attribute | Type | Unit | Description |
|--------|-----------|------|------|-------------|
| `Season` | `year` | integer | — | Calendar year of the championship. |
| `Season` | `url` | string | — | Reference URL for the season. |
| `Race` | `round` | integer | — | Round number within the season. |
| `Race` | `name` | string | — | Official race name. |
| `Race` | `date` | datetime | — | Race date. |
| `GrandPrix` | `name` | string | — | Grand Prix name. |
| `Circuit` | `name` | string | — | Circuit name. |
| `Circuit` | `location` | string | — | City / locality. |
| `Circuit` | `country` | string | — | Country. |
| `Circuit` | `length_km` | float | km | Track length. |
| `Circuit` | `lat` | float | deg | Latitude. |
| `Circuit` | `lng` | float | deg | Longitude. |
| `Driver` | `driver_ref` | string | — | Stable identifier / reference. |
| `Driver` | `code` | string | — | Three-letter driver code (e.g. VER). |
| `Driver` | `number` | integer | — | Permanent car number. |
| `Driver` | `forename` | string | — | First name. |
| `Driver` | `surname` | string | — | Last name. |
| `Driver` | `nationality` | string | — | Nationality. |
| `Driver` | `dob` | datetime | — | Date of birth. |
| `Constructor` | `constructor_ref` | string | — | Stable identifier. |
| `Constructor` | `name` | string | — | Constructor name. |
| `Constructor` | `nationality` | string | — | Nationality. |
| `Team` | `name` | string | — | Team name as reported by timing. |
| `Car` | `chassis` | string | — | Chassis designation. |
| `Car` | `power_unit` | string | — | Power unit manufacturer. |
| `Session` | `session_type` | string | — | FP/Q/S/R type code. |
| `Session` | `start_time` | datetime | — | Session start. |
| `Qualifying` | `q1` | float | s | Q1 time. |
| `Qualifying` | `q2` | float | s | Q2 time. |
| `Qualifying` | `q3` | float | s | Q3 time. |
| `Lap` | `lap_number` | integer | — | Lap index within the session. |
| `Lap` | `lap_time` | float | s | Total lap time. |
| `Lap` | `is_personal_best` | boolean | — | Whether this is the driver PB. |
| `Sector` | `sector_number` | integer | — | Sector index 1-3. |
| `SectorTime` | `time` | float | s | Sector time. |
| `LapTime` | `milliseconds` | integer | ms | Lap time. |
| `FastestLap` | `rank` | integer | — | Rank of the fastest lap. |
| `FastestLap` | `time` | float | s | Fastest lap time. |
| `Gap` | `seconds` | float | s | Gap. |
| `Interval` | `seconds` | float | s | Interval. |
| `Overtake` | `lap` | integer | — | Lap on which it occurred. |
| `PitStop` | `lap` | integer | — | Lap on which the stop occurred. |
| `PitStop` | `duration` | float | s | Stationary + pit-lane time. |
| `PitStop` | `stop_number` | integer | — | Ordinal stop for the driver. |
| `PitWindow` | `open_lap` | integer | — | First viable lap. |
| `PitWindow` | `close_lap` | integer | — | Last viable lap. |
| `Stint` | `stint_number` | integer | — | Ordinal stint. |
| `Stint` | `start_lap` | integer | — | First lap of the stint. |
| `Stint` | `end_lap` | integer | — | Last lap of the stint. |
| `TyreCompound` | `compound` | string | — | Compound name. |
| `TyreCompound` | `colour` | string | — | Marking colour. |
| `TyreLife` | `laps` | integer | — | Laps completed on the set. |
| `TyreLife` | `wear_pct` | float | % | Estimated wear. |
| `TrackTemperature` | `value` | float | C | Temp. |
| `AirTemperature` | `value` | float | C | Temp. |
| `Humidity` | `value` | float | % | Humidity. |
| `WindSpeed` | `value` | float | m/s | Speed. |
| `RainProbability` | `value` | float | % | Probability. |
| `FuelLoad` | `kg` | float | kg | Fuel mass. |
| `EngineMode` | `mode` | string | — | Mode label. |
| `ERS` | `charge_pct` | float | % | State of charge. |
| `DRS` | `active` | boolean | — | Whether DRS is open. |
| `Speed` | `value` | float | km/h | Speed. |
| `Throttle` | `value` | float | % | Throttle. |
| `Brake` | `value` | boolean | — | Brake on/off. |
| `RPM` | `value` | integer | — | RPM. |
| `Gear` | `value` | integer | — | Gear index. |
| `RaceControlMessage` | `message` | string | — | Message text. |
| `RaceControlMessage` | `lap` | integer | — | Lap of issue. |
| `Incident` | `description` | string | — | Incident detail. |
| `Penalty` | `seconds` | float | s | Time penalty. |
| `Penalty` | `reason` | string | — | Reason. |
| `Result` | `position` | integer | — | Finishing position. |
| `Result` | `points` | float | — | Points scored. |
| `Result` | `status` | string | — | Finish status. |
| `Position` | `value` | integer | — | Position number. |
| `Points` | `value` | float | — | Points. |
| `DriverStanding` | `points` | float | — | Cumulative points. |
| `DriverStanding` | `position` | integer | — | Standing position. |
| `DriverStanding` | `wins` | integer | — | Wins to date. |
| `ConstructorStanding` | `points` | float | — | Cumulative points. |
| `ConstructorStanding` | `position` | integer | — | Standing position. |
| `ConstructorStanding` | `wins` | integer | — | Wins to date. |
| `Strategy` | `name` | string | — | Strategy label. |
| `Strategy` | `n_stops` | integer | — | Planned pit stops. |
| `Prediction` | `target` | string | — | Predicted quantity. |
| `Prediction` | `confidence` | float | % | Confidence. |
| `Recommendation` | `text` | string | — | Recommendation text. |
| `Optimization` | `objective` | string | — | Objective optimised. |
| `Optimization` | `value` | float | — | Objective value. |
