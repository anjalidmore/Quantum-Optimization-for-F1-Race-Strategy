# Data Cleaning Audit

_Generated 2026-08-03 04:42 UTC._

## Table: `races`

| Step | Detail | Rows before | Rows after | Extra |
|------|--------|-------------|------------|-------|
| `deduplicate` | Removed exact duplicate rows. | 24 | 24 | removed=0 |
| `coerce_types` | Coerced numeric/time/date columns; parsed lap times to seconds. | 24 | 24 |  |
| `normalise_categoricals` | Trimmed/upper-cased 1 categorical column(s). | 24 | 24 | columns=['name'] |
| `impute_missing` | Imputed numeric/time by median, categorical by mode. | 24 | 24 | na_before=0, na_after=0 |
| `detect_outliers` | Capped IQR outliers. | 24 | 24 | outliers={'year': 0, 'round': 0} |

## Table: `drivers`

| Step | Detail | Rows before | Rows after | Extra |
|------|--------|-------------|------------|-------|
| `deduplicate` | Removed exact duplicate rows. | 10 | 10 | removed=0 |
| `coerce_types` | Coerced numeric/time/date columns; parsed lap times to seconds. | 10 | 10 |  |
| `normalise_categoricals` | Trimmed/upper-cased 3 categorical column(s). | 10 | 10 | columns=['driverRef', 'code', 'nationality'] |
| `impute_missing` | Imputed numeric/time by median, categorical by mode. | 10 | 10 | na_before=0, na_after=0 |
| `detect_outliers` | Capped IQR outliers. | 10 | 10 | outliers={} |

## Table: `constructors`

| Step | Detail | Rows before | Rows after | Extra |
|------|--------|-------------|------------|-------|
| `deduplicate` | Removed exact duplicate rows. | 6 | 6 | removed=0 |
| `coerce_types` | Coerced numeric/time/date columns; parsed lap times to seconds. | 6 | 6 |  |
| `normalise_categoricals` | Trimmed/upper-cased 3 categorical column(s). | 6 | 6 | columns=['constructorRef', 'name', 'nationality'] |
| `impute_missing` | Imputed numeric/time by median, categorical by mode. | 6 | 6 | na_before=0, na_after=0 |
| `detect_outliers` | Capped IQR outliers. | 6 | 6 | outliers={} |

## Table: `circuits`

| Step | Detail | Rows before | Rows after | Extra |
|------|--------|-------------|------------|-------|
| `deduplicate` | Removed exact duplicate rows. | 6 | 6 | removed=0 |
| `coerce_types` | Coerced numeric/time/date columns; parsed lap times to seconds. | 6 | 6 |  |
| `normalise_categoricals` | Trimmed/upper-cased 4 categorical column(s). | 6 | 6 | columns=['circuitRef', 'name', 'location', 'country'] |
| `impute_missing` | Imputed numeric/time by median, categorical by mode. | 6 | 6 | na_before=0, na_after=0 |
| `detect_outliers` | Capped IQR outliers. | 6 | 6 | outliers={'lat': 1, 'lng': 2, 'alt': 1} |

## Table: `results`

| Step | Detail | Rows before | Rows after | Extra |
|------|--------|-------------|------------|-------|
| `deduplicate` | Removed exact duplicate rows. | 242 | 240 | removed=2 |
| `coerce_types` | Coerced numeric/time/date columns; parsed lap times to seconds. | 240 | 240 |  |
| `normalise_categoricals` | Trimmed/upper-cased 0 categorical column(s). | 240 | 240 | columns=[] |
| `impute_missing` | Imputed numeric/time by median, categorical by mode. | 240 | 240 | na_before=31, na_after=0 |
| `detect_outliers` | Capped IQR outliers. | 240 | 240 | outliers={'grid': 0, 'position': 0, 'points': 1, 'laps': 9, 'fastestLapTime': 0, 'rank': 0} |

## Table: `pit_stops`

| Step | Detail | Rows before | Rows after | Extra |
|------|--------|-------------|------------|-------|
| `deduplicate` | Removed exact duplicate rows. | 479 | 475 | removed=4 |
| `coerce_types` | Coerced numeric/time/date columns; parsed lap times to seconds. | 475 | 475 |  |
| `normalise_categoricals` | Trimmed/upper-cased 0 categorical column(s). | 475 | 475 | columns=[] |
| `impute_missing` | Imputed numeric/time by median, categorical by mode. | 475 | 475 | na_before=484, na_after=475 |
| `detect_outliers` | Capped IQR outliers. | 475 | 475 | outliers={'stop': 0, 'lap': 0, 'time': 0, 'duration': 2, 'milliseconds': 4} |

## Table: `lap_times`

| Step | Detail | Rows before | Rows after | Extra |
|------|--------|-------------|------------|-------|
| `deduplicate` | Removed exact duplicate rows. | 13495 | 13363 | removed=132 |
| `coerce_types` | Coerced numeric/time/date columns; parsed lap times to seconds. | 13363 | 13363 |  |
| `normalise_categoricals` | Trimmed/upper-cased 0 categorical column(s). | 13363 | 13363 | columns=[] |
| `impute_missing` | Imputed numeric/time by median, categorical by mode. | 13363 | 13363 | na_before=267, na_after=0 |
| `detect_outliers` | Capped IQR outliers. | 13363 | 13363 | outliers={'lap': 0, 'position': 0, 'time': 0, 'milliseconds': 67} |

## Table: `fastf1_laps`

| Step | Detail | Rows before | Rows after | Extra |
|------|--------|-------------|------------|-------|
| `deduplicate` | Removed exact duplicate rows. | 555 | 550 | removed=5 |
| `coerce_types` | Coerced numeric/time/date columns; parsed lap times to seconds. | 550 | 550 |  |
| `normalise_categoricals` | Trimmed/upper-cased 7 categorical column(s). | 550 | 550 | columns=['Driver', 'Team', 'Compound', 'FreshTyre', 'TrackStatus', 'Rainfall', 'IsPersonalBest'] |
| `impute_missing` | Imputed numeric/time by median, categorical by mode. | 550 | 550 | na_before=22, na_after=0 |
| `detect_outliers` | Capped IQR outliers. | 550 | 550 | outliers={'LapNumber': 0, 'LapTime': 3, 'Stint': 0, 'TyreLife': 0, 'Sector1Time': 0, 'Sector2Time': 1, 'Sector3Time': 0, 'SpeedFL': 4, 'SpeedST': 4, 'AirTemp': 2, 'TrackTemp': 6, 'Humidity': 0, 'WindSpeed': 0} |

