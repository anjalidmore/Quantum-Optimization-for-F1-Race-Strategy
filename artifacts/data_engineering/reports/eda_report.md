# Exploratory Data Analysis Report

_Generated 2026-08-30 12:34 UTC._

This report summarises the domain analyses over the cleaned historical and FastF1 data. Figures are in `outputs/figures/`.

## Driver Analysis

|   driverId | surname    | code   |   races |   total_points |   avg_points |   wins |   podiums |   avg_finish |
|-----------:|:-----------|:-------|--------:|---------------:|-------------:|-------:|----------:|-------------:|
|         10 | Gasly      | GAS    |      24 |          281   |       11.708 |      6 |         9 |        4.792 |
|          1 | Hamilton   | HAM    |      24 |          256   |       10.667 |      2 |        12 |        4.583 |
|          3 | Leclerc    | LEC    |      24 |          232   |        9.667 |      2 |         6 |        5.417 |
|          9 | Piastri    | PIA    |      24 |          232   |        9.667 |      2 |         8 |        5.333 |
|          2 | Verstappen | VER    |      24 |          225   |        9.375 |      2 |         7 |        5.875 |
|          5 | Alonso     | ALO    |      24 |          223.5 |        9.312 |      1 |         7 |        4.875 |
|          8 | Perez      | PER    |      24 |          220   |        9.167 |      2 |         5 |        5.667 |
|          7 | Sainz      | SAI    |      24 |          202   |        8.417 |      2 |         5 |        5.708 |
|          6 | Russell    | RUS    |      24 |          196   |        8.167 |      2 |         4 |        5.875 |
|          4 | Norris     | NOR    |      24 |          170   |        7.083 |      1 |         4 |        6.292 |

## Constructor Analysis

|   constructorId | name         |   entries |   total_points |   avg_points |   wins |
|----------------:|:-------------|----------:|---------------:|-------------:|-------:|
|               3 | FERRARI      |        48 |          464   |        9.667 |      4 |
|               1 | MERCEDES     |        48 |          458   |        9.542 |      4 |
|               4 | MCLAREN      |        48 |          451   |        9.396 |      7 |
|               2 | RED BULL     |        48 |          445   |        9.271 |      4 |
|               5 | ASTON MARTIN |        24 |          223.5 |        9.312 |      1 |
|               6 | ALPINE       |        24 |          196   |        8.167 |      2 |

## Circuit Analysis

|   circuitId | name                         | country   |   entries |   finishers |   avg_points |   dnf_rate |
|------------:|:-----------------------------|:----------|----------:|------------:|-------------:|-----------:|
|           1 | CIRCUIT DE MONACO            | MONACO    |        40 |          40 |        9.938 |          0 |
|           2 | SILVERSTONE CIRCUIT          | UK        |        40 |          40 |        9.575 |          0 |
|           3 | AUTODROMO NAZIONALE MONZA    | ITALY     |        40 |          40 |        9.55  |          0 |
|           4 | CIRCUIT DE SPA-FRANCORCHAMPS | BELGIUM   |        40 |          40 |        9.2   |          0 |
|           5 | SUZUKA CIRCUIT               | JAPAN     |        40 |          40 |        8.875 |          0 |
|           6 | AUTÓDROMO JOSÉ CARLOS PACE   | BRAZIL    |        40 |          40 |        8.8   |          0 |

## Pit-Stop Analysis

| metric                    |   value |
|:--------------------------|--------:|
| count                     | 475     |
| mean_s                    |  23.037 |
| median_s                  |  23.047 |
| std_s                     |   2.44  |
| min_s                     |  16.456 |
| max_s                     |  28.733 |
| avg_stops_per_driver_race |   1.979 |

## Tyre Analysis

| compound   |   n_laps |   mean_laptime_s |   best_laptime_s |   deg_s_per_lap |
|:-----------|---------:|-----------------:|-----------------:|----------------:|
| MEDIUM     |       10 |           98.819 |           97.26  |         -0.0868 |
| HARD       |      588 |           98.911 |           96.156 |         -0.0574 |
| SOFT       |      457 |           99.594 |           94.003 |         -0.0611 |

## Lap-Time Analysis

| metric   |     value |
|:---------|----------:|
| count    | 13363     |
| mean_s   |    87.329 |
| median_s |    88.663 |
| std_s    |     4.984 |
| p05_s    |    79.092 |
| p95_s    |    93.114 |

## Weather Analysis

| channel   |   mean |   min |   max |   std |
|:----------|-------:|------:|------:|------:|
| AirTemp   |  26.75 |  26.2 |  27.3 |  0.33 |
| TrackTemp |  29.91 |  28.7 |  31.2 |  0.73 |
| Humidity  |  21.1  |  20   |  22   |  0.37 |
| WindSpeed |   0.51 |   0.1 |   0.9 |  0.23 |

## Season Analysis

|   year |   races |   total_points |   avg_points_per_entry |
|-------:|--------:|---------------:|-----------------------:|
|   2022 |      12 |         1087   |                  9.058 |
|   2023 |      12 |         1150.5 |                  9.588 |

## Safety-Car / Track-Status Analysis

|   status |   laps |
|---------:|-------:|
|        1 |    979 |
|       12 |     26 |
|       21 |     19 |
|      126 |     13 |
|      671 |     13 |
|     2671 |      5 |

