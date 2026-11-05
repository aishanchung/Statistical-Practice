# V3_LITE EDA — pre-clustering

## Scope
- 2024–2025; WEEKEND and LONGHOLIDAY; 24 h/day.
- OD times use BottleneckTime (or the available fallback). DayType comes from the Y-date calendar.
- Four-VD Y: available VD weights renormalized; Y missing only when all four are invalid.
- Y=2 threshold is the value used when V3_Y_5MIN.parquet was built (check its run configuration).

## Data checks
- OD files read: 1; absent part numbers: []
- Calendar dates: 230; zero-OD dates: 0; total selected OD trips: 16,068,709.
- Original Y build total OD trips: 16068709.
- OD/Y count mismatch: **False**.
- TripLength: AUTO guessed kilometers; raw quantiles=[5.585211359936108, 72.78111131174035, 268.8836639615826, 357.78295438451926]. VERIFY!

## Group counts
| Year | DayType | days | od_trips | unique_od_pairs |
| --- | --- | --- | --- | --- |
| 2024 | LONGHOLIDAY | 15 | 1171680 | 5879 |
| 2024 | WEEKEND | 98 | 6787300 | 8775 |
| 2025 | LONGHOLIDAY | 31 | 2233064 | 6939 |
| 2025 | WEEKEND | 86 | 5876665 | 9007 |

## Y missingness
| Year | DayType | expected_bins | y_valid_bins | all_missing_bins | partial_vd_bins | y_rule_violations |
| --- | --- | --- | --- | --- | --- | --- |
| 2024 | LONGHOLIDAY | 4320 | 4304 | 16.0 | 34.0 | 0.0 |
| 2024 | WEEKEND | 28224 | 26720 | 1504.0 | 823.0 | 0.0 |
| 2025 | LONGHOLIDAY | 8928 | 8857 | 71.0 | 61.0 | 0.0 |
| 2025 | WEEKEND | 24768 | 24566 | 202.0 | 835.0 | 0.0 |

## Congestion events
- All Y=1/2 runs: 2,349.
- Runs >=15 min: 678. Missing Y breaks runs.

## OD clustering implications
- Compare OD group shares and volumes by hour / holiday phase.
- Select feature sets and K using 2024 training / 2025 held-out predictive performance, not only silhouette.
- The 2024/2025 DayType composition differs; daily-normalized patterns matter.
- OD flow at target five-minute interval is **not** a valid advance predictor unless it is actually available at prediction time.
- TripLength auto-detection, holiday calendar classification, and count mismatches require human verification.

## Holiday-phase correction (2026-11-05)
- 2024-01-01 is the last day of the 2023-12-30–2024-01-01 three-day holiday, not a standalone holiday.
- 2024 uses solid lines; 2025 dashed lines; first/middle/last use matched colors across years.
