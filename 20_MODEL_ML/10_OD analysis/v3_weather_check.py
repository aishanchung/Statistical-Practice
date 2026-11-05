#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test whether 2024/2025 holiday weather patterns differ in V3.

This is descriptive, not a causal weather-effect estimate. Aggregate weather
to five-minute bins before comparing with VD-based congestion Y so that a
busy traffic bin is not counted thousands of times as a weather observation.

Requires duckdb, pandas, numpy, matplotlib; import helpers from v3_eda_full.py
in the same directory.
"""

import argparse
from pathlib import Path

from v3_eda_full import files_for, parquet_sql, qident, find_col


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--od-input", required=True)
    p.add_argument("--y-input", required=True)
    p.add_argument("--output", default="outputs/v3_weather_check")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--memory-limit", default="4GB")
    args = p.parse_args()

    import duckdb
    import pandas as pd
    out = Path(args.output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET threads={args.threads}")
    con.execute(f"SET memory_limit='{args.memory_limit}'")
    od_files, absent = files_for(args.od_input, False)
    y_files, _ = files_for(args.y_input, True)
    od_src = parquet_sql(od_files)
    y_src = parquet_sql(y_files)
    names = con.execute(f"DESCRIBE SELECT * FROM {od_src}").df().column_name.tolist()
    time_col = find_col(names, "BottleneckTime", "Time5", "TargetTime")
    rain_col = find_col(names, "Weather_Precipitation_mm", "Rainfall", "RainfallMM", "Precipitation")
    temp_col = find_col(names, "Weather_Temperature_C", "Temperature")
    wind_col = find_col(names, "Weather_WindSpeed_mps", "WindSpeed")
    if not time_col or not rain_col:
        raise ValueError(f"Need time and weather rain columns; found: {names}")
    metric = lambda name: ("AVG(TRY_CAST(r." + qident(name) + " AS DOUBLE))") if name else "NULL::DOUBLE"

    con.execute(f"CREATE VIEW Y0 AS SELECT * FROM {y_src}")
    # Group weather by observation time, never by the number of vehicle trips.
    # If multiple weather readings fall in one bin we take their mean; review
    # the original precipitation observation interval before interpreting mm.
    print("Aggregating V3 weather to five-minute bins...", flush=True)
    con.execute(f"""CREATE TEMP TABLE W AS
        SELECT time_bucket(INTERVAL '5 minutes', CAST(r.{qident(time_col)} AS TIMESTAMP)) AS Time5,
               {metric(rain_col)} AS rain_mm,
               {metric(temp_col)} AS temperature_c,
               {metric(wind_col)} AS wind_mps,
               COUNT(*) AS od_records
          FROM {od_src} r
          JOIN (SELECT DISTINCT CAST(Date AS DATE) Date FROM Y0) c
            ON CAST(r.{qident(time_col)} AS DATE)=c.Date
         GROUP BY 1""")
    w = con.execute("""SELECT y.Time5, y.Year, y.DayType, y.Y,
           y.n_valid_vd, y.od_trips_5min, w.rain_mm, w.temperature_c, w.wind_mps,
           CASE WHEN w.rain_mm IS NULL THEN 'unknown'
                WHEN w.rain_mm>0 THEN 'rain' ELSE 'no_rain' END AS rain_class
          FROM Y0 y LEFT JOIN W w USING (Time5) ORDER BY 1""").df()
    w["Date"] = pd.to_datetime(w.Time5).dt.normalize()
    w["hour"] = pd.to_datetime(w.Time5).dt.hour
    w.to_csv(out / "weather_5min_and_y.csv", index=False, encoding="utf-8-sig")

    by_year = w.groupby(["Year", "DayType", "rain_class"], as_index=False).agg(
        five_min_bins=("Time5", "count"), avg_od_trips=("od_trips_5min", "mean"))
    by_year["pct_bins"] = by_year.five_min_bins / by_year.groupby(["Year", "DayType"]).five_min_bins.transform("sum") * 100
    by_year.to_csv(out / "weather_by_year_daytype.csv", index=False, encoding="utf-8-sig")

    by_y = w[w.Y.notna()].groupby(["Year", "DayType", "rain_class", "Y"], as_index=False).agg(
        valid_y_bins=("Time5", "count"))
    by_y["pct_within_rain_class"] = by_y.valid_y_bins / by_y.groupby(
        ["Year", "DayType", "rain_class"]).valid_y_bins.transform("sum") * 100
    by_y.to_csv(out / "rain_vs_y.csv", index=False, encoding="utf-8-sig")

    daily = w.groupby(["Year", "DayType", "Date"], as_index=False).agg(
        weather_coverage_bins=("rain_mm", "count"),
        rainy_5min_bins=("rain_mm", lambda x: int((x > 0).sum())),
        average_temp_c=("temperature_c", "mean"),
        average_wind_mps=("wind_mps", "mean"),
        total_traffic=("od_trips_5min", "sum"))
    daily["rainy_pct_of_observed"] = 100 * daily.rainy_5min_bins / daily.weather_coverage_bins.replace(0, float("nan"))
    daily.to_csv(out / "daily_weather_and_traffic.csv", index=False, encoding="utf-8-sig")

    # Independent 2024 and 2025 lines; styles/palette follow holiday chart.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {2024: "#3274a1", 2025: "#db7b27"}
    fig, ax = plt.subplots(figsize=(9, 5))
    for year in [2024, 2025]:
        for typ, style in [("WEEKEND", ":"), ("LONGHOLIDAY", "-")]:
            part = w[(w.Year==year)&(w.DayType==typ)&(w.rain_class!='unknown')]
            pct = part.groupby("hour").rain_class.apply(lambda x: 100*(x=='rain').mean())
            ax.plot(pct.index, pct.values, label=f"{year} {typ}", color=colors[year], linestyle=style)
    ax.set(title="V3: rainy-bin share by time of day", xlabel="Hour", ylabel="Rainy bins (%)")
    ax.legend()
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(out / "rain_by_hour_year.png", dpi=180)
    plt.close(fig)
    print("Saved weather_by_year_daytype.csv, rain_vs_y.csv, daily_weather_and_traffic.csv and plot")
    print("IMPORTANT: weather association does not prove weather caused the year difference.")


if __name__ == "__main__":
    main()
