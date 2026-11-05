#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""V3_LITE (2024–2025 weekend/long-holiday) EDA before OD clustering.

Outputs reproducible aggregate CSVs, PNG charts, and summary.md. Heavy OD
aggregations run in DuckDB; only small result tables are loaded into pandas.

Example (PowerShell; one line per option with a PowerShell backtick):
    & ".\.venv\Scripts\python.exe" ".\10_OD analysis\v3_eda_full.py" `
      --od-input "D:\統計實務\Data\V3_LITE\V3_parts" `
      --y-input "D:\統計實務\Data\V3_LITE\V3_Y_5MIN.parquet" `
      --output ".\outputs\v3_eda_full" --threads 4 --memory-limit 4GB

Requires: duckdb, pandas, numpy, matplotlib (installed in the same .venv).
The original DayType is used; V3_DayType is intentionally ignored.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


FOCUS = ("WEEKEND", "LONGHOLIDAY")
SITES = [61.670, 63.990, 65.990, 69.510]


def options():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--od-input", required=True, help="V3_LITE.parquet or a directory of part_*.parquet files")
    p.add_argument("--y-input", required=True, help="V3_Y_5MIN.parquet from four-VD pipeline")
    p.add_argument("--vd-input", help="Optional V3_LITE_VD_4SITES.parquet")
    p.add_argument("--output", default="outputs/v3_eda_full")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--memory-limit", default="4GB")
    p.add_argument("--length-unit", choices=["auto", "km", "m"], default="auto")
    p.add_argument("--inspect", action="store_true", help="Print schemas and sample values without full aggregation")
    p.add_argument("--allow-incomplete-parts", action="store_true", help="Continue despite gaps in part_0000... numbering")
    return p.parse_args()


def files_for(raw: str, allow_incomplete: bool):
    path = Path(raw).expanduser()
    if path.is_file():
        if path.suffix.lower() != ".parquet":
            raise ValueError(f"Not a parquet file: {path}")
        return [str(path.resolve())], []
    if not path.is_dir():
        raise FileNotFoundError(path)
    paths = sorted(path.glob("part_*.parquet"))
    if not paths:
        paths = sorted(path.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No Parquet files in {path}")
    numbered = [int(m.group(1)) for f in paths if (m := re.fullmatch(r"part_(\d+)\.parquet", f.name))]
    absent = sorted(set(range(max(numbered) + 1)) - set(numbered)) if numbered else []
    if absent and not allow_incomplete:
        raise ValueError(f"Missing OD parts: {absent}. Rerun with --allow-incomplete-parts only for exploratory analysis.")
    return [str(f.resolve()) for f in paths], absent


def qident(value):
    return '"' + value.replace('"', '""') + '"'


def parquet_sql(paths):
    # DuckDB SQL string literals require single quotes, unlike JSON strings.
    paths_quoted = ["'" + p.replace("\\", "/").replace("'", "''") + "'" for p in paths]
    return "read_parquet([" + ", ".join(paths_quoted) + "], union_by_name=true)"


def save_query(con, sql, out, name, *, plot=None):
    df = con.execute(sql).df()
    df.to_csv(out / (name + ".csv"), index=False, encoding="utf-8-sig")
    if plot is not None and not df.empty:
        plot(df, out / (name + ".png"))
    print(f"  {name}: {len(df):,} rows", flush=True)
    return df


def write_csv(df, out, name, *, plot=None):
    df.to_csv(out / (name + ".csv"), index=False, encoding="utf-8-sig")
    if plot is not None and not df.empty:
        plot(df, out / (name + ".png"))
    print(f"  {name}: {len(df):,} rows", flush=True)
    return df


def line_chart(x, hue, value, xlabel, ylabel, title):
    def draw(df, path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        for key, part in df.groupby(hue, dropna=False):
            part = part.sort_values(x)
            ax.plot(part[x], part[value], marker="o", markersize=2, label=str(key))
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=.25)
        fig.tight_layout()
        fig.savefig(path, dpi=170)
        plt.close(fig)
    return draw


def holiday_phase_chart(df, path):
    """Same holiday-phase color across years; year encoded by line style.

    2024 = solid, 2025 = dashed. A consistent phase palette prevents
    misleading color comparisons across calendar years.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"first": "#3274a1", "middle": "#2b985e", "last": "#db7b27"}
    styles = {2024: "-", 2025: "--"}
    labels = {"first": "First day", "middle": "Middle days", "last": "Last day"}
    fig, ax = plt.subplots(figsize=(10, 5.4))
    for phase in ("first", "middle", "last"):
        for year in (2024, 2025):
            part = df[(df["phase"] == phase) & (df["Year"] == year)].sort_values("hour")
            if part.empty:
                continue
            ax.plot(part["hour"], part["avg_trips_per_day_hour"],
                    color=colors[phase], linestyle=styles[year], linewidth=2.4,
                    label=f"{labels[phase]} · {year}")
    ax.set(xlabel="Hour (0–23)", ylabel="Mean trips per day-hour",
           title="Holiday phases: 2024 (solid) vs. 2025 (dashed)")
    ax.set_xticks(range(0, 24, 2))
    ax.grid(alpha=.25)
    ax.legend(loc="upper left", ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def bar_chart(category, value, title, ylabel):
    def draw(df, path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(df[category].astype(str), df[value])
        ax.set(title=title, ylabel=ylabel)
        ax.tick_params(axis="x", labelrotation=30)
        ax.grid(axis="y", alpha=.25)
        fig.tight_layout()
        fig.savefig(path, dpi=170)
        plt.close(fig)
    return draw


def multi_bar_chart(group, series, value, title, ylabel):
    def draw(df, path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        part = df.pivot_table(index=group, columns=series, values=value, aggfunc="sum", fill_value=0)
        fig, ax = plt.subplots(figsize=(10, 5))
        part.plot(kind="bar", ax=ax, width=.8)
        ax.set(title=title, ylabel=ylabel, xlabel="")
        ax.tick_params(axis="x", labelrotation=25)
        ax.legend(fontsize=8, title="")
        ax.grid(axis="y", alpha=.25)
        fig.tight_layout()
        fig.savefig(path, dpi=170)
        plt.close(fig)
    return draw


def md_table(df):
    # Avoid pandas.to_markdown(), which requires the optional tabulate package.
    columns = list(df.columns)
    if not columns:
        return "(empty)"
    lines = ["| " + " | ".join(columns) + " |",
             "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in df.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def schema(con, path_sql):
    return con.execute(f"DESCRIBE SELECT * FROM {path_sql}").df().column_name.tolist()


def find_col(cols, *names):
    mapping = {c.lower(): c for c in cols}
    for name in names:
        if name.lower() in mapping:
            return mapping[name.lower()]
    return None


def main():
    args = options()
    try:
        import duckdb
        import pandas as pd
        import numpy as np
    except ImportError as exc:
        raise SystemExit("Please install dependencies: python -m pip install duckdb pandas numpy matplotlib") from exc

    od_files, absent = files_for(args.od_input, args.allow_incomplete_parts)
    y_files, _ = files_for(args.y_input, True)
    con = duckdb.connect()
    con.execute(f"SET threads={args.threads}")
    con.execute(f"SET memory_limit='{args.memory_limit}'")
    od_sql, y_sql = parquet_sql(od_files), parquet_sql(y_files)
    od_cols, y_cols = schema(con, od_sql), schema(con, y_sql)
    print(f"OD files: {len(od_files)}; missing numbered parts: {absent}")
    print("OD columns:", od_cols)
    print("Y columns:", y_cols)
    vd_cols = []
    if args.vd_input:
        vd_files, _ = files_for(args.vd_input, True)
        vd_sql = parquet_sql(vd_files)
        vd_cols = schema(con, vd_sql)
        print("VD columns:", vd_cols)

    time_col = find_col(od_cols, "BottleneckTime", "Time5", "TargetTime")
    day_col = find_col(od_cols, "DayType")
    origin_col = find_col(od_cols, "GantryO", "Origin", "O")
    dest_col = find_col(od_cols, "GantryD", "Destination", "D")
    vehicle_col = find_col(od_cols, "VehicleType")
    length_col = find_col(od_cols, "TripLength")
    if any(c is None for c in [time_col, origin_col, dest_col]):
        raise ValueError("OD requires BottleneckTime/Time5, GantryO and GantryD. Use --inspect to check schema.")
    need_y = {"Time5", "Date", "DayType", "Year", "Y", "n_valid_vd", "C_weighted"}
    if not need_y.issubset(set(y_cols)):
        raise ValueError(f"Y schema missing: {sorted(need_y - set(y_cols))}. Regenerate Y with the available-VD rule.")

    if args.inspect:
        print("OD sample:")
        print(con.execute(f"SELECT * FROM {od_sql} LIMIT 4").df().to_string(index=False))
        print("Y sample:")
        print(con.execute(f"SELECT * FROM {y_sql} LIMIT 4").df().to_string(index=False))
        if length_col:
            print("TripLength quantiles:")
            print(con.execute(f"SELECT approx_quantile(TRY_CAST({qident(length_col)} AS DOUBLE), [.05,.5,.95,.99]) FROM {od_sql}").fetchall())
        return

    out = Path(args.output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "input_files.json").write_text(json.dumps({"od_files": od_files, "y_files": y_files,
        "vd_input": args.vd_input, "missing_parts": absent, "od_columns": od_cols, "y_columns": y_cols},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print("[1/6] Loading V3 views and expected holiday calendar...", flush=True)
    con.execute(f"CREATE VIEW y0 AS SELECT * FROM {y_sql}")
    con.execute("""CREATE TEMP VIEW calendar AS
        SELECT CAST(Date AS DATE) AS Date, MAX(CAST(Year AS INTEGER)) AS Year,
               MAX(UPPER(TRIM(DayType))) AS DayType
          FROM y0 WHERE UPPER(TRIM(DayType)) IN ('WEEKEND','LONGHOLIDAY')
         GROUP BY 1""")

    # Use the authoritative date-level V3 calendar; do not trust record-level
    # DayType in a date with conflicting labels.
    t = qident(time_col)
    o, d = qident(origin_col), qident(dest_col)
    vehicle_expr = f"CAST(r.{qident(vehicle_col)} AS VARCHAR)" if vehicle_col else "'UNKNOWN'"
    length_expr = f"TRY_CAST(r.{qident(length_col)} AS DOUBLE)" if length_col else "NULL::DOUBLE"
    weather_col = find_col(od_cols, "RainFlag", "rain_flag", "IsRain", "Rain", "Precipitation")
    weather_expr = f"CAST(r.{qident(weather_col)} AS VARCHAR)" if weather_col else "NULL::VARCHAR"
    con.execute(f"""CREATE TEMP VIEW trips AS
        SELECT CAST(r.{t} AS TIMESTAMP) AS EventTime,
               time_bucket(INTERVAL '5 minutes', CAST(r.{t} AS TIMESTAMP)) AS Time5,
               CAST(r.{o} AS VARCHAR) AS GantryO, CAST(r.{d} AS VARCHAR) AS GantryD,
               {vehicle_expr} AS VehicleType, {length_expr} AS TripLength,
               {weather_expr} AS RainValue, c.Date, c.Year, c.DayType
          FROM {od_sql} r
          JOIN calendar c ON CAST(r.{t} AS DATE) = c.Date
         WHERE r.{t} IS NOT NULL AND r.{o} IS NOT NULL AND r.{d} IS NOT NULL""")

    y_check = save_query(con, """SELECT Year, DayType,
        COUNT(*) AS expected_bins, COUNT(Y) AS y_valid_bins,
        SUM(CASE WHEN n_valid_vd=0 THEN 1 ELSE 0 END) AS all_missing_bins,
        SUM(CASE WHEN n_valid_vd BETWEEN 1 AND 3 THEN 1 ELSE 0 END) AS partial_vd_bins,
        SUM(CASE WHEN (Y IS NULL) != (n_valid_vd=0) THEN 1 ELSE 0 END) AS y_rule_violations
        FROM y0 GROUP BY 1,2 ORDER BY 1,2""", out, "y_availability")
    if y_check.y_rule_violations.sum():
        raise ValueError("Y has missing values inconsistent with n_valid_vd. Please rebuild Y before analysis.")

    print("[2/6] OD demand and 24-hour temporal patterns...", flush=True)
    date_counts = save_query(con, """SELECT c.Date, c.Year, c.DayType,
        EXTRACT(ISODOW FROM c.Date) AS isodow, COUNT(t.Time5) AS trips
        FROM calendar c LEFT JOIN trips t ON c.Date=t.Date
        GROUP BY 1,2,3,4 ORDER BY 1""", out, "daily_demand")
    summary = save_query(con, """SELECT Year, DayType, COUNT(DISTINCT Date) AS days,
        COUNT(*) AS od_trips, COUNT(DISTINCT (GantryO, GantryD)) AS unique_od_pairs
        FROM trips GROUP BY 1,2 ORDER BY 1,2""", out, "data_summary")
    daily_group = date_counts.groupby(["Year", "DayType"], as_index=False).agg(
        mean_daily_trips=("trips", "mean"), median_daily_trips=("trips", "median"))
    write_csv(daily_group, out, "mean_daily_demand", plot=multi_bar_chart(
        "DayType", "Year", "mean_daily_trips", "Mean daily trips (not raw totals)", "Trips/day"))
    # Calendar-driven denominator includes all focus dates, even zero-OD days.
    hourly = save_query(con, """WITH counts AS (
        SELECT Year, DayType, EXTRACT(HOUR FROM EventTime) AS hour, COUNT(*) AS trips
        FROM trips GROUP BY 1,2,3
    ), days AS (
        SELECT Year, DayType, COUNT(*) AS days FROM calendar GROUP BY 1,2
    ) SELECT counts.Year, counts.DayType, counts.hour, counts.trips,
      counts.trips * 1.0 / days.days AS avg_trips_per_day_hour
      FROM counts JOIN days USING (Year,DayType)
      ORDER BY Year, DayType, hour""", out, "hourly_by_daytype_year")
    hourly["series"] = hourly["Year"].astype(str) + " " + hourly["DayType"]
    line_chart("hour", "series", "avg_trips_per_day_hour", "Hour (0–23)",
        "Mean OD trips per day-hour", "V3: 24-hour demand by year / day type")(hourly, out / "hourly_by_daytype_year.png")

    save_query(con, """SELECT Year, DayType, EXTRACT(MONTH FROM Date) AS month,
      COUNT(*) AS days, AVG(trips) AS avg_daily_trips, SUM(trips) AS total_trips
      FROM (SELECT c.Date,c.Year,c.DayType,COUNT(t.Time5) AS trips
            FROM calendar c LEFT JOIN trips t ON c.Date=t.Date
            GROUP BY 1,2,3) GROUP BY 1,2,3 ORDER BY 1,2,3""", out, "monthly_demand")
    weekend = save_query(con, """WITH h AS (
        SELECT Year, EXTRACT(ISODOW FROM EventTime) AS isodow,
          EXTRACT(HOUR FROM EventTime) AS hour, COUNT(*) AS trips
        FROM trips WHERE DayType='WEEKEND' GROUP BY 1,2,3
    ), days AS (SELECT Year, EXTRACT(ISODOW FROM Date) AS isodow, COUNT(*) AS days
        FROM calendar WHERE DayType='WEEKEND' GROUP BY 1,2)
    SELECT h.Year, h.isodow, h.hour, h.trips * 1.0 / days.days AS avg_trips_per_day_hour
    FROM h JOIN days USING (Year,isodow) ORDER BY 1,2,3""", out, "weekend_saturday_sunday")
    weekend["series"] = weekend.Year.astype(str) + " " + weekend.isodow.map({6: "Sat", 7: "Sun"}).fillna("other")
    line_chart("hour", "series", "avg_trips_per_day_hour", "Hour", "Mean trips/day-hour",
               "V3: Saturdays vs Sundays")(weekend, out / "weekend_saturday_sunday.png")

    # Holiday phases depend on consecutive LONGHOLIDAY dates, not simply the
    # first/last row of all dates; date gaps split separate holiday episodes.
    cal = con.execute("SELECT Date, Year, DayType FROM calendar ORDER BY Date").df()
    cal.Date = pd.to_datetime(cal.Date)
    lh = cal[cal.DayType == "LONGHOLIDAY"].copy().sort_values("Date")
    lh["block"] = (lh.Date.diff().dt.days.ne(1) | lh.Year.ne(lh.Year.shift())).cumsum()
    lh["position"] = lh.groupby("block").cumcount()
    lh["total"] = lh.groupby("block").Date.transform("size")
    lh["phase"] = np.select([lh.total.eq(1), lh.position.eq(0), lh.position.eq(lh.total - 1)],
                             ["single", "first", "last"], default="middle")
    # 2024-01-01 is the LAST day of the official 2023-12-30 to 2024-01-01
    # New Year's holiday. Those first two days lie outside the V3 date range;
    # without this boundary override, the observed lone day is mislabeled
    # "single". Retain Jan 1 in the 2024 sample and label it correctly.
    boundary_last = lh.Date.eq(pd.Timestamp("2024-01-01")) & lh.phase.eq("single")
    lh["phase_source"] = "within_V3_calendar"
    lh.loc[boundary_last, "phase"] = "last"
    lh.loc[boundary_last, "total"] = 3
    lh.loc[boundary_last, "phase_source"] = "official_cross_year_holiday"
    if lh.phase.eq("single").any():
        print("WARNING: Some LONGHOLIDAY dates are isolated in the V3 calendar; verify the official holiday span.")
    write_csv(lh[["Date", "Year", "block", "phase", "total", "phase_source"]],
              out, "holiday_phase_dates")
    if not lh.empty:
        con.register("holiday_phases", lh[["Date", "Year", "phase"]])
        phase = save_query(con, """WITH h AS (
            SELECT p.Year, p.phase, EXTRACT(HOUR FROM t.EventTime) AS hour, COUNT(*) AS trips
            FROM trips t JOIN holiday_phases p USING (Date,Year) GROUP BY 1,2,3
        ), nd AS (SELECT Year,phase,COUNT(*) AS days FROM holiday_phases GROUP BY 1,2)
        SELECT h.Year,h.phase,h.hour,h.trips*1.0/nd.days AS avg_trips_per_day_hour
        FROM h JOIN nd USING(Year,phase) ORDER BY 1,2,3""", out, "holiday_phase_hourly")
        holiday_phase_chart(phase, out / "holiday_phase_hourly.png")

    print("[3/6] Vehicle composition, trip distance, OD long tail...", flush=True)
    vehicles = save_query(con, """WITH counts AS (
        SELECT Year, DayType, VehicleType, COUNT(*) AS trips FROM trips GROUP BY 1,2,3)
        SELECT *, ROUND(100*trips*1.0/SUM(trips) OVER(PARTITION BY Year,DayType),2) AS pct
        FROM counts ORDER BY Year,DayType,trips DESC""", out, "vehicle_mix")
    vehicles["series"] = vehicles.Year.astype(str) + " " + vehicles.DayType
    multi_bar_chart("VehicleType", "series", "pct", "Vehicle mix by date type", "Share (%)")(
        vehicles, out / "vehicle_mix.png")
    save_query(con, """SELECT Year,DayType,VehicleType,
        EXTRACT(HOUR FROM EventTime) AS hour,COUNT(*) AS trips
        FROM trips GROUP BY 1,2,3,4 ORDER BY 1,2,3,4""", out, "vehicle_hourly")

    unit_note = "TripLength column missing"
    if length_col:
        quants = con.execute("""SELECT approx_quantile(TripLength,[.01,.50,.95,.99])
             FROM trips WHERE TripLength>0 AND isfinite(TripLength)""").fetchone()[0]
        if args.length_unit == "auto":
            # Sanity-check inferred units; user should confirm metadata if unsure.
            factor = 1000.0 if quants and quants[1] > 2000 else 1.0
            unit_note = f"AUTO guessed {'meters' if factor==1000 else 'kilometers'}; raw quantiles={quants}. VERIFY!"
        else:
            factor = 1000.0 if args.length_unit == "m" else 1.0
            unit_note = f"Explicit --length-unit={args.length_unit}; raw quantiles={quants}"
        print("TripLength:", unit_note)
        distances = save_query(con, f"""WITH lengths AS (
          SELECT Year, DayType, TripLength/{factor} AS km FROM trips
          WHERE TripLength > 0 AND isfinite(TripLength)
        ) SELECT Year,DayType,
          CASE WHEN km<50 THEN '0–50' WHEN km<100 THEN '50–100'
               WHEN km<200 THEN '100–200' WHEN km<500 THEN '200–500'
               ELSE '500+' END AS distance_km, COUNT(*) AS trips
          FROM lengths GROUP BY 1,2,3 ORDER BY 1,2,3""", out, "distance_distribution")
        distances["series"] = distances.Year.astype(str) + " " + distances.DayType
        distances["pct"] = 100 * distances.trips / distances.groupby("series").trips.transform("sum")
        multi_bar_chart("distance_km", "series", "pct", "Trip length distributions", "Share (%)")(
            distances, out / "distance_distribution.png")
    else:
        print("  Skipping distance: no TripLength column")

    pairs = save_query(con, """SELECT GantryO,GantryD,
        SUM(CASE WHEN Year=2024 THEN 1 ELSE 0 END) AS trips_2024,
        SUM(CASE WHEN Year=2025 THEN 1 ELSE 0 END) AS trips_2025,
        COUNT(*) AS trips_total
        FROM trips GROUP BY 1,2 ORDER BY trips_total DESC""", out, "od_pair_counts")
    if not pairs.empty:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.loglog(np.arange(1, len(pairs) + 1), pairs.trips_total.to_numpy(), linewidth=1.5)
        ax.set(title="V3: OD long-tail distribution", xlabel="OD pair rank (log)",
               ylabel="Trip count (log)")
        ax.grid(alpha=.25)
        fig.tight_layout()
        fig.savefig(out / "od_rank_long_tail.png", dpi=170)
        plt.close(fig)
        top20 = pairs.head(20)[["GantryO", "GantryD"]]
        con.register("top20_od_pairs", top20)
        od_top = save_query(con, """SELECT t.GantryO,t.GantryD,
              EXTRACT(HOUR FROM EventTime) AS hour,COUNT(*) AS trips
              FROM trips t INNER JOIN top20_od_pairs p USING(GantryO,GantryD)
              GROUP BY 1,2,3 ORDER BY 1,2,3""", out, "top20_od_hourly")
        top5_keys = set(zip(pairs.head(5).GantryO, pairs.head(5).GantryD))
        top5 = od_top[[ (a,b) in top5_keys for a,b in zip(od_top.GantryO,od_top.GantryD) ]].copy()
        top5["pair"] = top5.GantryO.astype(str) + " → " + top5.GantryD.astype(str)
        line_chart("hour", "pair", "trips", "Hour", "Trips across all included dates",
                   "Top five OD pairs: time-of-day shape")(top5, out / "top5_od_hourly.png")
    # Concentration and low-volume diagnostics are small pandas calculations.
    total = int(pairs.trips_total.sum())
    concentration = []
    for top in [1, 5, 10, 20, 50, 100, 500]:
        concentration.append({"top_pairs": top, "trip_share_pct":
            100 * pairs.trips_total.head(top).sum() / total if total else np.nan})
    write_csv(pd.DataFrame(concentration), out, "od_concentration", plot=bar_chart(
        "top_pairs", "trip_share_pct", "OD pair concentration", "Share of all trips (%)"))
    coverage = []
    for n in [10, 30, 100, 300, 500, 1000]:
        selected = pairs[pairs.trips_2024 >= n]
        coverage.append({"min_2024_trips": n, "selected_pairs": len(selected),
                         "2024_trip_share_pct": 100*selected.trips_2024.sum()/max(1,pairs.trips_2024.sum()),
                         "2025_trip_share_pct": 100*selected.trips_2025.sum()/max(1,pairs.trips_2025.sum())})
    write_csv(pd.DataFrame(coverage), out, "od_train_threshold_coverage")

    if weather_col:
        print("[4/6] Available weather field:", weather_col, flush=True)
        rain = save_query(con, """SELECT Year,DayType,RainValue,COUNT(*) AS trips
            FROM trips GROUP BY 1,2,3 ORDER BY 1,2,3""", out, "weather_demand_raw")
        rain["series"] = rain.Year.astype(str) + " " + rain.DayType
        rain["pct"] = 100 * rain.trips / rain.groupby("series").trips.transform("sum")
        multi_bar_chart("RainValue", "series", "pct", "Observed rain labels (record-level)", "Share (%)")(
            rain, out / "weather_demand_raw.png")
    else:
        print("[4/6] No recognized rain field in OD; weather EDA skipped. Do not invent RainFlag.")
    weather_candidates = {
        "temperature": ["Weather_Temperature_C", "Temperature", "AirTemperature", "Temp", "Temperature_C"],
        "dew_point": ["Weather_Dewpoint_C", "DewPoint", "Dewpoint", "DewPointTemperature"],
        "wind_speed": ["Weather_WindSpeed_mps", "WindSpeed", "Wind_Speed"],
        "rain_mm": ["Weather_Precipitation_mm", "Rainfall", "PrecipitationMM", "RainfallMM", "Precipitation"],
    }
    for label, names in weather_candidates.items():
        weather_numeric_col = find_col(od_cols, *names)
        if weather_numeric_col is None:
            continue
        # Average across trips is travel-weighted and is not a meteorological
        # station-hour climatology. Export raw descriptive summaries only.
        save_query(con, f"""SELECT c.Year,c.DayType,EXTRACT(HOUR FROM CAST(r.{t} AS TIMESTAMP)) AS hour,
             COUNT(TRY_CAST(r.{qident(weather_numeric_col)} AS DOUBLE)) AS n,
             AVG(TRY_CAST(r.{qident(weather_numeric_col)} AS DOUBLE)) AS trip_weighted_mean,
             approx_quantile(TRY_CAST(r.{qident(weather_numeric_col)} AS DOUBLE),.5) AS median
             FROM {od_sql} r JOIN calendar c ON CAST(r.{t} AS DATE)=c.Date
             GROUP BY 1,2,3 ORDER BY 1,2,3""", out, "weather_" + label + "_hourly")

    print("[5/6] Congestion Y, missingness and OD/Y associations...", flush=True)
    y_dist = save_query(con, """SELECT Year,DayType,Y,COUNT(*) AS bins,
        ROUND(100*COUNT(*)*1.0/SUM(COUNT(*)) OVER(PARTITION BY Year,DayType),3) AS pct_all_bins
        FROM y0 GROUP BY 1,2,3 ORDER BY 1,2,3""", out, "y_distribution")
    valid_y = y_dist[y_dist.Y.notna()].copy()
    valid_y["series"] = valid_y.Year.astype(str) + " " + valid_y.DayType
    valid_y["pct_valid"] = 100*valid_y.bins / valid_y.groupby("series").bins.transform("sum")
    multi_bar_chart("Y", "series", "pct_valid", "Y classes (valid time bins only)", "Share (%)")(
        valid_y, out / "y_distribution.png")
    yh = save_query(con, """SELECT Year,DayType,EXTRACT(HOUR FROM Time5) AS hour,
       COUNT(Y) AS valid_bins,COUNT(*) AS expected_bins,
       SUM(CASE WHEN Y=2 THEN 1 ELSE 0 END) AS broad_bins,
       SUM(CASE WHEN Y=1 THEN 1 ELSE 0 END) AS local_bins,
       100.0*SUM(CASE WHEN Y=2 THEN 1 ELSE 0 END)/NULLIF(COUNT(Y),0) AS broad_pct_of_valid
       FROM y0 GROUP BY 1,2,3 ORDER BY 1,2,3""", out, "y_hourly")
    yh["series"] = yh.Year.astype(str) + " " + yh.DayType
    line_chart("hour", "series", "broad_pct_of_valid", "Hour", "Y=2 among valid bins (%)",
               "V3: Broad congestion by hour")(yh, out / "y_hourly.png")
    save_query(con, """SELECT Year,DayType,EXTRACT(MONTH FROM Time5) AS month,
        COUNT(*) AS bins, COUNT(Y) AS valid_bins,
        SUM(CASE WHEN Y=2 THEN 1 ELSE 0 END) AS broad_bins
        FROM y0 GROUP BY 1,2,3 ORDER BY 1,2,3""", out, "y_monthly")
    save_query(con, """SELECT Year,DayType,n_valid_vd, COUNT(*) AS bins
        FROM y0 GROUP BY 1,2,3 ORDER BY 1,2,3""", out, "vd_valid_site_counts")
    # Congestion events: adjacency is five minutes, and date discontinuities
    # split events. Missing bins interrupt events; cannot claim continuity.
    events = save_query(con, """WITH active AS (
        SELECT Time5,Year,DayType,Y,
            LAG(Time5) OVER (ORDER BY Time5) AS prev_time,
            LAG(Y) OVER (ORDER BY Time5) AS prev_y
        FROM y0 WHERE Y IS NOT NULL
    ), runs AS (
        SELECT *, SUM(CASE WHEN Y!=prev_y OR prev_y IS NULL
                       OR prev_time IS NULL OR Time5-prev_time!=INTERVAL '5 minutes'
                       THEN 1 ELSE 0 END) OVER (ORDER BY Time5) AS run_id FROM active
    ) SELECT MIN(Time5) AS start_time, MAX(Time5) AS last_bin,
          MIN(Year) AS Year, MIN(DayType) AS DayType, MIN(Y) AS Y,
          COUNT(*) AS n_bins, COUNT(*)*5 AS duration_min
       FROM runs GROUP BY run_id HAVING MIN(Y) IN (1,2)
       ORDER BY start_time""", out, "y_congestion_events")
    five = save_query(con, """SELECT y.Time5,y.Year,y.DayType,y.Y,y.n_valid_vd,
       y.C_weighted, COUNT(t.Time5) AS od_trips_5min
       FROM y0 y LEFT JOIN trips t ON t.Time5=y.Time5
       GROUP BY 1,2,3,4,5,6 ORDER BY 1""", out, "od_5min_join_y")
    association = five[five.Y.notna()].groupby(["Year", "DayType", "Y"], as_index=False).agg(
        bins=("Time5", "size"), mean_od_5min=("od_trips_5min", "mean"),
        median_od_5min=("od_trips_5min", "median"))
    write_csv(association, out, "od_volume_by_y")
    five["hour"] = pd.to_datetime(five.Time5).dt.hour
    five["week"] = pd.to_datetime(five.Time5).dt.isocalendar().week.astype(int)
    # 5-minute series are allowed as retrospective descriptions only. Lagging
    # predictors and availability constraints must be handled in forecasting.
    write_csv(five.groupby(["Year", "DayType", "hour", "Y"], dropna=False, as_index=False)
                  .agg(bins=("Time5", "size"), mean_od_5min=("od_trips_5min", "mean")),
              out, "od_volume_by_y_hour")

    if args.vd_input:
        if {"Time5", "SiteKM", "Speed", "MissingStatus"}.issubset(set(vd_cols)):
            save_query(con, f"""SELECT Year,DayType,SiteKM,MissingStatus,COUNT(*) AS bins,
               SUM(CASE WHEN TRY_CAST(Speed AS DOUBLE)<40 AND MissingStatus='VALID'
                        THEN 1 ELSE 0 END) AS congested_bins
               FROM {vd_sql} GROUP BY 1,2,3,4 ORDER BY 1,2,3,4""", out, "vd_station_missing_speed")
        else:
            print("  Skipped VD station-level breakdown: unexpected VD schema")

    print("[6/6] Data consistency checks and report...", flush=True)
    # Compare whole-day trip totals to the Y build's own record counts to flag
    # incomplete OD part uploads, not automatically to invalidate Y.
    joined = con.execute("""SELECT SUM(CAST(od_trips_5min AS BIGINT)) FROM y0""").fetchone()[0] if "od_trips_5min" in y_cols else None
    observed = total
    mismatch = (joined is not None and int(joined) != observed)
    if mismatch:
        print(f"WARNING OD count mismatch: current={observed:,} vs original Y={int(joined):,}. Check missing parts or duplicate files.")
    ndays = len(date_counts)
    no_od_dates = int((date_counts.trips == 0).sum())
    long_events = events[events.duration_min >= 15]
    summaries = [
        "# V3_LITE EDA — pre-clustering", "",
        "## Scope", "- 2024–2025; WEEKEND and LONGHOLIDAY; 24 h/day.",
        "- OD times use BottleneckTime (or the available fallback). DayType comes from the Y-date calendar.",
        "- Four-VD Y: available VD weights renormalized; Y missing only when all four are invalid.",
        "- Y=2 threshold is the value used when V3_Y_5MIN.parquet was built (check its run configuration).", "",
        "## Data checks", f"- OD files read: {len(od_files)}; absent part numbers: {absent}",
        f"- Calendar dates: {ndays}; zero-OD dates: {no_od_dates}; total selected OD trips: {observed:,}.",
        f"- Original Y build total OD trips: {joined if joined is not None else 'not available'}.",
        f"- OD/Y count mismatch: **{mismatch}**.",
        f"- TripLength: {unit_note}", "",
        "## Group counts", md_table(summary), "",
        "## Y missingness", md_table(y_check), "",
        f"## Congestion events", f"- All Y=1/2 runs: {len(events):,}.",
        f"- Runs >=15 min: {len(long_events):,}. Missing Y breaks runs.", "",
        "## OD clustering implications", "- Compare OD group shares and volumes by hour / holiday phase.",
        "- Select feature sets and K using 2024 training / 2025 held-out predictive performance, not only silhouette.",
        "- The 2024/2025 DayType composition differs; daily-normalized patterns matter.",
        "- OD flow at target five-minute interval is **not** a valid advance predictor unless it is actually available at prediction time.",
        "- TripLength auto-detection, holiday calendar classification, and count mismatches require human verification.",
    ]
    (out / "summary.md").write_text("\n".join(summaries) + "\n", encoding="utf-8")
    print(f"Done. Output: {out}")


if __name__ == "__main__":
    main()
