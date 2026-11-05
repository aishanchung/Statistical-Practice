# -*- coding: utf-8 -*-
"""
01_basic_eda.py
===============

V1 Core Master initial EDA (memory-safe; DuckDB)

Trip-level EDA:
- trips, VehicleType, TripLength

Time5-level EDA:
- M05A speed/volume and weather are first deduplicated to one row per Time5
  to avoid weighting busy intervals repeatedly just because more trip rows exist.

Input:
D:\\統計實務\\Data\\MASTER_V1_2024_2025\\*_MASTER_V1.parquet

Output:
D:\\統計實務\\Data\\EDA_V1_2024_2025
"""

from pathlib import Path
import warnings

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. Settings
# ============================================================

DATA_ROOT = Path(r"D:\統計實務\Data")
MASTER_ROOT = DATA_ROOT / "MASTER_V1_2024_2025"
OUT_ROOT = DATA_ROOT / "EDA_V1_2024_2025"
TABLE_DIR = OUT_ROOT / "tables"
FIG_DIR = OUT_ROOT / "figures"
TEMP_DIR = OUT_ROOT / "_duckdb_temp"

START_TS = "2024-01-01 00:00:00"
END_TS = "2026-01-01 00:00:00"

SCATTER_SAMPLE_N = 60000
TRIP_LENGTH_MAIN_MAX = 400
TRIP_LENGTH_BIN_WIDTH = 10
FIG_DPI = 180

DAYTYPE_ORDER = [
    "WEEKDAY",
    "WEEKEND",
    "PRE_LONG_HOLIDAY",
    "LONG_HOLIDAY",
    "STANDALONE_HOLIDAY",
]

VEHICLE_LABELS = {
    "31": "Passenger car (31)",
    "32": "Small truck (32)",
    "41": "Bus (41)",
    "42": "Truck (42)",
    "5": "Articulated (5)",
    "0": "Unknown (0)",
}


# ============================================================
# 2. Helpers
# ============================================================

def sql_path(path: Path) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


def route_key(file: Path) -> str:
    return file.stem.replace("_2024_2025_MASTER_V1", "")


def save_csv(df: pd.DataFrame, filename: str):
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLE_DIR / filename
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print("  saved table :", path)


def save_fig(fig, filename: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print("  saved figure:", path)


def make_connection():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(database=":memory:")
    con.execute(f"SET temp_directory='{sql_path(TEMP_DIR)}'")
    con.execute("SET preserve_insertion_order=false")
    return con


def set_plot_defaults():
    plt.rcParams["font.sans-serif"] = [
        "Microsoft JhengHei",
        "Microsoft YaHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 3. Trip-level summaries
# ============================================================

def query_file_summary(con, key: str) -> pd.DataFrame:
    return con.execute(f"""
        SELECT
            '{key}' AS RouteDirection,
            COUNT(*) AS Trips,
            COUNT(DISTINCT VehicleID) AS UniqueVehicleIDs,
            MIN(BottleneckTime) AS BottleneckTimeMin,
            MAX(BottleneckTime) AS BottleneckTimeMax,
            AVG(TripLength) AS MeanTripLength,
            approx_quantile(TripLength, 0.50) AS MedianTripLength,
            approx_quantile(TripLength, 0.90) AS P90TripLength,
            approx_quantile(TripLength, 0.99) AS P99TripLength,
            MAX(TripLength) AS MaxTripLength
        FROM trip
    """).fetchdf()


def query_daily_trips(con, key: str) -> pd.DataFrame:
    return con.execute(f"""
        SELECT
            '{key}' AS RouteDirection,
            CAST(BottleneckTime AS DATE) AS Date,
            ANY_VALUE(DayType) AS DayType,
            COUNT(*) AS Trips,
            COUNT(DISTINCT VehicleID) AS UniqueVehicleIDs
        FROM trip
        GROUP BY 1, 2
        ORDER BY Date
    """).fetchdf()


def query_hourly_trips(con, key: str) -> pd.DataFrame:
    return con.execute(f"""
        SELECT
            '{key}' AS RouteDirection,
            CAST(BottleneckTime AS DATE) AS Date,
            EXTRACT(hour FROM BottleneckTime)::INTEGER AS Hour,
            ANY_VALUE(DayType) AS DayType,
            COUNT(*) AS Trips
        FROM trip
        GROUP BY 1, 2, 3
        ORDER BY Date, Hour
    """).fetchdf()


def query_daytype_hour_profile(con, key: str) -> pd.DataFrame:
    return con.execute(f"""
        WITH hourly AS (
            SELECT
                CAST(BottleneckTime AS DATE) AS Date,
                EXTRACT(hour FROM BottleneckTime)::INTEGER AS Hour,
                ANY_VALUE(DayType) AS DayType,
                COUNT(*) AS Trips
            FROM trip
            GROUP BY 1, 2
        )
        SELECT
            '{key}' AS RouteDirection,
            DayType,
            Hour,
            COUNT(*) AS NDateHours,
            AVG(Trips) AS MeanTrips,
            MEDIAN(Trips) AS MedianTrips,
            approx_quantile(Trips, 0.25) AS Q25Trips,
            approx_quantile(Trips, 0.75) AS Q75Trips
        FROM hourly
        GROUP BY DayType, Hour
        ORDER BY DayType, Hour
    """).fetchdf()


def query_vehicle_composition(con, key: str) -> pd.DataFrame:
    return con.execute(f"""
        WITH x AS (
            SELECT
                DayType,
                CAST(VehicleType AS VARCHAR) AS VehicleType,
                COUNT(*) AS N
            FROM trip
            GROUP BY DayType, VehicleType
        )
        SELECT
            '{key}' AS RouteDirection,
            DayType,
            VehicleType,
            N,
            N * 1.0 / SUM(N) OVER (PARTITION BY DayType) AS Share
        FROM x
        ORDER BY DayType, VehicleType
    """).fetchdf()


def query_trip_length_summary(con, key: str) -> pd.DataFrame:
    return con.execute(f"""
        SELECT
            '{key}' AS RouteDirection,
            DayType,
            COUNT(*) AS N,
            AVG(TripLength) AS MeanTripLength,
            MEDIAN(TripLength) AS MedianTripLength,
            approx_quantile(TripLength, 0.75) AS P75TripLength,
            approx_quantile(TripLength, 0.90) AS P90TripLength,
            approx_quantile(TripLength, 0.95) AS P95TripLength,
            approx_quantile(TripLength, 0.99) AS P99TripLength,
            AVG(CASE WHEN TripLength > 500 THEN 1 ELSE 0 END)
                AS VeryLongTripShare
        FROM trip
        GROUP BY DayType
        ORDER BY DayType
    """).fetchdf()


def query_trip_length_hist(con, key: str) -> pd.DataFrame:
    max_main = TRIP_LENGTH_MAIN_MAX
    bw = TRIP_LENGTH_BIN_WIDTH

    return con.execute(f"""
        SELECT
            '{key}' AS RouteDirection,
            CASE
                WHEN TripLength >= {max_main} THEN '{max_main}+'
                ELSE CAST(FLOOR(TripLength / {bw}) * {bw} AS INTEGER)::VARCHAR
            END AS BinStartKM,
            COUNT(*) AS N
        FROM trip
        GROUP BY 2
        ORDER BY
            CASE
                WHEN BinStartKM = '{max_main}+' THEN 999999
                ELSE CAST(BinStartKM AS INTEGER)
            END
    """).fetchdf()


# ============================================================
# 4. Build one row per Time5
# ============================================================

def build_interval5(con):
    con.execute("DROP TABLE IF EXISTS interval5")

    con.execute(f"""
        CREATE TEMP TABLE interval5 AS
        SELECT
            Time5,
            CAST(Time5 AS DATE) AS Date,
            EXTRACT(hour FROM Time5)::INTEGER AS Hour,

            ANY_VALUE(DayType) AS DayType,
            ANY_VALUE(IsWeekend) AS IsWeekend,
            ANY_VALUE(IsLongHoliday) AS IsLongHoliday,
            ANY_VALUE(IsPreLongHoliday) AS IsPreLongHoliday,
            ANY_VALUE(HolidayPhase) AS HolidayPhase,
            ANY_VALUE(HolidayPhaseGroup) AS HolidayPhaseGroup,
            ANY_VALUE(IsSummer) AS IsSummer,

            ANY_VALUE(M05_Speed_Weighted) AS M05_Speed_Weighted,
            ANY_VALUE(M05_Volume_Total) AS M05_Volume_Total,

            ANY_VALUE(M05_Speed_31) AS M05_Speed_31,
            ANY_VALUE(M05_Volume_31) AS M05_Volume_31,
            ANY_VALUE(M05_Speed_32) AS M05_Speed_32,
            ANY_VALUE(M05_Volume_32) AS M05_Volume_32,
            ANY_VALUE(M05_Speed_41) AS M05_Speed_41,
            ANY_VALUE(M05_Volume_41) AS M05_Volume_41,
            ANY_VALUE(M05_Speed_42) AS M05_Speed_42,
            ANY_VALUE(M05_Volume_42) AS M05_Volume_42,
            ANY_VALUE(M05_Speed_5) AS M05_Speed_5,
            ANY_VALUE(M05_Volume_5) AS M05_Volume_5,

            ANY_VALUE(WeatherTime) AS WeatherTime,
            ANY_VALUE(Weather_Temperature_C) AS Temperature_C,
            ANY_VALUE(Weather_Dewpoint_C) AS Dewpoint_C,
            ANY_VALUE(Weather_Precipitation_mm) AS Precipitation_mm,
            ANY_VALUE(Weather_WindSpeed_mps) AS WindSpeed_mps,

            COUNT(*) AS M06BTripsInInterval

        FROM trip
        WHERE
            Time5 >= TIMESTAMP '{START_TS}'
            AND Time5 < TIMESTAMP '{END_TS}'
        GROUP BY Time5
        ORDER BY Time5
    """)


def query_interval_summary(con, key: str) -> pd.DataFrame:
    return con.execute(f"""
        SELECT
            '{key}' AS RouteDirection,
            COUNT(*) AS NTime5,
            COUNT(M05_Speed_Weighted) AS NValidWeightedSpeed,
            COUNT(*) FILTER (WHERE M05_Speed_Weighted IS NULL)
                AS NWeightedSpeedNA,
            AVG(M05_Speed_Weighted) AS MeanSpeed,
            MEDIAN(M05_Speed_Weighted) AS MedianSpeed,
            approx_quantile(M05_Speed_Weighted, 0.05) AS P05Speed,
            approx_quantile(M05_Speed_Weighted, 0.95) AS P95Speed,
            AVG(M05_Volume_Total) AS MeanVolume,
            MEDIAN(M05_Volume_Total) AS MedianVolume,
            approx_quantile(M05_Volume_Total, 0.95) AS P95Volume,
            MAX(M05_Volume_Total) AS MaxVolume
        FROM interval5
    """).fetchdf()


def query_hourly_speed(con, key: str) -> pd.DataFrame:
    return con.execute(f"""
        SELECT
            '{key}' AS RouteDirection,
            Date,
            Hour,
            AVG(M05_Speed_Weighted) AS MeanSpeed,
            MEDIAN(M05_Speed_Weighted) AS MedianSpeed,
            AVG(M05_Volume_Total) AS MeanVolume,
            SUM(M06BTripsInInterval) AS M06BTrips
        FROM interval5
        GROUP BY Date, Hour
        ORDER BY Date, Hour
    """).fetchdf()


def query_rain_speed_summary(con, key: str) -> pd.DataFrame:
    return con.execute(f"""
        WITH x AS (
            SELECT
                CASE
                    WHEN Precipitation_mm IS NULL THEN 'Missing'
                    WHEN Precipitation_mm = 0 THEN '0'
                    WHEN Precipitation_mm <= 1 THEN '(0,1]'
                    WHEN Precipitation_mm <= 5 THEN '(1,5]'
                    ELSE '>5'
                END AS RainBin,
                M05_Speed_Weighted AS Speed,
                M05_Volume_Total AS Volume
            FROM interval5
        )
        SELECT
            '{key}' AS RouteDirection,
            RainBin,
            COUNT(*) AS NTime5,
            COUNT(Speed) AS NValidSpeed,
            AVG(Speed) AS MeanSpeed,
            MEDIAN(Speed) AS MedianSpeed,
            approx_quantile(Speed, 0.25) AS Q25Speed,
            approx_quantile(Speed, 0.75) AS Q75Speed,
            AVG(Volume) AS MeanVolume
        FROM x
        GROUP BY RainBin
        ORDER BY
            CASE RainBin
                WHEN '0' THEN 1
                WHEN '(0,1]' THEN 2
                WHEN '(1,5]' THEN 3
                WHEN '>5' THEN 4
                ELSE 5
            END
    """).fetchdf()


def fetch_interval_for_plot(con) -> pd.DataFrame:
    return con.execute("""
        SELECT
            Time5,
            Date,
            Hour,
            DayType,
            M05_Speed_Weighted,
            M05_Volume_Total,
            Precipitation_mm,
            M06BTripsInInterval
        FROM interval5
    """).fetchdf()


# ============================================================
# 5. Plots
# ============================================================

def plot_daily_trips(df: pd.DataFrame, key: str):
    x = df.sort_values("Date")
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(pd.to_datetime(x["Date"]), x["Trips"], linewidth=0.8)
    ax.set_title(f"{key} - Daily trips")
    ax.set_xlabel("Date")
    ax.set_ylabel("Trips")
    ax.grid(alpha=0.2)
    save_fig(fig, f"{key}_01_daily_trips.png")


def plot_hourly_median(df: pd.DataFrame, key: str):
    profile = (
        df.groupby("Hour", as_index=False)["Trips"]
        .median()
        .rename(columns={"Trips": "MedianTrips"})
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        profile["Hour"],
        profile["MedianTrips"],
        marker="o",
        linewidth=1.8,
    )
    ax.set_title(f"{key} - Median hourly trip profile")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Median trips per hour")
    ax.set_xticks(range(24))
    ax.grid(alpha=0.2)
    save_fig(fig, f"{key}_02_hourly_profile.png")


def plot_daytype_hour_profile(df: pd.DataFrame, key: str):
    fig, ax = plt.subplots(figsize=(10, 6))

    for daytype in DAYTYPE_ORDER:
        g = df[df["DayType"] == daytype].sort_values("Hour")
        if g.empty:
            continue
        ax.plot(
            g["Hour"],
            g["MedianTrips"],
            marker="o",
            markersize=3,
            linewidth=1.6,
            label=daytype,
        )

    ax.set_title(f"{key} - Hourly profile by DayType")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Median trips per hour")
    ax.set_xticks(range(24))
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    save_fig(fig, f"{key}_03_daytype_hour_profile.png")


def plot_speed_heatmap(hourly_speed: pd.DataFrame, key: str):
    temp = hourly_speed.copy()
    temp["Date"] = pd.to_datetime(temp["Date"])

    pivot = temp.pivot(index="Hour", columns="Date", values="MedianSpeed")
    pivot = pivot.reindex(range(24))

    fig, ax = plt.subplots(figsize=(16, 6))
    img = ax.imshow(
        pivot.values,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
    )

    ax.set_title(f"{key} - Hourly median speed heatmap")
    ax.set_xlabel("Date")
    ax.set_ylabel("Hour")
    ax.set_yticks(range(24))

    n_dates = len(pivot.columns)
    if n_dates > 0:
        tick_idx = np.linspace(0, n_dates - 1, min(10, n_dates), dtype=int)
        ax.set_xticks(tick_idx)
        ax.set_xticklabels(
            [pd.Timestamp(pivot.columns[i]).strftime("%Y-%m-%d") for i in tick_idx],
            rotation=45,
            ha="right",
        )

    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label("Median speed (km/h)")
    save_fig(fig, f"{key}_04_speed_heatmap.png")


def plot_volume_speed(interval: pd.DataFrame, key: str):
    d = interval[
        interval["M05_Speed_Weighted"].notna()
        & interval["M05_Volume_Total"].notna()
        & (interval["M05_Volume_Total"] > 0)
    ].copy()

    if len(d) > SCATTER_SAMPLE_N:
        d = d.sample(SCATTER_SAMPLE_N, random_state=2026)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        d["M05_Volume_Total"],
        d["M05_Speed_Weighted"],
        s=5,
        alpha=0.15,
    )
    ax.set_title(f"{key} - Volume vs weighted speed")
    ax.set_xlabel("M05 total volume (veh / 5 min)")
    ax.set_ylabel("Weighted speed (km/h)")
    ax.grid(alpha=0.15)
    save_fig(fig, f"{key}_05_volume_speed.png")


def plot_vehicle_composition(df: pd.DataFrame, key: str):
    d = df.copy()
    d["VehicleType"] = d["VehicleType"].astype(str)
    d["VehicleLabel"] = d["VehicleType"].map(VEHICLE_LABELS).fillna(d["VehicleType"])

    pivot = (
        d.pivot_table(
            index="DayType",
            columns="VehicleLabel",
            values="Share",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(DAYTYPE_ORDER)
        .dropna(how="all")
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    bottom = np.zeros(len(pivot))

    for col in pivot.columns:
        vals = pivot[col].values * 100
        ax.bar(pivot.index, vals, bottom=bottom, label=col)
        bottom += vals

    ax.set_title(f"{key} - Vehicle composition by DayType")
    ax.set_xlabel("DayType")
    ax.set_ylabel("Share (%)")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(fontsize=8)
    save_fig(fig, f"{key}_06_vehicle_composition.png")


def plot_trip_length_hist(df: pd.DataFrame, key: str):
    d = df.copy()

    def bucket_order(x):
        x = str(x)
        if x.endswith("+"):
            return TRIP_LENGTH_MAIN_MAX
        return int(x)

    d["BucketOrder"] = d["BinStartKM"].map(bucket_order)
    d = d.sort_values("BucketOrder")

    labels = [
        (
            f"{int(x)}+"
            if int(x) >= TRIP_LENGTH_MAIN_MAX
            else f"{int(x)}-{int(x) + TRIP_LENGTH_BIN_WIDTH}"
        )
        for x in d["BucketOrder"]
    ]

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(np.arange(len(d)), d["N"].values)

    step = max(1, len(d) // 12)
    idx = np.arange(0, len(d), step)
    ax.set_xticks(idx)
    ax.set_xticklabels([labels[i] for i in idx], rotation=45, ha="right")
    ax.set_title(f"{key} - TripLength distribution")
    ax.set_xlabel("TripLength bin (km)")
    ax.set_ylabel("Trips")
    save_fig(fig, f"{key}_07_trip_length_hist.png")


def plot_rain_speed(summary: pd.DataFrame, key: str):
    order = ["0", "(0,1]", "(1,5]", ">5", "Missing"]
    d = summary.set_index("RainBin").reindex(order).dropna(how="all").reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(d["RainBin"], d["MedianSpeed"])
    ax.set_title(f"{key} - Median speed by hourly rainfall")
    ax.set_xlabel("Hourly precipitation (mm)")
    ax.set_ylabel("Median weighted speed (km/h)")
    save_fig(fig, f"{key}_08_rain_speed.png")


# ============================================================
# 6. Process one file
# ============================================================

def process_file(file: Path):
    key = route_key(file)

    print("\n" + "=" * 90)
    print(key)
    print("=" * 90)

    con = make_connection()
    con.execute(f"""
        CREATE VIEW trip AS
        SELECT *
        FROM read_parquet('{sql_path(file)}')
    """)

    print("  [1] Trip-level summaries")

    file_summary = query_file_summary(con, key)
    daily = query_daily_trips(con, key)
    hourly = query_hourly_trips(con, key)
    daytype_hour = query_daytype_hour_profile(con, key)
    vehicle = query_vehicle_composition(con, key)
    trip_length = query_trip_length_summary(con, key)
    trip_hist = query_trip_length_hist(con, key)

    save_csv(file_summary, f"{key}_01_file_summary.csv")
    save_csv(daily, f"{key}_02_daily_trips.csv")
    save_csv(hourly, f"{key}_03_hourly_trips.csv")
    save_csv(daytype_hour, f"{key}_04_daytype_hour_profile.csv")
    save_csv(vehicle, f"{key}_05_vehicle_composition.csv")
    save_csv(trip_length, f"{key}_06_trip_length_summary.csv")
    save_csv(trip_hist, f"{key}_07_trip_length_hist.csv")

    print("  [2] Build one-row-per-Time5 traffic state")
    build_interval5(con)

    interval_summary = query_interval_summary(con, key)
    hourly_speed = query_hourly_speed(con, key)
    rain_speed = query_rain_speed_summary(con, key)
    interval_plot = fetch_interval_for_plot(con)

    save_csv(interval_summary, f"{key}_08_interval_summary.csv")
    save_csv(hourly_speed, f"{key}_09_hourly_speed.csv")
    save_csv(rain_speed, f"{key}_10_rain_speed_summary.csv")

    print("  [3] Figures")
    plot_daily_trips(daily, key)
    plot_hourly_median(hourly, key)
    plot_daytype_hour_profile(daytype_hour, key)
    plot_speed_heatmap(hourly_speed, key)
    plot_volume_speed(interval_plot, key)
    plot_vehicle_composition(vehicle, key)
    plot_trip_length_hist(trip_hist, key)
    plot_rain_speed(rain_speed, key)

    con.close()

    return {
        "summary": file_summary,
        "interval_summary": interval_summary,
        "daytype_hour": daytype_hour,
        "vehicle": vehicle,
        "trip_length": trip_length,
        "rain_speed": rain_speed,
    }


# ============================================================
# 7. Combined tables
# ============================================================

def make_combined_tables(results):
    save_csv(
        pd.concat([x["summary"] for x in results], ignore_index=True),
        "ALL_01_file_summary.csv",
    )
    save_csv(
        pd.concat([x["interval_summary"] for x in results], ignore_index=True),
        "ALL_02_interval_summary.csv",
    )
    save_csv(
        pd.concat([x["daytype_hour"] for x in results], ignore_index=True),
        "ALL_03_daytype_hour_profile.csv",
    )
    save_csv(
        pd.concat([x["vehicle"] for x in results], ignore_index=True),
        "ALL_04_vehicle_composition.csv",
    )
    save_csv(
        pd.concat([x["trip_length"] for x in results], ignore_index=True),
        "ALL_05_trip_length_summary.csv",
    )
    save_csv(
        pd.concat([x["rain_speed"] for x in results], ignore_index=True),
        "ALL_06_rain_speed_summary.csv",
    )


# ============================================================
# 8. README
# ============================================================

def write_readme():
    text = """V1 Core Master - Basic EDA
==========================

Data levels
-----------
1. Trip-level
   Trips / VehicleType / TripLength / OD use original Master trip rows.

2. Time5-level
   Speed / Volume / Weather are deduplicated to one row per Time5 first.
   This avoids weighting busy intervals repeatedly because the Master contains
   one M05A state copied to many trip rows.

Important limitation
--------------------
The Master is trip-based. Therefore interval5 only contains Time5 values where
at least one M06B trip passed the research bottleneck.

For the final 5-minute Modeling Dataset, use a complete time grid / original
M05A source so that each day can contain the full 288 five-minute intervals.

Main figures
------------
01 Daily trips
02 Median hourly profile
03 Hourly profile by DayType
04 Speed heatmap
05 Volume-Speed
06 Vehicle composition
07 TripLength distribution
08 Rain-Speed

Next stage
----------
OD EDA:
- Top Origins
- Top Destinations
- Top OD pairs
- OD matrix heatmap
- Top1 / Top5 share
- HHI
- Entropy
- JS divergence
- Dynamic OD visualization
"""

    path = OUT_ROOT / "README_EDA.txt"
    path.write_text(text, encoding="utf-8")
    print("saved:", path)


# ============================================================
# 9. Main
# ============================================================

def main():
    warnings.filterwarnings("ignore")
    set_plot_defaults()

    files = sorted(MASTER_ROOT.glob("*_MASTER_V1.parquet"))

    if not files:
        raise FileNotFoundError(f"Cannot find Master parquet: {MASTER_ROOT}")

    print("=" * 90)
    print("V1 CORE MASTER - BASIC EDA")
    print("=" * 90)
    print("Input :", MASTER_ROOT)
    print("Output:", OUT_ROOT)
    print("Files :", len(files))

    results = []

    for i, file in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] {file.name}")
        results.append(process_file(file))

    print("\n" + "=" * 90)
    print("Combined tables")
    print("=" * 90)

    make_combined_tables(results)
    write_readme()

    print("\n" + "=" * 90)
    print("EDA COMPLETE")
    print("=" * 90)
    print("Tables :", TABLE_DIR)
    print("Figures:", FIG_DIR)
    print()
    print("Next: inspect figures, then send ALL_*.csv and any unusual figures back to me.")


if __name__ == "__main__":
    main()
