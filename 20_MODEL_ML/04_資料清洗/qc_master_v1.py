# -*- coding: utf-8 -*-
"""
QC for V1 Core Master parquet files
-----------------------------------
Input:
  D:\\統計實務\\Data\\MASTER_V1_2024_2025\\*_MASTER_V1.parquet
Output:
  D:\\統計實務\\Data\\QC_V1_2024_2025\\

Principle:
- First-pass QC only flags suspicious records; it does NOT delete or modify the Master data.
- Uses DuckDB to scan Parquet directly, so the large files do not need to be fully loaded into RAM.
"""

from pathlib import Path
import re
import sys
import duckdb
import pandas as pd

# ============================================================
# 1. Paths / settings
# ============================================================
DATA_ROOT = Path(r"D:\統計實務\Data")
MASTER_ROOT = DATA_ROOT / "MASTER_V1_2024_2025"
OUT_ROOT = DATA_ROOT / "QC_V1_2024_2025"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

FILE_PATTERN = "*_MASTER_V1.parquet"
START_TS = "2024-01-01 00:00:00"
END_TS = "2026-01-01 00:00:00"
SAMPLE_N = 20

# Expensive exact-duplicate check. Turn on after the fast QC has passed.
RUN_EXPENSIVE_DUPLICATE_CHECK = False

EXPECTED_VEHICLE_TYPES = {"31", "32", "41", "42", "5"}
EXPECTED_DAY_TYPES = {
    "WEEKDAY",
    "WEEKEND",
    "PRE_LONG_HOLIDAY",
    "LONG_HOLIDAY",
    "STANDALONE_HOLIDAY",
}

# Expected study segment pairs, based on the four route-direction files.
EXPECTED_SEGMENTS = {
    "N3_Daxi_Longtan_N": {
        "bottleneck_gantry": "03F0648N",
        "m05_from": "03F0648N",
        "m05_to": "03F0559N",
    },
    "N3_Daxi_Longtan_S": {
        "bottleneck_gantry": "03F0648S",
        "m05_from": "03F0559S",
        "m05_to": "03F0648S",
    },
    "N5_Pinglin_Toucheng_N": {
        "bottleneck_gantry": "05F0287N",
        "m05_from": "05F0287N",
        "m05_to": "05F0055N",
    },
    "N5_Pinglin_Toucheng_S": {
        "bottleneck_gantry": "05F0287S",
        "m05_from": "05F0055S",
        "m05_to": "05F0287S",
    },
}

SPEED_COLS = [
    "M05_Speed_31", "M05_Speed_32", "M05_Speed_41",
    "M05_Speed_42", "M05_Speed_5"
]
VOLUME_COLS = [
    "M05_Volume_31", "M05_Volume_32", "M05_Volume_41",
    "M05_Volume_42", "M05_Volume_5"
]

KEY_NUMERIC_COLS = [
    "TripLength",
    *SPEED_COLS,
    *VOLUME_COLS,
    "M05_Volume_Total",
    "M05_Speed_Weighted",
    "M05_Speed_OwnType",
    "M05_Volume_OwnType",
    "Weather_Temperature_C",
    "Weather_Dewpoint_C",
    "Weather_Precipitation_mm",
    "Weather_WindSpeed_mps",
]

CATEGORY_COLS = [
    "VehicleType", "DayType", "HolidayPhase", "HolidayPhaseGroup",
    "IsWeekend", "IsLongHoliday", "IsPreLongHoliday",
    "IsStandaloneHoliday", "IsHolidayWindow", "IsSummer",
    "M05_Matched", "Weather_Matched", "Calendar_Matched",
    "MasterRoute", "MasterDirection", "Bottleneck", "Direction",
    "BottleneckGantry", "M05_GantryFrom", "M05_GantryTo",
    "GantryFrom", "GantryTo",
]

EXPECTED_CORE_COLUMNS = [
    "VehicleID", "VehicleType", "DetectionTimeO", "GantryO",
    "DetectionTimeD", "GantryD", "TripLength", "TripEnd",
    "TripInformation", "BottleneckTime", "Time5", "WeatherTime",
    "CalendarDate", "M05_Speed_Weighted", "M05_Volume_Total",
    "Weather_Temperature_C", "Weather_Dewpoint_C",
    "Weather_Precipitation_mm", "Weather_WindSpeed_mps",
    "WeekdayNum", "IsWeekend", "DayType", "HolidayPhase",
    "IsSummer", "Calendar_Matched",
]

# ============================================================
# 2. Helpers
# ============================================================

def qident(name: str) -> str:
    """DuckDB-safe quoted identifier."""
    return '"' + name.replace('"', '""') + '"'


def sql_string(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def path_sql(p: Path) -> str:
    return str(p).replace("\\", "/").replace("'", "''")


def file_key(file: Path) -> str:
    stem = file.stem
    for k in EXPECTED_SEGMENTS:
        if stem.startswith(k):
            return k
    return stem.replace("_2024_2025_MASTER_V1", "")


def first_existing(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None


def add_rule(rules, file_name, rule_id, severity, description, count, total):
    count = int(count or 0)
    rate = (count / total) if total else None
    rules.append({
        "file": file_name,
        "rule_id": rule_id,
        "severity": severity,
        "description": description,
        "n_flagged": count,
        "total_rows": int(total),
        "flag_rate": rate,
    })


def run_count(con, where_sql: str) -> int:
    return con.execute(f"SELECT COUNT(*) FROM t WHERE {where_sql}").fetchone()[0]


def add_samples(con, sample_rows, file_name, rule_id, severity, where_sql, cols):
    existing = [c for c in cols if c in get_columns(con)]
    if not existing:
        return
    select_sql = ", ".join(qident(c) for c in existing)
    try:
        df = con.execute(
            f"SELECT {select_sql} FROM t WHERE {where_sql} LIMIT {SAMPLE_N}"
        ).df()
    except Exception:
        return
    if df.empty:
        return
    df.insert(0, "severity", severity)
    df.insert(0, "rule_id", rule_id)
    df.insert(0, "file", file_name)
    sample_rows.append(df)


def get_columns(con):
    return [r[0] for r in con.execute("DESCRIBE SELECT * FROM t").fetchall()]


def get_types(con):
    return {r[0]: r[1] for r in con.execute("DESCRIBE SELECT * FROM t").fetchall()}

# ============================================================
# 3. QC per file
# ============================================================

def qc_one_file(file: Path):
    con = duckdb.connect(database=":memory:")
    con.execute(f"CREATE VIEW t AS SELECT * FROM read_parquet('{path_sql(file)}')")

    cols = get_columns(con)
    types = get_types(con)
    total = con.execute("SELECT COUNT(*) FROM t").fetchone()[0]
    fkey = file_key(file)

    schema_rows = [
        {
            "file": file.name,
            "column": c,
            "duckdb_type": types[c],
            "is_expected_core": c in EXPECTED_CORE_COLUMNS,
        }
        for c in cols
    ]

    missing_expected = [c for c in EXPECTED_CORE_COLUMNS if c not in cols]

    # --------------------------------------------------------
    # File summary
    # --------------------------------------------------------
    summary = {
        "file": file.name,
        "file_key": fkey,
        "rows": int(total),
        "columns": len(cols),
        "missing_expected_columns": ";".join(missing_expected),
    }

    if "BottleneckTime" in cols:
        mn, mx = con.execute(
            "SELECT MIN(BottleneckTime), MAX(BottleneckTime) FROM t"
        ).fetchone()
        summary["BottleneckTime_min"] = mn
        summary["BottleneckTime_max"] = mx

    # --------------------------------------------------------
    # Missingness
    # --------------------------------------------------------
    missing_rows = []
    for c in cols:
        nmiss = con.execute(
            f"SELECT COUNT(*) FROM t WHERE {qident(c)} IS NULL"
        ).fetchone()[0]
        missing_rows.append({
            "file": file.name,
            "column": c,
            "n_missing": int(nmiss),
            "total_rows": int(total),
            "missing_rate": (nmiss / total) if total else None,
        })

    # --------------------------------------------------------
    # Numeric summaries (selected variables)
    # --------------------------------------------------------
    numeric_rows = []
    for c in KEY_NUMERIC_COLS:
        if c not in cols:
            continue
        # TRY_CAST makes this robust to numeric-like strings.
        qc = qident(c)
        sql = f"""
        WITH x AS (
            SELECT TRY_CAST({qc} AS DOUBLE) AS v
            FROM t
            WHERE {qc} IS NOT NULL
        )
        SELECT
            COUNT(v) AS n,
            MIN(v) AS min,
            approx_quantile(v, 0.001) AS p001,
            approx_quantile(v, 0.01) AS p01,
            approx_quantile(v, 0.05) AS p05,
            approx_quantile(v, 0.50) AS p50,
            approx_quantile(v, 0.95) AS p95,
            approx_quantile(v, 0.99) AS p99,
            approx_quantile(v, 0.999) AS p999,
            MAX(v) AS max,
            AVG(v) AS mean,
            STDDEV_SAMP(v) AS sd
        FROM x
        """
        row = con.execute(sql).fetchone()
        names = [d[0] for d in con.description]
        rec = dict(zip(names, row))
        rec.update({"file": file.name, "column": c})
        numeric_rows.append(rec)

    # --------------------------------------------------------
    # Category counts
    # --------------------------------------------------------
    category_rows = []
    for c in CATEGORY_COLS:
        if c not in cols:
            continue
        dfc = con.execute(
            f"""
            SELECT CAST({qident(c)} AS VARCHAR) AS value, COUNT(*) AS n
            FROM t
            GROUP BY 1
            ORDER BY n DESC
            """
        ).df()
        if not dfc.empty:
            dfc.insert(0, "column", c)
            dfc.insert(0, "file", file.name)
            category_rows.append(dfc)

    # --------------------------------------------------------
    # Rules
    # --------------------------------------------------------
    rules = []
    sample_rows = []

    # Schema completeness
    add_rule(
        rules, file.name, "SCHEMA_MISSING_EXPECTED", "HARD",
        "V1 expected core columns missing from this parquet",
        len(missing_expected), max(len(EXPECTED_CORE_COLUMNS), 1)
    )

    # Time range
    if "BottleneckTime" in cols:
        w = (
            f"BottleneckTime < TIMESTAMP {sql_string(START_TS)} "
            f"OR BottleneckTime >= TIMESTAMP {sql_string(END_TS)}"
        )
        n = run_count(con, w)
        add_rule(rules, file.name, "TIME_OUTSIDE_STUDY_PERIOD", "HARD",
                 "BottleneckTime outside 2024-01-01 to 2025-12-31", n, total)
        add_samples(con, sample_rows, file.name, "TIME_OUTSIDE_STUDY_PERIOD", "HARD", w,
                    ["VehicleID", "BottleneckTime", "SourceDate", "SourceFile"])

    # O / D logic
    if {"DetectionTimeO", "DetectionTimeD"}.issubset(cols):
        w = "DetectionTimeO > DetectionTimeD"
        n = run_count(con, w)
        add_rule(rules, file.name, "O_AFTER_D", "HARD",
                 "DetectionTimeO later than DetectionTimeD", n, total)
        add_samples(con, sample_rows, file.name, "O_AFTER_D", "HARD", w,
                    ["VehicleID", "DetectionTimeO", "DetectionTimeD", "GantryO", "GantryD", "TripLength"])

    if {"DetectionTimeO", "DetectionTimeD", "BottleneckTime"}.issubset(cols):
        w = "BottleneckTime < DetectionTimeO OR BottleneckTime > DetectionTimeD"
        n = run_count(con, w)
        add_rule(rules, file.name, "BOTTLENECK_OUTSIDE_TRIP_TIME", "HARD",
                 "BottleneckTime falls outside the trip O-D time interval", n, total)
        add_samples(con, sample_rows, file.name, "BOTTLENECK_OUTSIDE_TRIP_TIME", "HARD", w,
                    ["VehicleID", "DetectionTimeO", "BottleneckTime", "DetectionTimeD", "GantryO", "GantryD"])

    # Time5 alignment: expected ceil-to-5-min label
    if {"BottleneckTime", "Time5"}.issubset(cols):
        w = "Time5 < BottleneckTime OR Time5 > BottleneckTime + INTERVAL '5 minutes'"
        n = run_count(con, w)
        add_rule(rules, file.name, "TIME5_ALIGNMENT", "HARD",
                 "Time5 not within [BottleneckTime, BottleneckTime+5 min]", n, total)
        add_samples(con, sample_rows, file.name, "TIME5_ALIGNMENT", "HARD", w,
                    ["VehicleID", "BottleneckTime", "Time5"])

        w2 = "minute(Time5) % 5 <> 0 OR second(Time5) <> 0"
        n2 = run_count(con, w2)
        add_rule(rules, file.name, "TIME5_NOT_ON_GRID", "HARD",
                 "Time5 is not exactly on a 5-minute boundary", n2, total)

    # Weather alignment: expected floor-to-hour
    if {"BottleneckTime", "WeatherTime"}.issubset(cols):
        w = (
            "WeatherTime > BottleneckTime OR "
            "BottleneckTime >= WeatherTime + INTERVAL '1 hour'"
        )
        n = run_count(con, w)
        add_rule(rules, file.name, "WEATHER_TIME_ALIGNMENT", "HARD",
                 "WeatherTime is not the containing hour for BottleneckTime", n, total)
        add_samples(con, sample_rows, file.name, "WEATHER_TIME_ALIGNMENT", "HARD", w,
                    ["VehicleID", "BottleneckTime", "WeatherTime"])

    # TripLength
    if "TripLength" in cols:
        n = run_count(con, "TRY_CAST(TripLength AS DOUBLE) < 0")
        add_rule(rules, file.name, "TRIP_LENGTH_NEGATIVE", "HARD",
                 "TripLength is negative", n, total)

        n = run_count(con, "TRY_CAST(TripLength AS DOUBLE) = 0")
        add_rule(rules, file.name, "TRIP_LENGTH_ZERO", "REVIEW",
                 "TripLength equals zero; review rather than auto-delete", n, total)

        n = run_count(con, "TRY_CAST(TripLength AS DOUBLE) > 500")
        add_rule(rules, file.name, "TRIP_LENGTH_OVER_500KM", "REVIEW",
                 "TripLength > 500 km; review extreme trips", n, total)

    # VehicleType
    if "VehicleType" in cols:
        allowed = ",".join(sql_string(x) for x in sorted(EXPECTED_VEHICLE_TYPES))
        w = f"CAST(VehicleType AS VARCHAR) NOT IN ({allowed}) AND VehicleType IS NOT NULL"
        n = run_count(con, w)
        add_rule(rules, file.name, "UNEXPECTED_VEHICLE_TYPE", "REVIEW",
                 "VehicleType outside expected {31,32,41,42,5}", n, total)
        add_samples(con, sample_rows, file.name, "UNEXPECTED_VEHICLE_TYPE", "REVIEW", w,
                    ["VehicleID", "VehicleType", "BottleneckTime", "GantryO", "GantryD"])

    # Speed / volume basic validity
    for c in SPEED_COLS:
        if c not in cols:
            continue
        qc = qident(c)
        n = run_count(con, f"TRY_CAST({qc} AS DOUBLE) < 0")
        add_rule(rules, file.name, f"{c}_NEGATIVE", "HARD",
                 f"{c} is negative", n, total)
        n = run_count(con, f"TRY_CAST({qc} AS DOUBLE) > 180")
        add_rule(rules, file.name, f"{c}_OVER_180", "REVIEW",
                 f"{c} > 180 km/h; review extreme values", n, total)

    for c in VOLUME_COLS:
        if c not in cols:
            continue
        qc = qident(c)
        n = run_count(con, f"TRY_CAST({qc} AS DOUBLE) < 0")
        add_rule(rules, file.name, f"{c}_NEGATIVE", "HARD",
                 f"{c} is negative", n, total)

    # Vehicle-class speed-volume pairing
    for s, v in zip(SPEED_COLS, VOLUME_COLS):
        if s not in cols or v not in cols:
            continue
        qs, qv = qident(s), qident(v)
        n = run_count(con,
            f"TRY_CAST({qv} AS DOUBLE) = 0 AND TRY_CAST({qs} AS DOUBLE) > 0")
        add_rule(rules, file.name, f"{s}_POSITIVE_WHEN_VOLUME_ZERO", "REVIEW",
                 f"{s}>0 while corresponding volume is 0", n, total)

        n = run_count(con,
            f"TRY_CAST({qv} AS DOUBLE) > 0 AND TRY_CAST({qs} AS DOUBLE) = 0")
        add_rule(rules, file.name, f"{s}_ZERO_WHEN_VOLUME_POSITIVE", "REVIEW",
                 f"{s}=0 while corresponding volume >0", n, total)

    # Total volume consistency
    if "M05_Volume_Total" in cols and all(c in cols for c in VOLUME_COLS):
        sumvol = " + ".join(f"COALESCE(TRY_CAST({qident(c)} AS DOUBLE),0)" for c in VOLUME_COLS)
        w = f"ABS(COALESCE(TRY_CAST(M05_Volume_Total AS DOUBLE),0) - ({sumvol})) > 0.001"
        n = run_count(con, w)
        add_rule(rules, file.name, "M05_VOLUME_TOTAL_MISMATCH", "HARD",
                 "M05_Volume_Total differs from sum of class volumes", n, total)
        add_samples(con, sample_rows, file.name, "M05_VOLUME_TOTAL_MISMATCH", "HARD", w,
                    ["Time5", *VOLUME_COLS, "M05_Volume_Total"])

    # Weighted speed recomputation consistency
    if "M05_Speed_Weighted" in cols and all(c in cols for c in SPEED_COLS + VOLUME_COLS):
        num_terms = []
        den_terms = []
        for s, v in zip(SPEED_COLS, VOLUME_COLS):
            qs, qv = qident(s), qident(v)
            valid = f"TRY_CAST({qv} AS DOUBLE) > 0 AND TRY_CAST({qs} AS DOUBLE) IS NOT NULL"
            num_terms.append(
                f"CASE WHEN {valid} THEN TRY_CAST({qs} AS DOUBLE)*TRY_CAST({qv} AS DOUBLE) ELSE 0 END"
            )
            den_terms.append(
                f"CASE WHEN {valid} THEN TRY_CAST({qv} AS DOUBLE) ELSE 0 END"
            )
        num = " + ".join(num_terms)
        den = " + ".join(den_terms)
        weighted = f"(({num}) / NULLIF(({den}),0))"
        w = (
            f"({den}) > 0 AND "
            f"(M05_Speed_Weighted IS NULL OR "
            f"ABS(TRY_CAST(M05_Speed_Weighted AS DOUBLE) - {weighted}) > 0.01)"
        )
        n = run_count(con, w)
        add_rule(rules, file.name, "M05_WEIGHTED_SPEED_MISMATCH", "HARD",
                 "Stored M05_Speed_Weighted differs from recomputed weighted speed by >0.01 km/h", n, total)
        add_samples(con, sample_rows, file.name, "M05_WEIGHTED_SPEED_MISMATCH", "HARD", w,
                    ["Time5", *SPEED_COLS, *VOLUME_COLS, "M05_Speed_Weighted"])

    # Match flags
    for c in ["M05_Matched", "Weather_Matched", "Calendar_Matched"]:
        if c in cols:
            w = f"COALESCE(TRY_CAST({qident(c)} AS BOOLEAN), FALSE) = FALSE"
            n = run_count(con, w)
            severity = "HARD" if c == "Calendar_Matched" else "REVIEW"
            add_rule(rules, file.name, f"{c.upper()}_FALSE", severity,
                     f"Rows not successfully matched for {c}", n, total)

    # Weather physics / review ranges
    if "Weather_Precipitation_mm" in cols:
        n = run_count(con, "TRY_CAST(Weather_Precipitation_mm AS DOUBLE) < 0")
        add_rule(rules, file.name, "WEATHER_PRECIP_NEGATIVE", "HARD",
                 "Negative precipitation", n, total)
        n = run_count(con, "TRY_CAST(Weather_Precipitation_mm AS DOUBLE) > 100")
        add_rule(rules, file.name, "WEATHER_PRECIP_OVER_100MMH", "REVIEW",
                 "Hourly precipitation >100 mm; extreme, verify but do not auto-delete", n, total)

    if "Weather_WindSpeed_mps" in cols:
        n = run_count(con, "TRY_CAST(Weather_WindSpeed_mps AS DOUBLE) < 0")
        add_rule(rules, file.name, "WEATHER_WIND_NEGATIVE", "HARD",
                 "Negative wind speed", n, total)
        n = run_count(con, "TRY_CAST(Weather_WindSpeed_mps AS DOUBLE) > 40")
        add_rule(rules, file.name, "WEATHER_WIND_OVER_40MPS", "REVIEW",
                 "Wind speed >40 m/s; extreme, verify", n, total)

    if "Weather_Temperature_C" in cols:
        n = run_count(con, "TRY_CAST(Weather_Temperature_C AS DOUBLE) < -5 OR TRY_CAST(Weather_Temperature_C AS DOUBLE) > 45")
        add_rule(rules, file.name, "WEATHER_TEMP_EXTREME", "REVIEW",
                 "Temperature outside -5 to 45 C; review", n, total)

    if {"Weather_Dewpoint_C", "Weather_Temperature_C"}.issubset(cols):
        n = run_count(con,
            "TRY_CAST(Weather_Dewpoint_C AS DOUBLE) > TRY_CAST(Weather_Temperature_C AS DOUBLE) + 0.5")
        add_rule(rules, file.name, "DEWPOINT_ABOVE_TEMPERATURE", "REVIEW",
                 "Dew point exceeds temperature by >0.5 C", n, total)

    # Calendar internal consistency
    if {"CalendarDate", "WeekdayNum"}.issubset(cols):
        n = run_count(con, "TRY_CAST(WeekdayNum AS INTEGER) <> isodow(CalendarDate)")
        add_rule(rules, file.name, "WEEKDAYNUM_MISMATCH", "HARD",
                 "WeekdayNum does not match CalendarDate", n, total)

    if {"CalendarDate", "IsWeekend"}.issubset(cols):
        n = run_count(con,
            "TRY_CAST(IsWeekend AS INTEGER) <> CASE WHEN isodow(CalendarDate) IN (6,7) THEN 1 ELSE 0 END")
        add_rule(rules, file.name, "ISWEEKEND_MISMATCH", "HARD",
                 "IsWeekend inconsistent with CalendarDate", n, total)

    if {"CalendarDate", "IsSummer"}.issubset(cols):
        n = run_count(con,
            "TRY_CAST(IsSummer AS INTEGER) <> CASE WHEN month(CalendarDate) IN (7,8) THEN 1 ELSE 0 END")
        add_rule(rules, file.name, "ISSUMMER_MISMATCH", "HARD",
                 "IsSummer inconsistent with July-August operational definition", n, total)

    if "DayType" in cols:
        allowed = ",".join(sql_string(x) for x in sorted(EXPECTED_DAY_TYPES))
        w = f"DayType IS NOT NULL AND CAST(DayType AS VARCHAR) NOT IN ({allowed})"
        n = run_count(con, w)
        add_rule(rules, file.name, "UNEXPECTED_DAYTYPE", "HARD",
                 "DayType outside expected categories", n, total)

    # Expected route / gantry pair checks
    expected = EXPECTED_SEGMENTS.get(fkey)
    if expected:
        if "BottleneckGantry" in cols:
            w = f"BottleneckGantry IS NOT NULL AND CAST(BottleneckGantry AS VARCHAR) <> {sql_string(expected['bottleneck_gantry'])}"
            n = run_count(con, w)
            add_rule(rules, file.name, "BOTTLENECK_GANTRY_MISMATCH", "HARD",
                     f"BottleneckGantry differs from expected {expected['bottleneck_gantry']}", n, total)

        from_col = first_existing(cols, ["M05_GantryFrom", "GantryFrom"])
        to_col = first_existing(cols, ["M05_GantryTo", "GantryTo"])
        if from_col:
            w = f"{qident(from_col)} IS NOT NULL AND CAST({qident(from_col)} AS VARCHAR) <> {sql_string(expected['m05_from'])}"
            n = run_count(con, w)
            add_rule(rules, file.name, "M05_GANTRY_FROM_MISMATCH", "HARD",
                     f"M05 upstream gantry differs from expected {expected['m05_from']}", n, total)
        if to_col:
            w = f"{qident(to_col)} IS NOT NULL AND CAST({qident(to_col)} AS VARCHAR) <> {sql_string(expected['m05_to'])}"
            n = run_count(con, w)
            add_rule(rules, file.name, "M05_GANTRY_TO_MISMATCH", "HARD",
                     f"M05 downstream gantry differs from expected {expected['m05_to']}", n, total)

    # M05 consistency within the same 5-minute key (one road state per Time5)
    if "Time5" in cols and "M05_Speed_Weighted" in cols:
        n = con.execute("""
            SELECT COUNT(*)
            FROM (
                SELECT Time5
                FROM t
                GROUP BY Time5
                HAVING COUNT(DISTINCT M05_Speed_Weighted) > 1
            ) x
        """).fetchone()[0]
        add_rule(rules, file.name, "MULTIPLE_WEIGHTED_SPEEDS_PER_TIME5", "HARD",
                 "A Time5 interval has more than one M05_Speed_Weighted value within a direction file", n,
                 con.execute("SELECT COUNT(DISTINCT Time5) FROM t").fetchone()[0])

    # Optional exact duplicate check
    if RUN_EXPENSIVE_DUPLICATE_CHECK:
        dup_cols = [
            c for c in [
                "VehicleID", "DetectionTimeO", "GantryO", "DetectionTimeD",
                "GantryD", "TripLength", "TripEnd", "TripInformation"
            ] if c in cols
        ]
        if dup_cols:
            group_cols = ", ".join(qident(c) for c in dup_cols)
            ndup = con.execute(f"""
                SELECT COALESCE(SUM(cnt - 1),0)
                FROM (
                    SELECT COUNT(*) AS cnt
                    FROM t
                    GROUP BY {group_cols}
                    HAVING COUNT(*) > 1
                ) d
            """).fetchone()[0]
            add_rule(rules, file.name, "EXACT_DUPLICATE_TRIPS", "REVIEW",
                     "Repeated identical trip records across core M06B fields", ndup, total)

    con.close()

    return {
        "summary": summary,
        "schema": schema_rows,
        "missing": missing_rows,
        "numeric": numeric_rows,
        "categories": category_rows,
        "rules": rules,
        "samples": sample_rows,
    }

# ============================================================
# 4. Main
# ============================================================

def main():
    files = sorted(MASTER_ROOT.glob(FILE_PATTERN))
    if not files:
        # Fall back in case the user kept the old MASTER suffix.
        files = sorted(MASTER_ROOT.glob("*.parquet"))

    if not files:
        raise FileNotFoundError(f"No parquet files found under: {MASTER_ROOT}")

    print("=" * 80)
    print("V1 CORE MASTER - FIRST PASS QC")
    print("=" * 80)
    print("Input :", MASTER_ROOT)
    print("Output:", OUT_ROOT)
    print("Files :", len(files))
    print("Duplicate check:", RUN_EXPENSIVE_DUPLICATE_CHECK)

    summaries = []
    schemas = []
    missings = []
    numerics = []
    category_frames = []
    rules = []
    sample_frames = []

    for i, file in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] QC: {file.name}")
        res = qc_one_file(file)
        summaries.append(res["summary"])
        schemas.extend(res["schema"])
        missings.extend(res["missing"])
        numerics.extend(res["numeric"])
        category_frames.extend(res["categories"])
        rules.extend(res["rules"])
        sample_frames.extend(res["samples"])

    pd.DataFrame(summaries).to_csv(
        OUT_ROOT / "01_file_summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(schemas).to_csv(
        OUT_ROOT / "02_schema.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(missings).to_csv(
        OUT_ROOT / "03_missingness.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(numerics).to_csv(
        OUT_ROOT / "04_numeric_quantiles.csv", index=False, encoding="utf-8-sig"
    )

    if category_frames:
        pd.concat(category_frames, ignore_index=True).to_csv(
            OUT_ROOT / "05_category_counts.csv", index=False, encoding="utf-8-sig"
        )
    else:
        pd.DataFrame().to_csv(
            OUT_ROOT / "05_category_counts.csv", index=False, encoding="utf-8-sig"
        )

    rules_df = pd.DataFrame(rules)
    if not rules_df.empty:
        rules_df["flag_rate_pct"] = rules_df["flag_rate"] * 100
        rules_df = rules_df.sort_values(
            ["severity", "n_flagged", "file", "rule_id"],
            ascending=[True, False, True, True]
        )
    rules_df.to_csv(
        OUT_ROOT / "06_rule_summary.csv", index=False, encoding="utf-8-sig"
    )

    if sample_frames:
        pd.concat(sample_frames, ignore_index=True, sort=False).to_csv(
            OUT_ROOT / "07_flag_samples.csv", index=False, encoding="utf-8-sig"
        )
    else:
        pd.DataFrame().to_csv(
            OUT_ROOT / "07_flag_samples.csv", index=False, encoding="utf-8-sig"
        )

    # Compact rule file containing only non-zero flags
    if not rules_df.empty:
        rules_df[rules_df["n_flagged"] > 0].to_csv(
            OUT_ROOT / "08_nonzero_flags_only.csv", index=False, encoding="utf-8-sig"
        )

    # Small text recap for quick inspection
    with open(OUT_ROOT / "QC_README.txt", "w", encoding="utf-8") as f:
        f.write("V1 Core Master QC outputs\n")
        f.write("========================\n\n")
        f.write("01_file_summary.csv      : row counts, date range, schema completeness\n")
        f.write("02_schema.csv            : actual columns and types\n")
        f.write("03_missingness.csv       : missing count/rate for every column\n")
        f.write("04_numeric_quantiles.csv : key numeric min/quantiles/max/mean/sd\n")
        f.write("05_category_counts.csv   : VehicleType, DayType, match flags, etc.\n")
        f.write("06_rule_summary.csv      : all QC rules and flag rates\n")
        f.write("07_flag_samples.csv      : examples of flagged records\n")
        f.write("08_nonzero_flags_only.csv: only rules with flagged records\n\n")
        f.write("Important: this is a flagging pass, NOT an automatic deletion pass.\n")

    print("\n" + "=" * 80)
    print("QC finished")
    print("=" * 80)
    print("Please upload these small files for review:")
    print("  01_file_summary.csv")
    print("  03_missingness.csv")
    print("  04_numeric_quantiles.csv")
    print("  06_rule_summary.csv")
    print("  08_nonzero_flags_only.csv")
    print("  07_flag_samples.csv (if not too large)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[ERROR]", repr(e), file=sys.stderr)
        raise
