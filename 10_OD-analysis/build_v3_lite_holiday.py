#!/usr/bin/env python3
"""Build holiday-focused V3_LITE and audit *expected* four-VD coverage.

Core definitions:
  - Date/DayType: OD's BottleneckTime date and modal DayType per calendar date.
    Saturday/Sunday with no OD records are inferred WEEKEND. Missing weekday
    LONGHOLIDAY dates require an optional authoritative --calendar-csv.
  - Included: WEEKEND or LONGHOLIDAY dates, full 00:00-24:00 days ONLY.
    There is no separate afternoon window or peak-hour flag.
  - VD: K69.510, K65.990, K63.990, K61.670 (K61.000 excluded).
  - Missing: expected 5-minute grid, including missing rows and invalid speeds.
  - Y: strict 4/4 observed, speed < 40, weighted C, 3 classes with c=0.5.

OD records and their day labels need not be observed for every expected date;
zero OD records are explicitly a *coverage flag*, NOT proof of a missing file.
No forecast model should treat retrospective BottleneckTime counts as known in
real time before the corresponding trip is completed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


SITES = {
    "VD-N3-N-69.510-M-LOOP": ("69.510", 0.0417),
    "VD-N3-N-65.990-M-RS": ("65.990", 0.4600),
    "VD-N3-N-63.990-M-LOOP": ("63.990", 0.3600),
    "VD-N3-N-61.670-M-RS": ("61.670", 0.1383),
}
FOCUS = {"WEEKEND", "LONGHOLIDAY"}
DAYTYPES = {"WEEKDAY", "WEEKEND", "LONGHOLIDAY", "STANDALONEHOLIDAY"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--od-input", required=True, help="91 V2 Parquet parts (directory) or one .parquet")
    p.add_argument("--vd-input", required=True, help="Five-station VD LITE Parquet file/directory")
    p.add_argument("--output", default="outputs/V3_LITE_HOLIDAY")
    p.add_argument("--start-date", default="2024-01-01")
    p.add_argument("--end-date", default="2025-12-31")
    p.add_argument("--calendar-csv", help="Optional authoritative overrides: Date,DayType (YYYY-MM-DD)")
    p.add_argument("--c", type=float, default=0.50, help="Threshold for Y=2 (weighted C >= c)")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--memory-limit", default="4GB")
    p.add_argument("--audit-only", action="store_true", help="Skip trip-level and OD-pair V3 exports")
    p.add_argument("--skip-trip-level", action="store_true", help="Keep only OD pair/5-minute counts")
    return p.parse_args(argv)


def parquet_files(spec: str) -> list[Path]:
    src = Path(spec).expanduser()
    if src.is_file() and src.suffix.lower() == ".parquet":
        result = [src.resolve()]
    elif src.is_dir():
        result = sorted(src.glob("part_*.parquet")) or sorted(src.glob("*.parquet"))
        result = [p.resolve() for p in result]
    else:
        result = []
    if not result:
        raise FileNotFoundError(f"No Parquet input: {spec}")
    return result


def quote(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def source_sql(files: list[Path]) -> str:
    return "read_parquet([" + ",".join(quote(p.as_posix()) for p in files) + "], union_by_name=true)"


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def save_parquet(con: duckdb.DuckDBPyConnection, frame: pd.DataFrame, path: Path) -> None:
    con.register("v3_export_frame", frame)
    try:
        con.execute(f"COPY v3_export_frame TO {quote(path.as_posix())} (FORMAT PARQUET, COMPRESSION ZSTD)")
    finally:
        con.unregister("v3_export_frame")


def save_query_parquet(con: duckdb.DuckDBPyConnection, query: str, path: Path) -> None:
    con.execute(f"COPY ({query}) TO {quote(path.as_posix())} (FORMAT PARQUET, COMPRESSION ZSTD)")


def calendar_from_od(con: duckdb.DuckDBPyConnection, args: argparse.Namespace,
                     out: Path) -> pd.DataFrame:
    # The V2 DayType is sometimes inconsistent even within one calendar date.
    # Explicitly report conflicts and use the modal label with a deterministic tie rule.
    print("[1/5] Auditing OD dates/DayType (out-of-core)...", flush=True)
    counts = con.execute("""
        SELECT CAST(BottleneckTime AS DATE) AS Date,
               UPPER(TRIM(COALESCE(DayType, 'UNKNOWN'))) AS DayType,
               COUNT(*) AS n_records
          FROM od_src
         WHERE BottleneckTime >= ? AND BottleneckTime < ?
         GROUP BY 1, 2
         ORDER BY 1, 2
    """, [pd.Timestamp(args.start_date), pd.Timestamp(args.end_date) + pd.Timedelta(days=1)]).df()
    save_csv(counts, out / "OD_daytype_counts.csv")
    if not counts.empty:
        ranks = counts.copy()
        weekend = ranks.Date.dt.dayofweek >= 5
        ranks["calendar_match"] = np.where(weekend,
                                             ranks.DayType.isin({"WEEKEND", "LONGHOLIDAY"}),
                                             ranks.DayType.isin({"WEEKDAY", "LONGHOLIDAY", "STANDALONEHOLIDAY"}))
        # Modal DayType wins; a tie is broken using the calendar weekday,
        # and both candidate labels remain in OD_daytype_conflicts.csv.
        ranks = ranks.sort_values(["Date", "n_records", "calendar_match", "DayType"],
                                  ascending=[True, False, False, True])
        winner = ranks.drop_duplicates("Date")["Date DayType".split()].rename(
            columns={"DayType": "OD_modal_DayType"})
        total = counts.groupby("Date", as_index=False).n_records.sum().rename(
            columns={"n_records": "od_records"})
        diversity = counts.groupby("Date", as_index=False).DayType.nunique().rename(
            columns={"DayType": "n_distinct_daytypes"})
        conflicts = counts[counts.Date.isin(diversity.loc[diversity.n_distinct_daytypes > 1, "Date"])]
        cal = winner.merge(total, on="Date").merge(diversity, on="Date")
    else:
        conflicts = counts.copy()
        cal = pd.DataFrame(columns=["Date", "OD_modal_DayType", "od_records", "n_distinct_daytypes"])
    save_csv(conflicts, out / "OD_daytype_conflicts.csv")

    expected = pd.DataFrame({"Date": pd.date_range(args.start_date, args.end_date, freq="D")})
    cal = expected.merge(cal, on="Date", how="left")
    cal["od_records"] = cal.od_records.fillna(0).astype("int64")
    cal["n_distinct_daytypes"] = cal.n_distinct_daytypes.fillna(0).astype(int)
    cal["DayType"] = cal.OD_modal_DayType
    cal["calendar_source"] = np.where(cal.od_records > 0, "OD_MODAL", "NOT_IDENTIFIED")
    inferred = (cal.od_records == 0) & (cal.Date.dt.dayofweek >= 5)
    cal.loc[inferred, "DayType"] = "WEEKEND"
    cal.loc[inferred, "calendar_source"] = "WEEKDAY_INFERRED_NO_OD"
    # A weekend must not disappear from the focus merely because V2 labelled
    # all observed trips as WEEKDAY. Flag this instead of concealing it.
    mismatch = (cal.od_records > 0) & (cal.Date.dt.dayofweek >= 5) & cal.DayType.eq("WEEKDAY")
    cal.loc[mismatch, "DayType"] = "WEEKEND"
    cal.loc[mismatch, "calendar_source"] = "WEEKEND_CALENDAR_OD_MISMATCH"

    if args.calendar_csv:
        overrides = pd.read_csv(args.calendar_csv, dtype=str)
        if not {"Date", "DayType"}.issubset(overrides.columns):
            raise ValueError("--calendar-csv must contain Date,DayType")
        overrides = overrides[["Date", "DayType"]].copy()
        overrides["Date"] = pd.to_datetime(overrides.Date, errors="raise").dt.normalize()
        overrides["DayType"] = overrides.DayType.str.strip().str.upper()
        if overrides.Date.duplicated().any():
            raise ValueError("--calendar-csv contains duplicate Dates")
        bad = sorted(set(overrides.DayType.dropna()) - DAYTYPES)
        if bad or overrides.DayType.isna().any():
            raise ValueError(f"Invalid calendar DayType: {bad}")
        cal = cal.merge(overrides.rename(columns={"DayType": "override"}), on="Date", how="left")
        is_override = cal.override.notna()
        cal.loc[is_override, "DayType"] = cal.loc[is_override, "override"]
        cal.loc[is_override, "calendar_source"] = "CALENDAR_CSV"
        cal = cal.drop(columns="override")

    cal["Date"] = pd.to_datetime(cal.Date).dt.normalize()
    cal["is_focus_date"] = cal.DayType.isin(FOCUS)
    cal["Year"] = cal.Date.dt.year
    cal["daytype_conflict"] = cal.n_distinct_daytypes > 1
    cal["od_zero_records"] = cal.od_records == 0
    save_csv(cal, out / "V3_calendar_all_dates.csv")
    save_csv(cal[cal.is_focus_date], out / "V3_calendar_focus_dates.csv")
    save_csv(cal[(cal.od_zero_records & cal.Date.dt.dayofweek.lt(5)) |
                 cal.daytype_conflict |
                 cal.calendar_source.eq("WEEKEND_CALENDAR_OD_MISMATCH")],
             out / "OD_dates_needing_calendar_review.csv")
    missing_weekday = int((cal.od_zero_records & cal.Date.dt.dayofweek.lt(5)).sum())
    if missing_weekday:
        print(f"WARNING: {missing_weekday} weekday(s) have no OD records. Their holiday "
              "status cannot be inferred; supply --calendar-csv if applicable.", flush=True)
    print(f"  {cal.is_focus_date.sum()} focus dates, "
          f"{cal.daytype_conflict.sum()} conflicting OD DayType dates.", flush=True)
    return cal


def od_exports(con: duckdb.DuckDBPyConnection, cal: pd.DataFrame,
               args: argparse.Namespace, out: Path) -> pd.DataFrame:
    print("[2/5] Aggregating OD trips in expected holiday dates...", flush=True)
    focus = cal.loc[cal.is_focus_date, ["Date", "DayType"]].copy()
    con.register("v3_focus_dates", focus)
    join = "JOIN v3_focus_dates c ON CAST(o.BottleneckTime AS DATE) = c.Date"
    periods = "time_bucket(INTERVAL '5 minutes', o.BottleneckTime)"
    # Five-minute OD counts are observational; an empty bin is not automatically
    # a missing source file. Retain zero bins in the Y grid later.
    bins = con.execute(f"""
        SELECT {periods} AS Time5, COUNT(*) AS od_trips_5min
          FROM od_src o {join}
         GROUP BY 1
    """).df()
    if not args.audit_only:
        q = f"""
            SELECT {periods} AS Time5, c.DayType AS V3_DayType,
                   o.GantryO, o.GantryD, o.VehicleType,
                   COUNT(*) AS TripCount
              FROM od_src o {join}
             GROUP BY 1, 2, 3, 4, 5
        """
        save_query_parquet(con, q, out / "V3_OD_5MIN_BY_PAIR.parquet")
        if not args.skip_trip_level:
            raw_cols = [r[0] for r in con.execute("DESCRIBE SELECT * FROM od_src").fetchall()]
            keep = [x for x in raw_cols if x not in {"VD_61p000_Speed", "VD_61p000_Occupancy"}]
            select_cols = ", ".join("o.\"" + c.replace('"', '""') + "\"" for c in keep)
            q = f"SELECT {select_cols}, c.DayType AS V3_DayType FROM od_src o {join}"
            print("  Writing filtered trip-level V3_LITE (may take several minutes)...", flush=True)
            save_query_parquet(con, q, out / "V3_LITE_OD_TRIPS.parquet")
    return bins


def vd_expected(con: duckdb.DuckDBPyConnection, cal: pd.DataFrame,
                args: argparse.Namespace, out: Path) -> pd.DataFrame:
    print("[3/5] Building complete four-site × five-minute VD grid...", flush=True)
    focus = cal.loc[cal.is_focus_date, ["Date", "DayType"]]
    all_times = pd.DatetimeIndex(np.concatenate([
        pd.date_range(day, periods=288, freq="5min").values for day in focus.Date
    ])) if not focus.empty else pd.DatetimeIndex([])
    times = pd.DataFrame({"Time5": all_times})
    times["Date"] = times.Time5.dt.normalize()
    times = times.merge(focus, on="Date", how="left", validate="many_to_one")
    site_frame = pd.DataFrame([{"VDID": key, "SiteKM": value[0], "weight": value[1]}
                               for key, value in SITES.items()])
    grid = times.merge(site_frame, how="cross")
    con.register("v3_focus_dates_vd", focus)
    site_sql = ",".join(quote(site) for site in SITES)
    vd = con.execute(f"""
        SELECT time_bucket(INTERVAL '5 minutes', v.TargetTime) AS Time5,
               v.VDID, v.Speed, v.Occupancy, v.VDType
          FROM vd_src v
          JOIN v3_focus_dates_vd c ON CAST(v.TargetTime AS DATE) = c.Date
         WHERE v.VDID IN ({site_sql})
    """).df()
    dup = vd[vd.duplicated(["Time5", "VDID"], keep=False)].sort_values(["Time5", "VDID"])
    save_csv(dup, out / "VD_duplicate_station_time.csv")
    if not dup.empty:
        raise ValueError("Duplicate four-VD station/time records; see VD_duplicate_station_time.csv")
    grid = grid.merge(vd, on=["Time5", "VDID"], how="left", validate="one_to_one", indicator=True)
    grid["Speed"] = pd.to_numeric(grid.Speed, errors="coerce")
    grid["MissingStatus"] = np.select([
        grid._merge.eq("left_only"), grid.Speed.isna() | grid.Speed.lt(0),
    ], ["ROW_ABSENT", "INVALID_SPEED"], default="VALID")
    grid = grid.drop(columns="_merge")
    grid["Year"] = grid.Time5.dt.year
    save_parquet(con, grid, out / "V3_LITE_VD_4SITES.parquet")
    return grid


def missing_runs(rows: pd.DataFrame, grouping: list[str],
                 scope: str, joint: bool = False) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    work = rows.copy().sort_values(grouping + ["Time5"])
    grouped = work.groupby(grouping, dropna=False, sort=False)
    breaks = grouped.Time5.diff().ne(pd.Timedelta(minutes=5))
    work["run_id"] = breaks.cumsum()
    keys = grouping + ["run_id"]
    records = work.groupby(keys, as_index=False, sort=False).agg(
        start_time=("Time5", "min"), end_last_bin=("Time5", "max"),
        n_5min_bins=("Time5", "size"))
    records["end_exclusive"] = records.end_last_bin + pd.Timedelta(minutes=5)
    records["duration_min"] = records.n_5min_bins * 5
    records["scope"] = scope
    if not joint:
        reasons = work.groupby(keys, as_index=False).agg(
            row_absent_bins=("MissingStatus", lambda x: int(x.eq("ROW_ABSENT").sum())),
            invalid_speed_bins=("MissingStatus", lambda x: int(x.eq("INVALID_SPEED").sum())))
        records = records.merge(reasons, on=keys, how="left")
    return records.drop(columns="run_id")


def audit_vd(grid: pd.DataFrame, out: Path, args: argparse.Namespace) -> pd.DataFrame:
    print("[4/5] Computing holiday missing rates by date/site and contiguous runs...", flush=True)
    grid = grid.copy()
    grid["is_missing"] = grid.MissingStatus.ne("VALID")
    day_records = []
    summary_records = []
    daily_overview = []
    station_runs = []
    joint_runs = []
    # Single scope only: every included holiday's entire 24 hours.
    for scope, subset in [("full_day", grid)]:
        # Actual denominator: expected time bins x 4 sites, never just observed rows.
        day = subset.groupby(["Year", "Date", "DayType", "SiteKM"], as_index=False).agg(
            expected_bins=("Time5", "size"), valid_bins=("MissingStatus", lambda x: int(x.eq("VALID").sum())),
            row_absent_bins=("MissingStatus", lambda x: int(x.eq("ROW_ABSENT").sum())),
            invalid_speed_bins=("MissingStatus", lambda x: int(x.eq("INVALID_SPEED").sum())))
        day["missing_bins"] = day.expected_bins - day.valid_bins
        day["missing_pct"] = 100 * day.missing_bins / day.expected_bins
        day["scope"] = scope
        day_records.append(day)
        by_date = subset.groupby(["Year", "Date", "DayType", "Time5"], as_index=False).agg(
            n_missing=("is_missing", "sum"))
        date_agg = by_date.groupby(["Year", "Date", "DayType"], as_index=False).agg(
            expected_5min_bins=("Time5", "size"), missing_site_bins=("n_missing", "sum"))
        date_agg["expected_site_bins"] = 4 * date_agg.expected_5min_bins
        date_agg["site_missing_pct"] = 100 * date_agg.missing_site_bins / date_agg.expected_site_bins
        date_agg["scope"] = scope
        for n in range(5):
            num = by_date.loc[by_date.n_missing.eq(n)].groupby("Date").size()
            date_agg[f"bins_missing_{n}_sites"] = date_agg.Date.map(num).fillna(0).astype(int)
        daily_overview.append(date_agg)

        joint = subset.groupby(["Year", "DayType", "Time5"], as_index=False).agg(
            n_missing=("is_missing", "sum"))
        for (year, daytype), part in joint.groupby(["Year", "DayType"]):
            n_t = len(part)
            rec = {"Year": int(year), "DayType": daytype, "scope": scope,
                   "expected_time_bins": n_t, "expected_site_bins": n_t * 4,
                   "complete_4vd_bins": int(part.n_missing.eq(0).sum())}
            for n in range(1, 5):
                rec[f"bins_missing_{n}_sites"] = int(part.n_missing.eq(n).sum())
            rec["missing_site_bins"] = int(part.n_missing.sum())
            rec["missing_pct"] = 100 * rec["missing_site_bins"] / rec["expected_site_bins"] if n_t else np.nan
            rec["strict_Y_unavailable_pct"] = 100 * (1 - rec["complete_4vd_bins"] / n_t) if n_t else np.nan
            summary_records.append(rec)

        station_rows = subset.loc[subset.is_missing, ["SiteKM", "Time5", "MissingStatus"]]
        if not station_rows.empty:
            station_runs.append(missing_runs(station_rows, ["SiteKM"], scope))
        sets = subset.loc[subset.is_missing].groupby("Time5").SiteKM.apply(
            lambda x: ",".join(sorted(x, reverse=True))).reset_index(name="missing_sites")
        if not sets.empty:
            sets["n_missing_sites"] = sets.missing_sites.str.count(",") + 1
            joint_runs.append(missing_runs(sets, ["missing_sites", "n_missing_sites"], scope, joint=True))

    station_daily = pd.concat(day_records, ignore_index=True) if day_records else pd.DataFrame()
    save_csv(station_daily, out / "VD_missing_by_date_station.csv")
    save_csv(pd.concat(daily_overview, ignore_index=True) if daily_overview else pd.DataFrame(),
             out / "VD_missing_by_date.csv")
    if not station_daily.empty:
        by_station = station_daily.groupby(["Year", "DayType", "SiteKM", "scope"], as_index=False)[
            ["expected_bins", "valid_bins", "row_absent_bins", "invalid_speed_bins", "missing_bins"]].sum()
        by_station["missing_pct"] = 100 * by_station.missing_bins / by_station.expected_bins
        save_csv(by_station, out / "VD_missing_by_station_summary.csv")
        monthly = station_daily.copy()
        monthly["Month"] = pd.to_datetime(monthly.Date).dt.strftime("%Y-%m")
        by_month = monthly.groupby(["Month", "DayType", "SiteKM", "scope"], as_index=False)[
            ["expected_bins", "valid_bins", "row_absent_bins", "invalid_speed_bins", "missing_bins"]].sum()
        by_month["missing_pct"] = 100 * by_month.missing_bins / by_month.expected_bins
        save_csv(by_month, out / "VD_missing_by_month_station.csv")
    else:
        save_csv(pd.DataFrame(), out / "VD_missing_by_station_summary.csv")
        save_csv(pd.DataFrame(), out / "VD_missing_by_month_station.csv")
    save_csv(pd.DataFrame(summary_records), out / "VD_missing_summary.csv")
    save_csv(pd.concat(station_runs, ignore_index=True) if station_runs else pd.DataFrame(),
             out / "VD_missing_runs_by_station.csv")
    save_csv(pd.concat(joint_runs, ignore_index=True) if joint_runs else pd.DataFrame(),
             out / "VD_missing_runs_joint.csv")
    return grid


def y_and_daily_coverage(con: duckdb.DuckDBPyConnection, grid: pd.DataFrame,
                         od_bins: pd.DataFrame, cal: pd.DataFrame,
                         args: argparse.Namespace, out: Path) -> None:
    print("[5/5] Writing strict four-VD Y and date-level availability...", flush=True)
    frame = grid.copy()
    frame["weighted_congested"] = np.where(frame.MissingStatus.eq("VALID"),
                                            np.where(frame.Speed < 40, frame.weight, 0.0), 0.0)
    y = frame.groupby(["Time5", "Date", "DayType", "Year"], as_index=False).agg(
        n_valid_vd=("MissingStatus", lambda x: int(x.eq("VALID").sum())),
        C_weighted_sum=("weighted_congested", "sum"))
    y["C_weighted"] = y.C_weighted_sum.where(y.n_valid_vd.eq(4))
    y = y.drop(columns="C_weighted_sum")
    labels = np.select([y.C_weighted.eq(0), y.C_weighted.ge(args.c)], [0, 2], default=1)
    y["Y"] = pd.Series(labels, dtype="Int8").where(y.n_valid_vd.eq(4))
    y["n_missing_vd"] = 4 - y.n_valid_vd
    od_bins["Time5"] = pd.to_datetime(od_bins.Time5)
    y = y.merge(od_bins, on="Time5", how="left", validate="one_to_one")
    y["od_trips_5min"] = y.od_trips_5min.fillna(0).astype("int64")
    y["od_zero_records_bin"] = y.od_trips_5min.eq(0)
    save_parquet(con, y, out / "V3_Y_5MIN.parquet")

    daily = y.groupby(["Date", "DayType", "Year"], as_index=False).agg(
        expected_5min_bins=("Time5", "size"), strict_Y_valid_bins=("n_valid_vd", lambda x: int(x.eq(4).sum())),
        all_4vd_missing_bins=("n_valid_vd", lambda x: int(x.eq(0).sum())),
        od_5min_zero_bins=("od_zero_records_bin", "sum"), od_records=("od_trips_5min", "sum"))
    daily["scope"] = "full_day"
    daily["strict_Y_missing_pct"] = 100 * (1 - daily.strict_Y_valid_bins / daily.expected_5min_bins)
    daily["od_zero_bin_pct_NOT_missing_proof"] = 100 * daily.od_5min_zero_bins / daily.expected_5min_bins
    daily = daily.merge(cal[["Date", "calendar_source", "daytype_conflict", "od_zero_records"]],
                        on="Date", how="left", validate="many_to_one")
    save_csv(daily, out / "V3_focus_date_coverage.csv")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    start = pd.Timestamp(args.start_date)
    end = pd.Timestamp(args.end_date)
    if start > end:
        raise ValueError("Check --start-date/--end-date")
    if not 0 < args.c <= 1:
        raise ValueError("--c must be in (0, 1]")
    if not np.isclose(sum(w for _, w in SITES.values()), 1.0):
        raise AssertionError("Four VD weights must sum to one")
    out = Path(args.output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    odfiles = parquet_files(args.od_input)
    vdfiles = parquet_files(args.vd_input)
    con = duckdb.connect()
    try:
        con.execute(f"SET threads={int(args.threads)}")
        con.execute(f"SET memory_limit={quote(args.memory_limit)}")
        con.execute(f"CREATE VIEW od_src AS SELECT * FROM {source_sql(odfiles)}")
        con.execute(f"CREATE VIEW vd_src AS SELECT * FROM {source_sql(vdfiles)}")
        od_columns = {x[0] for x in con.execute("DESCRIBE SELECT * FROM od_src").fetchall()}
        vd_columns = {x[0] for x in con.execute("DESCRIBE SELECT * FROM vd_src").fetchall()}
        if not {"BottleneckTime", "DayType", "GantryO", "GantryD", "VehicleType"} <= od_columns:
            raise ValueError(f"OD schema missing: {sorted({'BottleneckTime', 'DayType', 'GantryO', 'GantryD', 'VehicleType'} - od_columns)}")
        if not {"TargetTime", "VDID", "Speed", "Occupancy", "VDType"} <= vd_columns:
            raise ValueError(f"VD schema missing: {sorted({'TargetTime', 'VDID', 'Speed', 'Occupancy', 'VDType'} - vd_columns)}")
        cal = calendar_from_od(con, args, out)
        od_bins = od_exports(con, cal, args, out)
        grid = vd_expected(con, cal, args, out)
        audit_vd(grid, out, args)
        y_and_daily_coverage(con, grid, od_bins, cal, args, out)
        meta = {"od_input_files": len(odfiles), "vd_input_files": len(vdfiles),
                "start_date": args.start_date, "end_date": args.end_date,
                "focus_daytypes": sorted(FOCUS), "excluded_daytype": "STANDALONEHOLIDAY",
                "time_scope": "full_day_24h",
                "focus_dates": int(cal.is_focus_date.sum()),
                "focus_weekday_dates_with_zero_od": int((cal.od_zero_records & cal.Date.dt.dayofweek.lt(5)).sum()),
                "daytype_conflicts": int(cal.daytype_conflict.sum()),
                "VD_sites": list(SITES.keys()), "VD_weights": {v[0]: v[1] for v in SITES.values()},
                "speed_threshold_kmh": 40, "c": args.c,
                "strict_missing_rule": "All 4 stations must have Speed >= 0 (not null)",
                "trip_level_saved": not (args.audit_only or args.skip_trip_level),
                "model_leakage_warning": "BottleneckTime retrospective OD demand is NOT real-time prediction input"}
        (out / "V3_run_config.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"DONE: {out}\n  Focus dates: {meta['focus_dates']}, Y rows: {grid.Time5.nunique():,}", flush=True)
    finally:
        con.close()


if __name__ == "__main__":
    main()
