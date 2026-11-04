#!/usr/bin/env python3
"""Four-VD congestion labels and candidate-c sensitivity (no future information).

Input can be a long-format five-minute VD CSV/Parquet (one row per detector/time)
or a wide-format table (one speed column per detector).  No missing speed is
silently treated as free-flow traffic.

Example (wide):
  python four_vd_c_sensitivity.py --input vd.csv --time-col Time5 \
    --wide-speeds K69.510 K65.990 K63.990 K61.670 --output outputs/vd_c

Example (long):
  python four_vd_c_sensitivity.py --input vd.parquet --time-col Time5 \
    --site-col VDID --speed-col Speed --output outputs/vd_c

Run --inspect first if your data columns or station IDs differ.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SITES = ("69.510", "65.990", "63.990", "61.670")
WEIGHTS = {"69.510": 0.0417, "65.990": 0.4600, "63.990": 0.3600, "61.670": 0.1383}
TIMES = ("Time5", "DataCollectTime", "DataTime", "DateTime", "Time", "Timestamp", "timestamp", "time", "time5")
SITE_NAMES = ("VDID", "VDId", "VD_ID", "VD", "station", "Station", "site", "mile", "Mileage")
SPEED_NAMES = ("Speed", "speed", "AvgSpeed", "AverageSpeed", "SpeedAvg", "speed_kmh", "V")
FLOW_NAMES = ("Volume", "volume", "Flow", "flow", "TotalVolume", "VehicleCount", "TotalFlow")


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input", required=True, help="VD Parquet/CSV file or directory of Parquet/CSV parts")
    p.add_argument("--output", default="outputs/four_vd_c_sensitivity")
    p.add_argument("--inspect", action="store_true", help="Show columns/sample station values without analysis")
    p.add_argument("--time-col", help="Column containing timestamp (five-minute bins are generated)")
    p.add_argument("--site-col", help="Detector ID/mileage column in long input")
    p.add_argument("--speed-col", help="Speed km/h column in long input")
    p.add_argument("--flow-col", help="Optional volume to combine multiple rows per VD/time by weighted speed")
    p.add_argument("--station-map", help="Optional CSV: raw_id,kilometer for non-mileage VD identifiers")
    p.add_argument("--wide-speeds", nargs=4, metavar=("K69", "K65", "K63", "K61"),
                   help="Four speed columns for 69.510,65.990,63.990,61.670 respectively")
    p.add_argument("--c-values", nargs="+", type=float, default=[0.25, 0.50, 0.75])
    p.add_argument("--train-year", type=int, default=2024, help="Year for main c selection report")
    p.add_argument("--min-valid", type=int, default=4, choices=[4],
                   help="Strict primary Y requires all four station speeds")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--memory-limit", default="4GB")
    return p.parse_args()


def files_from_input(spec: str) -> list[Path]:
    folder = Path(spec)
    if folder.is_dir():
        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in (".csv", ".parquet"))
    elif folder.is_file():
        files = [folder]
    else:
        raise FileNotFoundError(f"VD input not found: {spec}")
    if not files:
        raise ValueError(f"No .csv or .parquet VD data under {spec}")
    if len({p.suffix.lower() for p in files}) > 1:
        raise ValueError("Put CSV and Parquet in separate VD folders; mixed formats are not supported")
    return files


def read_vd(paths: list[Path], args: argparse.Namespace, limit: int | None = None) -> pd.DataFrame:
    conn = duckdb.connect(database=":memory:")
    conn.execute(f"SET threads={args.threads}")
    conn.execute(f"SET memory_limit='{args.memory_limit}'")
    all_paths = [str(p.resolve()) for p in paths]
    relation = "read_parquet(?)" if paths[0].suffix.lower() == ".parquet" else "read_csv_auto(?)"
    sql = f"SELECT * FROM {relation}"
    if limit:
        sql += f" LIMIT {int(limit)}"
    try:
        return conn.execute(sql, [all_paths]).df()
    finally:
        conn.close()


def choose_column(df: pd.DataFrame, explicit: str | None, candidates: tuple[str, ...], purpose: str) -> str:
    if explicit:
        if explicit not in df:
            raise ValueError(f"{purpose} column {explicit!r} not found. Columns: {df.columns.tolist()}")
        return explicit
    for candidate in candidates:
        if candidate in df:
            return candidate
    lowered = {str(col).lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    raise ValueError(f"Cannot identify {purpose} column. Use --{purpose}-col. Columns: {df.columns.tolist()}")


def normalize_site(value: object) -> str | None:
    if pd.isna(value):
        return None
    raw = str(value).strip().upper()
    try:
        numeric_mile = float(raw.strip("K "))
    except ValueError:
        numeric_mile = float("nan")
    if math.isfinite(numeric_mile):
        for site in SITES:
            if abs(numeric_mile - float(site)) < 0.0005:
                return site
    for site in SITES:
        # Exact mile: K69.510, 69.510K, 069.510 or 69.510.
        if raw.strip("K ") in (site, "0" + site):
            return site
        integer, decimals = site.split(".")
        # Common VD identifiers with a road prefix / suffix and embedded mileage.
        pattern = rf"(?<!\d)0?{integer}[.]?{decimals}(?!\d)"
        if re.search(pattern, raw):
            return site
    return None


def parse_time(values: pd.Series) -> pd.Series:
    result = pd.to_datetime(values, errors="coerce")
    # Explicit timezone-aware strings are converted to the project's Taiwan time.
    if isinstance(result.dtype, pd.DatetimeTZDtype):
        result = result.dt.tz_convert("Asia/Taipei").dt.tz_localize(None)
    return result.dt.floor("5min")


def prepare_wide(raw: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    time_col = choose_column(raw, args.time_col, TIMES, "time")
    times = parse_time(raw[time_col])
    if args.wide_speeds:
        missing = [c for c in args.wide_speeds if c not in raw.columns]
        if missing:
            raise ValueError(f"Wide speed columns not found: {missing}")
        cols = args.wide_speeds
        data = pd.DataFrame({"Time5": times})
        for site, col in zip(SITES, cols):
            data[site] = pd.to_numeric(raw[col], errors="coerce")
        # Duplicate five-minute rows: plain mean; provide already aggregated wide speed if needed.
        data = data.groupby("Time5", dropna=True, as_index=True).mean(numeric_only=True)
    else:
        station_col = choose_column(raw, args.site_col, SITE_NAMES, "site")
        speed_col = choose_column(raw, args.speed_col, SPEED_NAMES, "speed")
        data = pd.DataFrame({"Time5": times, "station": raw[station_col],
                             "speed": pd.to_numeric(raw[speed_col], errors="coerce")})
        if args.station_map:
            mapping = pd.read_csv(args.station_map, dtype={"raw_id": str})
            if not {"raw_id", "kilometer"}.issubset(mapping.columns):
                raise ValueError("--station-map must have columns raw_id,kilometer")
            mapping_dict = {str(k): normalize_site(v) for k, v in
                            zip(mapping.raw_id, mapping.kilometer)}
            data["site"] = data.station.astype(str).map(mapping_dict)
        else:
            data["site"] = data.station.map(normalize_site)
        if data.site.notna().sum() == 0:
            samples = raw[station_col].dropna().astype(str).unique()[:15].tolist()
            raise ValueError("Cannot match any of the four VD mileages in station IDs. "
                             f"Seen {samples}; use --station-map with raw_id,kilometer.")
        data = data.loc[data.site.notna() & data.Time5.notna()].copy()
        if args.flow_col:
            flow_col = choose_column(raw, args.flow_col, FLOW_NAMES, "flow")
            data["volume"] = pd.to_numeric(raw.loc[data.index, flow_col], errors="coerce")
            data["volume"] = data["volume"].clip(lower=0)
            # Missing-speed rows must contribute neither speed nor volume.
            data.loc[data.speed.isna(), "volume"] = np.nan
            data["weighted_speed"] = data["speed"] * data["volume"]
            grouped = data.groupby(["Time5", "site"]).agg(
                sum_weight=("weighted_speed", "sum"), total_volume=("volume", "sum"),
                mean_speed=("speed", "mean"))
            grouped["speed"] = np.where(grouped.total_volume > 0,
                                         grouped.sum_weight / grouped.total_volume, grouped.mean_speed)
            speeds = grouped["speed"]
        else:
            speeds = data.groupby(["Time5", "site"])["speed"].mean()
        data = speeds.unstack("site")
    data = data.reindex(columns=list(SITES))
    data.index = pd.DatetimeIndex(data.index)
    if data.empty:
        raise ValueError("No valid times in the VD data")
    # Include time slots for which all four VD rows are absent; do not mistake them for free flow.
    expected = pd.date_range(data.index.min(), data.index.max(), freq="5min")
    data = data.reindex(expected)
    data.index.name = "Time5"
    data = data.apply(pd.to_numeric, errors="coerce")
    return data.mask(~np.isfinite(data) | (data < 0))


def y_col(c: float) -> str:
    return "Y_c" + str(round(c * 100)).zfill(3)


def event_stats(labels: pd.Series) -> tuple[int, float, float]:
    severe = labels.eq(2).fillna(False).to_numpy(bool)
    if not severe.any():
        return 0, 0.0, 0.0
    boundaries = np.diff(np.r_[False, severe, False].astype(int))
    durations = (np.where(boundaries == -1)[0] - np.where(boundaries == 1)[0]) * 5
    return int(len(durations)), float(np.median(durations)), float(np.max(durations))


def analyze(data: pd.DataFrame, thresholds: list[float], train_year: int, out: Path) -> None:
    if not np.isclose(sum(WEIGHTS.values()), 1.0, atol=1e-9):
        raise AssertionError("The four VD weights must sum to one")
    valid = data[list(SITES)].notna().all(axis=1)
    binary = data[list(SITES)].lt(40).where(data[list(SITES)].notna(), np.nan)
    weighted = binary.mul(pd.Series(WEIGHTS), axis="columns").sum(axis=1).where(valid)
    observed = data.rename_axis("Time5").reset_index()
    observed["n_valid_vd"] = data.notna().sum(axis=1).to_numpy()
    observed["C_weighted"] = weighted.to_numpy()
    observed["year"] = observed.Time5.dt.year
    observed["month"] = observed.Time5.dt.to_period("M").astype(str)
    for c in thresholds:
        labels = pd.Series(np.nan, index=data.index, dtype=float)
        labels.loc[valid] = np.where(weighted.loc[valid].eq(0), 0,
                                      np.where(weighted.loc[valid].ge(c), 2, 1))
        observed[y_col(c)] = labels.to_numpy()

    observed.to_csv(out / "four_vd_y_timeseries.csv", index=False, encoding="utf-8-sig")

    yearly = []
    monthly = []
    for year, block in observed.groupby("year"):
        n_all = len(block)
        for c in thresholds:
            label = block[y_col(c)]
            n_valid = int(label.notna().sum())
            groups = label.value_counts()
            events, median_minutes, max_minutes = event_stats(label)
            yearly.append({"year": int(year), "c": c, "expected_bins": n_all,
                           "complete_four_vd_bins": n_valid, "missing_bins": n_all - n_valid,
                           "valid_fraction": n_valid / n_all if n_all else np.nan,
                           "n_Y0": int(groups.get(0.0, 0)), "n_Y1": int(groups.get(1.0, 0)),
                           "n_Y2": int(groups.get(2.0, 0)),
                           "pct_Y0": float(groups.get(0.0, 0) / n_valid) if n_valid else np.nan,
                           "pct_Y1": float(groups.get(1.0, 0) / n_valid) if n_valid else np.nan,
                           "pct_Y2": float(groups.get(2.0, 0) / n_valid) if n_valid else np.nan,
                           "Y2_event_count": events, "Y2_event_median_min": median_minutes,
                           "Y2_event_max_min": max_minutes})
        for month, mblock in block.groupby("month"):
            for c in thresholds:
                series = mblock[y_col(c)]
                n = int(series.notna().sum())
                monthly.append({"year": int(year), "month": month, "c": c,
                                "n_valid": n, "n_missing": len(series) - n,
                                "pct_Y0": (series.eq(0).sum() / n if n else np.nan),
                                "pct_Y1": (series.eq(1).sum() / n if n else np.nan),
                                "pct_Y2": (series.eq(2).sum() / n if n else np.nan)})
    annual_df = pd.DataFrame(yearly)
    annual_df.to_csv(out / "c_yearly_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(monthly).to_csv(out / "c_monthly_comparison.csv", index=False, encoding="utf-8-sig")

    # Four on/off stations -> at most 16 spatial-coverage states.
    # Missing timestamps are excluded below, so their temporary indicator is irrelevant.
    patterns = data[list(SITES)].lt(40).astype(int)
    complete = patterns.loc[valid].copy()
    complete["year"] = complete.index.year
    complete["pattern"] = complete[list(SITES)].astype(str).agg("".join, axis=1)
    pattern_counts = complete.groupby(["year", "pattern"]).size().reset_index(name="n_bins")
    combo_map = {"".join(map(str, bits)): sum(v * WEIGHTS[site] for v, site in zip(bits, SITES))
                 for bits in itertools.product([0, 1], repeat=4)}
    pattern_counts["C_weighted"] = pattern_counts.pattern.map(combo_map)
    pattern_counts.to_csv(out / "spatial_pattern_counts.csv", index=False, encoding="utf-8-sig")

    main = annual_df.loc[annual_df.year.eq(train_year)].sort_values("c")
    if main.empty:
        raise ValueError(f"Training year {train_year} has no timestamp observations")
    if main.complete_four_vd_bins.max() == 0:
        raise ValueError(f"No complete four-VD five-minute bins in training year {train_year}; "
                         "check station IDs and missing data before comparing c")
    print("\nFour-VD primary c sensitivity (training year only):")
    print(main[["year", "c", "complete_four_vd_bins", "missing_bins", "n_Y0", "n_Y1", "n_Y2",
                "pct_Y0", "pct_Y1", "pct_Y2", "Y2_event_count", "Y2_event_median_min"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("Note: 2025 output is DESCRIPTIVE ONLY; do not select c from the future test year.")

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    x = np.arange(len(main))
    bottom = np.zeros(len(main))
    for name, title, color in [("pct_Y0", "Y=0", "#2A9D8F"),
                               ("pct_Y1", "Y=1", "#E9C46A"),
                               ("pct_Y2", "Y=2", "#E76F51")]:
        vals = main[name].to_numpy(float)
        ax.bar(x, vals, bottom=bottom, label=title, color=color)
        bottom += vals
    ax.set_xticks(x, [f"c={v:g}" for v in main.c]); ax.set_ylim(0, 1)
    ax.set(ylabel="Share of valid 5-min bins", title=f"Four-VD Y classes vs c ({train_year})")
    ax.legend(loc="upper right"); fig.tight_layout()
    fig.savefig(out / "c_comparison_train.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = arguments()
    if any(not (0 < c <= 1) or not math.isfinite(c) for c in args.c_values):
        raise ValueError("Every c must be finite and within (0, 1]")
    thresholds = sorted(set(args.c_values))
    if len({y_col(c) for c in thresholds}) != len(thresholds):
        raise ValueError("Distinct c values must differ by at least one percentage point for Y column labels")
    paths = files_from_input(args.input)
    if args.inspect:
        sample = read_vd(paths, args, limit=15)
        print(f"Input files: {len(paths)}")
        print("Columns:", sample.columns.tolist())
        print(sample.head(5).to_string(index=False))
        for col in SITE_NAMES:
            if col in sample:
                print(f"Example {col} station IDs:", sample[col].astype(str).unique()[:12].tolist())
        print("\nUse --time-col, --site-col, --speed-col or --wide-speeds to resolve your schema.")
        return
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    raw = read_vd(paths, args)
    data = prepare_wide(raw, args)
    analyze(data, thresholds, args.train_year, out)
    with (out / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump({"input": args.input, "n_files": len(paths), "sites": list(SITES),
                   "weights": WEIGHTS, "c_values": thresholds, "train_year": args.train_year,
                   "missing_policy": "strict: all four must be present", "timestamp_alignment": "floor 5min",
                   "time_col": args.time_col, "site_col": args.site_col,
                   "speed_col": args.speed_col, "wide_speeds": args.wide_speeds,
                   "station_map": args.station_map}, f, ensure_ascii=False, indent=2)
    print(f"\nOutputs: {out.resolve()}")


if __name__ == "__main__":
    main()
