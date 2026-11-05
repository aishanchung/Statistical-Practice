#!/usr/bin/env python3
"""Rebuild the holiday V3 Y using any 1–4 observed VDs, without rescanning OD.

Input: an existing V3_LITE_holiday_full output folder from
build_v3_lite_holiday.py (old strict-4 Y is fine). Its trip-level and
OD-by-pair Parquet outputs are left unchanged. All four missing -> Y=NA.

Example:
  python rebuild_v3_y_available.py --existing-v3 outputs/V3_LITE_holiday_full \
    --output outputs/V3_LITE_holiday_Y_available --c 0.50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd

from build_v3_lite_holiday import audit_vd, y_and_daily_coverage


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--existing-v3", required=True, help="Folder containing the previous V3_Y and VD Parquet")
    parser.add_argument("--output", required=True, help="New output folder: existing V3 is never overwritten")
    parser.add_argument("--c", type=float, default=0.50)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory-limit", default="4GB")
    args = parser.parse_args(argv)
    if not (0 < args.c <= 1):
        parser.error("--c must be within (0, 1]")
    src = Path(args.existing_v3).expanduser().resolve()
    out = Path(args.output).expanduser().resolve()
    if src == out:
        parser.error("--output must differ from --existing-v3 to preserve original results")
    paths = {"VD": src / "V3_LITE_VD_4SITES.parquet",
             "Y": src / "V3_Y_5MIN.parquet",
             "calendar": src / "V3_calendar_all_dates.csv"}
    for label, path in paths.items():
        if not path.is_file():
            parser.error(f"Missing {label} source: {path}")
    if (out / "V3_Y_5MIN.parquet").exists():
        parser.error("--output already contains V3_Y_5MIN.parquet; select a new folder")
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    try:
        con.execute(f"SET threads={args.threads}")
        con.execute("SET memory_limit=?", [args.memory_limit])
        grid = con.execute("SELECT * FROM read_parquet(?)", [str(paths["VD"])]).df()
        old_y = con.execute("SELECT Time5,od_trips_5min,Year,DayType FROM read_parquet(?)",
                            [str(paths["Y"])]).df()
        if grid.duplicated(["Time5", "VDID"]).any():
            raise ValueError("Duplicate station/time rows in existing V3")
        if len(grid) != len(old_y) * 4 or old_y.Time5.duplicated().any():
            raise ValueError("VD grid and old Y timeline have inconsistent sizes")
        if not pd.Index(grid.Time5.unique()).difference(old_y.Time5).empty:
            raise ValueError("VD grid contains times that are missing from old Y")
        calendar = pd.read_csv(paths["calendar"], parse_dates=["Date"])
        required = {"Date", "calendar_source", "daytype_conflict", "od_zero_records"}
        if not required.issubset(calendar.columns):
            raise ValueError(f"Calendar missing columns: {sorted(required - set(calendar.columns))}")
        audit_vd(grid, out, args)
        y_and_daily_coverage(con, grid, old_y[["Time5", "od_trips_5min"]].copy(), calendar, args, out)

        oldconfig = src / "V3_run_config.json"
        metadata = json.loads(oldconfig.read_text(encoding="utf-8")) if oldconfig.exists() else {}
        metadata.pop("strict_missing_rule", None)
        metadata.update({"c": args.c,
                         "minimum_valid_vd": 1,
                         "missing_rule": "Any observed VD: renormalize observed weights; all 4 missing: Y=NA",
                         "Y_rebuilt_from": str(src),
                         "Y_rebuild_only": True,
                         "original_OD_parquet_unchanged": True})
        (out / "V3_run_config.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"DONE: New Y saved in {out}; original OD and V3 left unchanged.", flush=True)
    finally:
        con.close()


if __name__ == "__main__":
    main()
