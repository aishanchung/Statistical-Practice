#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rebuild holiday-phase statistics/chart from an existing V3 EDA output.

This does not rescan the original 16-million-trip Parquet file. It uses the
five-minute calendar-aligned OD counts already saved in od_5min_join_y.csv.
"""

from pathlib import Path
import argparse

import pandas as pd

from v3_eda_full import holiday_phase_chart


def restyle(folder: Path) -> None:
    dates_path = folder / "holiday_phase_dates.csv"
    bins_path = folder / "od_5min_join_y.csv"
    if not dates_path.exists() or not bins_path.exists():
        raise FileNotFoundError("Need holiday_phase_dates.csv and od_5min_join_y.csv in the output folder")

    dates = pd.read_csv(dates_path, parse_dates=["Date"])
    special = dates.Date.eq(pd.Timestamp("2024-01-01")) & dates.phase.eq("single")
    dates["phase_source"] = "within_V3_calendar"
    dates.loc[special, "phase"] = "last"
    dates.loc[special, "total"] = 3
    dates.loc[special, "phase_source"] = "official_cross_year_holiday"
    dates.to_csv(dates_path, index=False, encoding="utf-8-sig")

    bins = pd.read_csv(bins_path, parse_dates=["Time5"])
    bins["Date"] = bins.Time5.dt.normalize()
    bins["hour"] = bins.Time5.dt.hour
    merged = bins.merge(dates[["Date", "phase"]], on="Date", how="inner", validate="many_to_one")
    n_days = dates.groupby(["Year", "phase"], as_index=False).size().rename(columns={"size": "days"})
    agg = merged.groupby(["Year", "phase", "hour"], as_index=False).od_trips_5min.sum()
    agg = agg.merge(n_days, on=["Year", "phase"], validate="many_to_one")
    agg["avg_trips_per_day_hour"] = agg.od_trips_5min / agg.days
    data = agg[["Year", "phase", "hour", "avg_trips_per_day_hour"]].sort_values(
        ["Year", "phase", "hour"])
    data.to_csv(folder / "holiday_phase_hourly.csv", index=False, encoding="utf-8-sig")
    holiday_phase_chart(data, folder / "holiday_phase_hourly.png")

    summary = folder / "summary.md"
    if summary.exists():
        original = summary.read_text(encoding="utf-8")
        section = "\n## Holiday-phase correction (2026-11-05)\n- 2024-01-01 is the last day of the 2023-12-30–2024-01-01 three-day holiday, not a standalone holiday.\n- 2024 uses solid lines; 2025 dashed lines; first/middle/last use matched colors across years.\n"
        if "Holiday-phase correction (2026-11-05)" not in original:
            summary.write_text(original.rstrip() + "\n" + section, encoding="utf-8")
    print("Updated holiday_phase_dates.csv, holiday_phase_hourly.csv and .png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("output_folder", type=Path, help="Existing V3 EDA result directory")
    restyle(p.parse_args().output_folder)
