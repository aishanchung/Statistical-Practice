#!/usr/bin/env python3
r"""V3_LITE: concise OD-pair EDA before clustering (no clustering or Y leakage).

Example (PowerShell):
  python .\v3_od_quick_eda.py `
    --input "D:\統計實務\Data\V3_LITE\V3_LITE.parquet" `
    --calendar "D:\統計實務\Data\V3_LITE\V3_Y_5MIN.parquet" `
    --output "D:\\統計實務\\Data\\V3_LITE\\od_quick_eda"

Needs: pip install duckdb pandas numpy matplotlib
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input", default=r"D:\統計實務\Data\V3_LITE\V3_LITE.parquet",
                   help="V3 trip-level parquet, a directory of part_*.parquet, or a glob")
    p.add_argument("--calendar", default=r"D:\統計實務\Data\V3_LITE\V3_Y_5MIN.parquet",
                   help="V3_Y_5MIN.parquet; provides the authoritative Date/DayType calendar only")
    p.add_argument("--output", default=r"D:\統計實務\Data\V3_LITE\od_quick_eda")
    p.add_argument("--min-trips", type=int, default=300, help="Minimum 2024 trips for an OD pair")
    p.add_argument("--min-daytype-trips", type=int, default=30,
                   help="Minimum observations in EACH compared daytype/year for an interpretable comparison")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--memory-limit", default="4GB")
    return p.parse_args()


def paths(spec):
    p = Path(spec)
    if p.is_dir():
        found = sorted(p.glob("part_*.parquet")) or sorted(p.glob("*.parquet"))
    elif p.is_file():
        found = [p]
    else:
        found = [Path(x) for x in sorted(glob.glob(spec))]
    if not found:
        raise FileNotFoundError(f"找不到 Parquet：{spec}")
    nums = []
    for f in found:
        match = re.fullmatch(r"part_(\d+)\.parquet", f.name)
        if match:
            nums.append(int(match.group(1)))
    if len(nums) == len(found) and nums:
        absent = sorted(set(range(min(nums), max(nums) + 1)) - set(nums))
        if absent:
            raise ValueError(f"分檔編號不連續：{absent}，請補齊再執行。")
    return [str(x.resolve()).replace("\\", "/") for x in found]


def lit(s):
    """Quote a string for DuckDB SQL, without shell escaping pitfalls."""
    return "'" + str(s).replace("'", "''") + "'"


def parquet_sql(files):
    return "read_parquet([" + ",".join(map(lit, files)) + "], union_by_name=true)"


def save_csv(df, location):
    df.to_csv(location, index=False, encoding="utf-8-sig")


def jensen_shannon(a, b, pseudo=0.5):
    """JS divergence of two 6-bin counts, in bits; 0=identical, <=1."""
    a = np.asarray(a, float) + pseudo
    b = np.asarray(b, float) + pseudo
    a /= a.sum()
    b /= b.sum()
    m = (a + b) / 2
    return float(0.5 * (np.sum(a * np.log2(a / m)) + np.sum(b * np.log2(b / m))))


def counts_by_block(profile):
    x = profile.copy()
    x["block"] = (x["hour"] // 4).astype(int)
    return (x.groupby(["GantryO", "GantryD", "Year", "DayType", "block"], as_index=False)
            ["trips"].sum())


def main():
    args = parse_args()
    if args.min_trips <= 0 or args.min_daytype_trips <= 0:
        raise ValueError("--min-trips 與 --min-daytype-trips 須大於 0")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    try:
        import duckdb
    except ImportError as exc:
        raise SystemExit("缺少 duckdb。請先執行 python -m pip install duckdb pandas numpy matplotlib") from exc
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    src = paths(args.input)
    cal_files = paths(args.calendar)
    con = duckdb.connect()
    con.execute(f"SET threads={int(args.threads)}")
    con.execute(f"SET memory_limit={lit(args.memory_limit)}")

    print("[1/5] 檢查 V3 與日曆欄位", flush=True)
    od_src = parquet_sql(src)
    cal_src = parquet_sql(cal_files)
    od_schema = set(con.execute(f"DESCRIBE SELECT * FROM {od_src}").df()["column_name"])
    y_schema = set(con.execute(f"DESCRIBE SELECT * FROM {cal_src}").df()["column_name"])
    for required, cols in (({"GantryO", "GantryD", "BottleneckTime", "VehicleType", "TripLength"}, od_schema),
                           ({"Date", "DayType", "Year"}, y_schema)):
        if required - cols:
            raise ValueError(f"缺少必要欄位：{sorted(required - cols)}")

    # 只從 V3_Y_5MIN 取日期分類；不取 Y、速度、雨量等，避免把結果變數當作分群特徵。
    con.execute(f"""CREATE TEMP VIEW calendar_source AS
        SELECT CAST(Date AS DATE) AS Date, CAST(Year AS INTEGER) AS Year,
               UPPER(TRIM(CAST(DayType AS VARCHAR))) AS DayType
        FROM {cal_src}
        WHERE CAST(Year AS INTEGER) IN (2024,2025)
    """)
    conflicts = con.execute("""SELECT Date, COUNT(DISTINCT DayType) AS k
        FROM calendar_source GROUP BY Date HAVING k > 1""").df()
    if not conflicts.empty:
        raise ValueError(f"同一天出現不同 DayType，共 {len(conflicts)} 天；請檢查日曆。")
    con.execute("""CREATE TEMP TABLE calendar AS SELECT DISTINCT Date, Year, DayType
        FROM calendar_source WHERE DayType IN ('WEEKEND','LONGHOLIDAY')""")
    day_counts = con.execute("""SELECT Year,DayType,COUNT(*) AS days FROM calendar
                                  GROUP BY 1,2 ORDER BY 1,2""").df()
    if len(day_counts) != 4:
        raise ValueError(f"預期 2024/2025 各有 WEEKEND 與 LONGHOLIDAY；實際為\n{day_counts}")
    days = {(int(r.Year), str(r.DayType)): int(r.days) for r in day_counts.itertuples()}
    save_csv(day_counts, out / "00_calendar_days.csv")
    con.execute(f"""CREATE TEMP VIEW od AS
        SELECT CAST(t.GantryO AS VARCHAR) AS GantryO,
               CAST(t.GantryD AS VARCHAR) AS GantryD,
               t.BottleneckTime AS BottleneckTime,
               CAST(t.VehicleType AS VARCHAR) AS VehicleType,
               TRY_CAST(t.TripLength AS DOUBLE) AS TripLength,
               c.Year, c.DayType
        FROM {od_src} t
        INNER JOIN calendar c ON CAST(t.BottleneckTime AS DATE)=c.Date
        WHERE t.GantryO IS NOT NULL AND t.GantryD IS NOT NULL
          AND t.BottleneckTime IS NOT NULL
    """)

    print("[2/5] 所有 OD 配對、年度與日型數量", flush=True)
    counts = con.execute("""SELECT GantryO,GantryD,Year,DayType,COUNT(*) AS trips
        FROM od GROUP BY 1,2,3,4""").df()
    save_csv(counts, out / "01_od_pair_year_daytype_counts.csv")
    wide = counts.pivot_table(index=["GantryO", "GantryD"], columns=["Year", "DayType"],
                              values="trips", fill_value=0, aggfunc="sum")
    wide.columns = [f"n_{year}_{kind.lower()}" for year, kind in wide.columns]
    wide = wide.reset_index()
    for year in (2024, 2025):
        for dt in ("weekend", "longholiday"):
            name = f"n_{year}_{dt}"
            if name not in wide:
                wide[name] = 0
        wide[f"n_{year}"] = wide[f"n_{year}_weekend"] + wide[f"n_{year}_longholiday"]
        wide[f"holiday_weekend_daily_ratio_{year}"] = (
            (wide[f"n_{year}_longholiday"] / days[year, "LONGHOLIDAY"])
            / (wide[f"n_{year}_weekend"] / days[year, "WEEKEND"]).replace(0, np.nan))
    wide = wide.sort_values("n_2024", ascending=False).reset_index(drop=True)
    save_csv(wide, out / "02_od_pair_counts_and_holiday_ratio.csv")
    n2024, n2025 = int(wide.n_2024.sum()), int(wide.n_2025.sum())
    coverage = []
    for minimum in sorted({100, args.min_trips, 500}):
        sel = wide.n_2024 >= minimum
        coverage.append({"min_2024_trips": minimum, "selected_pairs": int(sel.sum()),
                         "2024_coverage_pct": 100 * wide.loc[sel, "n_2024"].sum() / max(n2024, 1),
                         "2025_coverage_pct": 100 * wide.loc[sel, "n_2025"].sum() / max(n2025, 1)})
    save_csv(pd.DataFrame(coverage), out / "03_od_selection_coverage.csv")
    eligible = wide.loc[wide.n_2024 >= args.min_trips, ["GantryO", "GantryD"]].copy()
    if eligible.empty:
        raise ValueError("沒有任何 OD 達到最小 2024 旅次數；請調低 --min-trips。")
    con.register("eligible_frame", eligible)
    con.execute("CREATE TEMP TABLE eligible AS SELECT * FROM eligible_frame")

    print(f"[3/5] {len(eligible):,} 個合格 OD：24 小時輪廓", flush=True)
    hourly = con.execute("""SELECT t.GantryO,t.GantryD,t.Year,t.DayType,
        CAST(EXTRACT(HOUR FROM t.BottleneckTime) AS INTEGER) AS hour,COUNT(*) AS trips
        FROM od t JOIN eligible e USING(GantryO,GantryD)
        GROUP BY 1,2,3,4,5""").df()
    hourly["group_trips"] = hourly.groupby(["GantryO", "GantryD", "Year", "DayType"])["trips"].transform("sum")
    hourly["share_pct"] = 100 * hourly.trips / hourly.group_trips
    hourly["mean_trips_per_day_at_hour"] = hourly.apply(
        lambda row: row.trips / days[int(row.Year), row.DayType], axis=1)
    save_csv(hourly, out / "04_od_hourly_profiles.csv")

    print("[4/5] 車種比例與距離", flush=True)
    vehicle = con.execute("""SELECT t.GantryO,t.GantryD,t.Year,
        COALESCE(t.VehicleType,'MISSING') AS VehicleType,COUNT(*) AS trips
        FROM od t JOIN eligible e USING(GantryO,GantryD)
        GROUP BY 1,2,3,4""").df()
    vehicle["share_pct"] = 100 * vehicle.trips / vehicle.groupby(
        ["GantryO", "GantryD", "Year"]).trips.transform("sum")
    save_csv(vehicle, out / "05_od_vehicle_mix.csv")
    distance = con.execute("""SELECT t.GantryO,t.GantryD,
        COUNT(TripLength) AS n_distance,
        approx_quantile(TripLength,0.25) AS triplen_q25_raw,
        approx_quantile(TripLength,0.50) AS triplen_median_raw,
        approx_quantile(TripLength,0.75) AS triplen_q75_raw
        FROM od t JOIN eligible e USING(GantryO,GantryD)
        WHERE t.Year=2024 AND TripLength IS NOT NULL AND TripLength>=0
        GROUP BY 1,2""").df()
    save_csv(distance, out / "06_od_distance_2024.csv")

    print("[5/5] 週末/連假差異、跨年穩定度及圖表", flush=True)
    block = counts_by_block(hourly)
    # Only make comparisons on paired, sufficiently populated date groups.
    metric_rows = []
    for (origin, dest), g in block.groupby(["GantryO", "GantryD"]):
        sequences = {}
        ns = {}
        for yr in (2024, 2025):
            for dt in ("WEEKEND", "LONGHOLIDAY"):
                z = g[(g.Year == yr) & (g.DayType == dt)]
                arr = np.bincount(z.block.astype(int), weights=z.trips, minlength=6)
                sequences[yr, dt] = arr
                ns[yr, dt] = int(arr.sum())
        row = {"GantryO": origin, "GantryD": dest,
               "n_2024_weekend": ns[2024, "WEEKEND"],
               "n_2024_holiday": ns[2024, "LONGHOLIDAY"],
               "n_2025_weekend": ns[2025, "WEEKEND"],
               "n_2025_holiday": ns[2025, "LONGHOLIDAY"]}
        a, b = sequences[2024, "WEEKEND"], sequences[2024, "LONGHOLIDAY"]
        if min(ns[2024, "WEEKEND"], ns[2024, "LONGHOLIDAY"]) >= args.min_daytype_trips:
            row["js_weekend_holiday_2024"] = jensen_shannon(a, b)
            for i in range(6):
                row[f"holiday_minus_weekend_b{i}_pp"] = 100 * (b[i] / b.sum() - a[i] / a.sum())
        for dt, suffix in (("WEEKEND", "weekend"), ("LONGHOLIDAY", "holiday")):
            if min(ns[2024, dt], ns[2025, dt]) >= args.min_daytype_trips:
                row[f"js_2024_2025_{suffix}"] = jensen_shannon(sequences[2024, dt], sequences[2025, dt])
        metric_rows.append(row)
    metrics = pd.DataFrame(metric_rows)
    save_csv(metrics, out / "07_od_profile_differences.csv")

    # 2024 weekends: heatmap of leading 40 ODs, with rows ordered by afternoon share.
    z = hourly[(hourly.Year == 2024) & (hourly.DayType == "WEEKEND")]
    big40 = wide.nlargest(40, "n_2024_weekend")
    pair_order = list(big40[["GantryO", "GantryD"]].itertuples(index=False, name=None))
    pp = z.pivot_table(index=["GantryO", "GantryD"], columns="hour", values="share_pct", fill_value=0)
    pp = pp.reindex(columns=range(24), fill_value=0)
    order = [key for key in pair_order if key in pp.index]
    order.sort(key=lambda key: pp.loc[key, list(range(12, 20))].sum())
    fig, ax = plt.subplots(figsize=(12, max(6, len(order) * 0.23)))
    im = ax.imshow(pp.loc[order].to_numpy(), aspect="auto", interpolation="nearest", cmap="YlGnBu")
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels(range(0, 24, 2))
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{o} > {d}" for o, d in order], fontsize=6)
    ax.set(xlabel="Hour of day", ylabel="OD pair (top 40 by 2024 weekend volume)",
           title="2024 WEEKEND: within-OD hourly trip share (%)")
    fig.colorbar(im, ax=ax, label="Share of each OD's weekend trips (%)")
    fig.tight_layout()
    fig.savefig(out / "A_od_hourly_heatmap_2024_weekend.png", dpi=160)
    plt.close(fig)

    # Compare 2024 weekend vs long-holiday hourly proportions for five high-volume OD pairs.
    selected = pair_order[:5]
    fig, axes = plt.subplots(5, 1, figsize=(10, 11), sharex=True, sharey=True)
    for ax, (o, d) in zip(axes, selected):
        for dt, color, style in (("WEEKEND", "#2479a7", "-"), ("LONGHOLIDAY", "#e0782a", "--")):
            t = hourly[(hourly.GantryO == o) & (hourly.GantryD == d) & (hourly.Year == 2024) &
                       (hourly.DayType == dt)]
            values = t.set_index("hour").share_pct.reindex(range(24), fill_value=0)
            ax.plot(range(24), values, style, color=color, label=dt, linewidth=1.6)
        ax.set_title(f"{o} -> {d}", loc="left", fontsize=9)
        ax.grid(alpha=0.2)
    axes[0].legend(loc="upper left", ncol=2)
    axes[-1].set_xticks(range(0, 24, 2))
    axes[-1].set_xlabel("Hour of day")
    fig.supylabel("Within-daytype trip share (%)")
    fig.suptitle("Five high-volume ODs: 2024 weekend vs long holiday")
    fig.tight_layout()
    fig.savefig(out / "B_top5_weekend_vs_holiday_2024.png", dpi=160)
    plt.close(fig)

    differences = [f"holiday_minus_weekend_b{i}_pp" for i in range(6)]
    difference_data = metrics[differences].dropna()
    if len(difference_data):
        fig, ax = plt.subplots(figsize=(9, 4.5))
        centers = difference_data.median().to_numpy()
        q25 = difference_data.quantile(.25).to_numpy()
        q75 = difference_data.quantile(.75).to_numpy()
        xx = np.arange(6)
        ax.bar(xx, centers, color="#3683aa", alpha=.9, label="Median across ODs")
        ax.errorbar(xx, centers, yerr=[centers - q25, q75 - centers], fmt="none", color="black", capsize=4,
                    label="Interquartile range")
        ax.axhline(0, color="gray", linewidth=.8)
        ax.set_xticks(xx)
        ax.set_xticklabels(["00-03", "04-07", "08-11", "12-15", "16-19", "20-23"])
        ax.set(ylabel="Holiday minus weekend share (percentage points)",
               title=f"2024 OD-level holiday shifts (n={len(difference_data)} reliable pairs)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "C_2024_holiday_minus_weekend.png", dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for field, label, color in (("js_2024_2025_weekend", "WEEKEND", "#2479a7"),
                                ("js_2024_2025_holiday", "LONGHOLIDAY", "#e0782a")):
        data = metrics[field].dropna()
        if not data.empty:
            ax.hist(data, bins=30, density=True, alpha=.55, label=f"{label} (n={len(data)})", color=color)
    ax.set(xlabel="Jensen-Shannon divergence across 2024 vs 2025 (0 = identical; bits)",
           ylabel="Density", title="OD hourly-profile stability across years (six 4-hour bins)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "D_year_to_year_profile_stability.png", dpi=160)
    plt.close(fig)

    # Compact feature screening is for interpretation only, not fitted clusters.
    v2024 = vehicle[vehicle.Year == 2024]
    v2024_wide = v2024.pivot_table(index=["GantryO", "GantryD"], columns="VehicleType",
                                  values="share_pct", fill_value=0)
    v2024_wide.columns = ["vehicle_share_pct_" + str(x) for x in v2024_wide.columns]
    feature = wide.merge(metrics.drop(columns=[c for c in metrics if c.startswith("n_")]),
                         on=["GantryO", "GantryD"], how="left")
    feature = feature.merge(v2024_wide.reset_index(), on=["GantryO", "GantryD"], how="left")
    feature = feature.merge(distance, on=["GantryO", "GantryD"], how="left")
    save_csv(feature[feature.n_2024 >= args.min_trips], out / "08_od_feature_screen_2024.csv")

    def summarize(name):
        z = metrics[name].dropna()
        return f"{z.median():.4f} (n={len(z)})" if len(z) else "NA"

    note = ["# V3 OD-pair quick EDA (pre-clustering)", "",
            "## Coverage and scope", "- Source: V3_LITE trip-level Parquet; date classes from V3_Y_5MIN calendar.",
            "- Only 2024/2025 WEEKEND and LONGHOLIDAY, 24 hours. No Y, VD speed, or weather is used as a cluster feature.",
            f"- Input parts: {len(src)}; distinct OD pairs: {len(wide):,}.",
            f"- OD trips: 2024 = {n2024:,}; 2025 = {n2025:,}.",
            f"- Eligible ODs (2024 >= {args.min_trips:,} trips): {len(eligible):,}.",
            "", "## Exposure days (from calendar)"]
    for row in day_counts.itertuples():
        note.append(f"- {row.Year} {row.DayType}: {row.days} days")
    note += ["", "## OD-specific profile differences (Jensen-Shannon divergence, 6 x 4h)",
             f"- 2024 weekend vs holiday median JS: {summarize('js_weekend_holiday_2024')}",
             f"- 2024 vs 2025 weekend median JS: {summarize('js_2024_2025_weekend')}",
             f"- 2024 vs 2025 long holiday median JS: {summarize('js_2024_2025_holiday')}",
             f"- Comparisons require at least {args.min_daytype_trips} trips per side; small-sample ODs are NA.",
             "", "## Interpretation and cautions",
             "- A is each OD's 2024 weekend hourly share (not absolute volume).",
             "- B compares the same OD under weekends and long holidays; solid WEEKEND, dashed LONGHOLIDAY.",
             "- C describes typical holiday changes per four-hour block across OD pairs; whiskers are OD IQR, not CI.",
             "- D describes 2024–2025 profile similarity. A low JS is more stable; holidays have few 2024 days.",
             "- TripLength units are not converted; check original unit before interpreting numeric distances.",
             "- VehicleType code 31/32/etc. are codes; do not label them as vehicle classes without a codebook.",
             "- OD profiles use BottleneckTime; this is a historical description, not a guaranteed known-ahead predictor.",
             "- Hour shares use counts divided by *that OD and DayType's* trips, so changing day counts do not distort shape.",
             "- Compare holiday volume using per-day counts; raw holiday total is not comparable to weekend total.",
             "- Model training/clustering should be based on 2024; treat 2025 as held-out evaluation.",
             "- Weather and Y are excluded from clustering. Consider weather as a time-varying predictor later.", ""]
    (out / "README_results.md").write_text("\n".join(note), encoding="utf-8")
    print(f"完成：{out}")
    print("請先看 A、B、C、D 四張圖以及 README_results.md", flush=True)


if __name__ == "__main__":
    main()
