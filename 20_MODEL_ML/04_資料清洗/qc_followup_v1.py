# -*- coding: utf-8 -*-
"""
qc_followup_v1.py
=================

V1 Core Master 第二輪 QC
1) 確認 M05_Speed_Weighted 缺值是否只發生在 M05_Volume_Total = 0
2) 深入檢查 TripLength > 500 km
3) 檢查 VehicleType = 0 與 OwnType 欄位缺值是否一致
4) Exact duplicate trip check（可關閉）

不修改 Master，只輸出小型 QC CSV。

預設輸入：
D:\統計實務\Data\MASTER_V1_2024_2025\*_MASTER_V1.parquet

預設輸出：
D:\統計實務\Data\QC_V1_FOLLOWUP
"""

from pathlib import Path
import time
import pandas as pd
import duckdb


# ============================================================
# 1. 設定
# ============================================================

DATA_ROOT = Path(r"D:\統計實務\Data")
MASTER_ROOT = DATA_ROOT / "MASTER_V1_2024_2025"
OUT_ROOT = DATA_ROOT / "QC_V1_FOLLOWUP"
TEMP_ROOT = OUT_ROOT / "_duckdb_temp"

# 第一輪 QC 已通過後，建議這次打開一次。
# 若電腦資源不足，可先改 False，其他 follow-up QC 仍會正常完成。
RUN_EXACT_DUPLICATE_CHECK = True

# Long-trip 門檻
LONG_TRIP_KM = 500

# 最多存幾個 duplicate group 範例
DUP_SAMPLE_GROUPS = 100


# ============================================================
# 2. 小工具
# ============================================================

def sql_path(path: Path) -> str:
    """DuckDB SQL 中安全使用 Windows 路徑。"""
    return str(path).replace("\\", "/").replace("'", "''")


def save_csv(df: pd.DataFrame, name: str):
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    path = OUT_ROOT / name
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  saved: {path}")


def make_connection():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(database=":memory:")

    # 讓大型 GROUP BY 必要時可以 spill 到磁碟。
    con.execute(f"SET temp_directory='{sql_path(TEMP_ROOT)}'")
    con.execute("SET preserve_insertion_order=false")

    return con


# ============================================================
# 3. Weighted Speed NA QC
# ============================================================

def check_weighted_speed_na(con, file_name: str) -> dict:

    q = """
    SELECT
        COUNT(*) AS total_rows,

        COUNT(*) FILTER (
            WHERE M05_Speed_Weighted IS NULL
        ) AS weighted_na,

        COUNT(*) FILTER (
            WHERE M05_Speed_Weighted IS NULL
              AND COALESCE(M05_Volume_Total, 0) = 0
        ) AS weighted_na_volume0,

        COUNT(*) FILTER (
            WHERE M05_Speed_Weighted IS NULL
              AND COALESCE(M05_Volume_Total, 0) > 0
        ) AS weighted_na_volume_positive,

        COUNT(*) FILTER (
            WHERE M05_Speed_Weighted IS NOT NULL
              AND COALESCE(M05_Volume_Total, 0) = 0
        ) AS weighted_present_volume0

    FROM t
    """

    row = con.execute(q).fetchdf().iloc[0].to_dict()

    total = int(row["total_rows"])
    na = int(row["weighted_na"])
    bad_na = int(row["weighted_na_volume_positive"])
    bad_present = int(row["weighted_present_volume0"])

    return {
        "file": file_name,
        **row,
        "weighted_na_rate_pct": 100 * na / total if total else 0,
        "status": (
            "PASS"
            if bad_na == 0 and bad_present == 0
            else "REVIEW"
        ),
        "interpretation": (
            "Weighted-speed NA only occurs when total volume is zero"
            if bad_na == 0 and bad_present == 0
            else "Weighted speed / total volume relationship needs review"
        ),
    }


# ============================================================
# 4. VehicleType = 0 QC
# ============================================================

def check_unknown_vehicle_type(con, file_name: str) -> dict:

    q = """
    SELECT
        COUNT(*) FILTER (
            WHERE TRY_CAST(VehicleType AS VARCHAR) = '0'
        ) AS vehicle_type_0,

        COUNT(*) FILTER (
            WHERE TRY_CAST(VehicleType AS VARCHAR) = '0'
              AND M05_Speed_OwnType IS NULL
              AND M05_Volume_OwnType IS NULL
        ) AS type0_own_both_na,

        COUNT(*) FILTER (
            WHERE TRY_CAST(VehicleType AS VARCHAR) <> '0'
              AND (
                    M05_Speed_OwnType IS NULL
                 OR M05_Volume_OwnType IS NULL
              )
        ) AS expected_type_own_na
    FROM t
    """

    row = con.execute(q).fetchdf().iloc[0].to_dict()

    n0 = int(row["vehicle_type_0"])
    n0_na = int(row["type0_own_both_na"])
    unexpected = int(row["expected_type_own_na"])

    return {
        "file": file_name,
        **row,
        "status": (
            "PASS"
            if n0 == n0_na and unexpected == 0
            else "REVIEW"
        ),
        "interpretation": (
            "OwnType NA is fully explained by VehicleType=0"
            if n0 == n0_na and unexpected == 0
            else "OwnType NA has cases not explained by VehicleType=0"
        ),
    }


# ============================================================
# 5. TripLength > 500 km QC
# ============================================================

def check_long_trips(con, file_name: str):

    # --------------------------------------------------------
    # Summary by increasingly extreme thresholds
    # --------------------------------------------------------

    summary = con.execute(f"""
        SELECT
            COUNT(*) AS total_rows,
            COUNT(*) FILTER (WHERE TripLength > {LONG_TRIP_KM}) AS gt500,
            COUNT(*) FILTER (WHERE TripLength > 600) AS gt600,
            COUNT(*) FILTER (WHERE TripLength > 700) AS gt700,
            COUNT(*) FILTER (WHERE TripLength > 800) AS gt800,
            COUNT(*) FILTER (WHERE TripLength > 900) AS gt900,
            MAX(TripLength) AS max_trip_length,
            AVG(TripLength) FILTER (WHERE TripLength > {LONG_TRIP_KM}) AS mean_gt500
        FROM t
    """).fetchdf()

    summary.insert(0, "file", file_name)

    total = int(summary.loc[0, "total_rows"])
    gt500 = int(summary.loc[0, "gt500"])
    summary["gt500_rate_pct"] = 100 * gt500 / total if total else 0

    # --------------------------------------------------------
    # 將全部 >500 km 旅次匯出（只有約一萬多筆，規模可控）
    # --------------------------------------------------------

    long_trips = con.execute(f"""
        SELECT
            VehicleID,
            VehicleType,
            DetectionTimeO,
            GantryO,
            DetectionTimeD,
            GantryD,
            TripLength,
            TripEnd,
            BottleneckTime,
            SourceDate,
            SourceHour,
            CASE
                WHEN DetectionTimeO IS NOT NULL
                 AND DetectionTimeD IS NOT NULL
                THEN date_diff(
                    'second',
                    DetectionTimeO,
                    DetectionTimeD
                ) / 3600.0
            END AS TripDurationHours,
            CASE
                WHEN TripInformation IS NOT NULL
                THEN list_count(
                    string_split(
                        TRY_CAST(TripInformation AS VARCHAR),
                        ';'
                    )
                )
            END AS PathItemCount,
            TripInformation
        FROM t
        WHERE TripLength > {LONG_TRIP_KM}
        ORDER BY TripLength DESC
    """).fetchdf()

    long_trips.insert(0, "file", file_name)

    # --------------------------------------------------------
    # 長旅次 Top OD
    # --------------------------------------------------------

    top_od = con.execute(f"""
        SELECT
            GantryO,
            GantryD,
            COUNT(*) AS n,
            MIN(TripLength) AS min_km,
            MEDIAN(TripLength) AS median_km,
            MAX(TripLength) AS max_km
        FROM t
        WHERE TripLength > {LONG_TRIP_KM}
        GROUP BY GantryO, GantryD
        ORDER BY n DESC
        LIMIT 100
    """).fetchdf()

    top_od.insert(0, "file", file_name)

    # --------------------------------------------------------
    # 長旅次最常出現日期
    # --------------------------------------------------------

    top_dates = con.execute(f"""
        SELECT
            CAST(BottleneckTime AS DATE) AS Date,
            COUNT(*) AS n,
            MAX(TripLength) AS max_km
        FROM t
        WHERE TripLength > {LONG_TRIP_KM}
        GROUP BY 1
        ORDER BY n DESC
        LIMIT 100
    """).fetchdf()

    top_dates.insert(0, "file", file_name)

    return summary, long_trips, top_od, top_dates


# ============================================================
# 6. Exact duplicate QC
# ============================================================

def check_exact_duplicates(con, file_name: str):

    """
    Exact duplicate 定義：
    下列 M06B core fields 完全相同：
      VehicleID
      DetectionTimeO
      GantryO
      DetectionTimeD
      GantryD
      TripLength
      TripEnd
      TripInformation

    為降低成本：
    第一階段先用不含 TripInformation 的 7 欄找 candidate groups；
    第二階段只對 candidate groups 加上 TripInformation 做真正 exact check。
    """

    key7 = """
        VehicleID,
        DetectionTimeO,
        GantryO,
        DetectionTimeD,
        GantryD,
        TripLength,
        TripEnd
    """

    print("  duplicate step 1/3: candidate keys...")

    con.execute("DROP TABLE IF EXISTS candidate_keys")

    con.execute(f"""
        CREATE TEMP TABLE candidate_keys AS
        SELECT
            {key7},
            COUNT(*) AS n_pre
        FROM t
        GROUP BY
            {key7}
        HAVING COUNT(*) > 1
    """)

    candidate_groups = con.execute(
        "SELECT COUNT(*) FROM candidate_keys"
    ).fetchone()[0]

    if candidate_groups == 0:

        return (
            pd.DataFrame([{
                "file": file_name,
                "candidate_groups_7key": 0,
                "exact_duplicate_groups": 0,
                "duplicate_extra_rows": 0,
                "status": "PASS",
            }]),
            pd.DataFrame()
        )

    print(
        f"  duplicate step 2/3: {candidate_groups:,} candidate groups; "
        "checking TripInformation..."
    )

    # DuckDB join 將 7-key candidate 限縮後，再加入 TripInformation。
    con.execute("DROP TABLE IF EXISTS exact_duplicate_groups")

    con.execute(f"""
        CREATE TEMP TABLE exact_duplicate_groups AS
        SELECT
            t.VehicleID,
            t.DetectionTimeO,
            t.GantryO,
            t.DetectionTimeD,
            t.GantryD,
            t.TripLength,
            t.TripEnd,
            t.TripInformation,
            COUNT(*) AS n
        FROM t
        INNER JOIN candidate_keys c
        ON  t.VehicleID      IS NOT DISTINCT FROM c.VehicleID
        AND t.DetectionTimeO IS NOT DISTINCT FROM c.DetectionTimeO
        AND t.GantryO        IS NOT DISTINCT FROM c.GantryO
        AND t.DetectionTimeD IS NOT DISTINCT FROM c.DetectionTimeD
        AND t.GantryD        IS NOT DISTINCT FROM c.GantryD
        AND t.TripLength     IS NOT DISTINCT FROM c.TripLength
        AND t.TripEnd        IS NOT DISTINCT FROM c.TripEnd
        GROUP BY
            t.VehicleID,
            t.DetectionTimeO,
            t.GantryO,
            t.DetectionTimeD,
            t.GantryD,
            t.TripLength,
            t.TripEnd,
            t.TripInformation
        HAVING COUNT(*) > 1
    """)

    stats = con.execute("""
        SELECT
            COUNT(*) AS exact_duplicate_groups,
            COALESCE(SUM(n - 1), 0) AS duplicate_extra_rows,
            MAX(n) AS max_group_size
        FROM exact_duplicate_groups
    """).fetchdf().iloc[0].to_dict()

    exact_groups = int(stats["exact_duplicate_groups"])
    extra_rows = int(stats["duplicate_extra_rows"])

    print("  duplicate step 3/3: extracting samples...")

    samples = con.execute(f"""
        SELECT *
        FROM exact_duplicate_groups
        ORDER BY n DESC
        LIMIT {DUP_SAMPLE_GROUPS}
    """).fetchdf()

    if not samples.empty:
        samples.insert(0, "file", file_name)

    summary = pd.DataFrame([{
        "file": file_name,
        "candidate_groups_7key": int(candidate_groups),
        "exact_duplicate_groups": exact_groups,
        "duplicate_extra_rows": extra_rows,
        "max_group_size": stats.get("max_group_size"),
        "status": "PASS" if extra_rows == 0 else "REVIEW",
    }])

    return summary, samples


# ============================================================
# 7. MAIN
# ============================================================

def main():

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    files = sorted(
        MASTER_ROOT.glob("*_MASTER_V1.parquet")
    )

    if not files:
        raise FileNotFoundError(
            f"找不到 V1 Master：{MASTER_ROOT}"
        )

    print("=" * 80)
    print("V1 CORE MASTER - FOLLOW-UP QC")
    print("=" * 80)
    print("Input :", MASTER_ROOT)
    print("Output:", OUT_ROOT)
    print("Files :", len(files))
    print("Exact duplicate check:", RUN_EXACT_DUPLICATE_CHECK)

    weighted_results = []
    vehicle_results = []
    long_summaries = []
    long_trip_parts = []
    top_od_parts = []
    top_date_parts = []
    duplicate_summaries = []
    duplicate_samples = []

    for i, file in enumerate(files, 1):

        print("\n" + "=" * 80)
        print(f"[{i}/{len(files)}] {file.name}")
        print("=" * 80)

        t0 = time.time()

        con = make_connection()

        con.execute(f"""
            CREATE VIEW t AS
            SELECT *
            FROM read_parquet('{sql_path(file)}')
        """)

        # 1) weighted speed NA
        print("  [1] weighted-speed NA check")
        weighted_results.append(
            check_weighted_speed_na(
                con,
                file.name
            )
        )

        # 2) VehicleType 0
        print("  [2] VehicleType=0 check")
        vehicle_results.append(
            check_unknown_vehicle_type(
                con,
                file.name
            )
        )

        # 3) long trips
        print("  [3] long-trip check")
        s, d, od, dates = check_long_trips(
            con,
            file.name
        )

        long_summaries.append(s)
        long_trip_parts.append(d)
        top_od_parts.append(od)
        top_date_parts.append(dates)

        # 4) duplicates
        if RUN_EXACT_DUPLICATE_CHECK:
            print("  [4] exact-duplicate check")
            ds, dp = check_exact_duplicates(
                con,
                file.name
            )

            duplicate_summaries.append(ds)

            if not dp.empty:
                duplicate_samples.append(dp)

        con.close()

        elapsed = time.time() - t0
        print(
            f"  done in {elapsed/60:.1f} min"
        )

    # ========================================================
    # Save
    # ========================================================

    save_csv(
        pd.DataFrame(weighted_results),
        "09_weighted_speed_na_check.csv"
    )

    save_csv(
        pd.DataFrame(vehicle_results),
        "10_vehicle_type0_check.csv"
    )

    save_csv(
        pd.concat(
            long_summaries,
            ignore_index=True
        ),
        "11_long_trip_summary.csv"
    )

    save_csv(
        pd.concat(
            long_trip_parts,
            ignore_index=True
        ),
        "12_long_trips_all.csv"
    )

    save_csv(
        pd.concat(
            top_od_parts,
            ignore_index=True
        ),
        "13_long_trip_top_od.csv"
    )

    save_csv(
        pd.concat(
            top_date_parts,
            ignore_index=True
        ),
        "14_long_trip_top_dates.csv"
    )

    if RUN_EXACT_DUPLICATE_CHECK:

        save_csv(
            pd.concat(
                duplicate_summaries,
                ignore_index=True
            ),
            "15_duplicate_summary.csv"
        )

        if duplicate_samples:
            dup_samples = pd.concat(
                duplicate_samples,
                ignore_index=True
            )
        else:
            dup_samples = pd.DataFrame()

        save_csv(
            dup_samples,
            "16_duplicate_samples.csv"
        )

    # ========================================================
    # Console summary
    # ========================================================

    print("\n" + "=" * 80)
    print("FOLLOW-UP QC COMPLETE")
    print("=" * 80)

    weighted_df = pd.DataFrame(weighted_results)

    print("\nWeighted speed:")
    print(
        weighted_df[
            [
                "file",
                "weighted_na",
                "weighted_na_volume0",
                "weighted_na_volume_positive",
                "weighted_present_volume0",
                "status",
            ]
        ].to_string(index=False)
    )

    print("\nVehicleType=0:")
    vehicle_df = pd.DataFrame(vehicle_results)
    print(
        vehicle_df[
            [
                "file",
                "vehicle_type_0",
                "type0_own_both_na",
                "expected_type_own_na",
                "status",
            ]
        ].to_string(index=False)
    )

    print("\nLong trips:")
    long_df = pd.concat(
        long_summaries,
        ignore_index=True
    )
    print(
        long_df[
            [
                "file",
                "gt500",
                "gt600",
                "gt700",
                "gt800",
                "gt900",
                "max_trip_length",
                "gt500_rate_pct",
            ]
        ].to_string(index=False)
    )

    if RUN_EXACT_DUPLICATE_CHECK:
        print("\nDuplicates:")
        dup_df = pd.concat(
            duplicate_summaries,
            ignore_index=True
        )
        print(
            dup_df.to_string(index=False)
        )

    print(
        "\n請把 09、10、11、13、15（若有）傳回來即可；"
        "12_long_trips_all.csv 若檔案不大也可以一起傳。"
    )


if __name__ == "__main__":
    main()
