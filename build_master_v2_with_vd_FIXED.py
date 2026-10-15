# -*- coding: utf-8 -*-
"""
build_master_v2_with_vd_FIXED.py
=================================

用途
----
將組員整理前的 VD lane-level CSV 重新整理成：

1) VD_LONG_CLEAN_2024_2025.parquet
   一列 = RouteDirection x Time5 x VDID

2) 各方向 VD_5MIN_WIDE parquet
   一列 = RouteDirection x Time5
   同一時間的多支 VD 攤成不同 columns

3) MASTER_V2_2024_2025
   以既有 MASTER_V1_M03A_2024_2025 為 base，
   將 VD wide table many-to-one left join 到每一筆 M06B trip。

重要定義
--------
- MASTER_V2 一列仍然 = 一筆 M06B trip。
- 同一個 Time5 的多筆 trip 會共享同一組 M03A / M05A / VD / Weather / Calendar context。
- VD 是「5-minute target grid 上的 1-minute snapshot」，不是 5 分鐘平均值。
- Target Time = File_Time（台灣本地時間）。
- DataCollectTime = 實際 VD observation time（台灣本地時間）。
- 只接受 abs(TargetTime - DataCollectTime) <= 4 分鐘的 snapshot 作為有效交通狀態。
- explicit Status 1/2/3 視為異常；Status 缺失保留為 unknown，不直接當成設備異常。
- Occupancy 全為 0 時，不把 speed=0 解讀成壅塞；SensorSpeedRaw 設為 NA。
- 研究用壅塞門檻 VD_IsCongested40 = (有效 speed < 40 km/h)。
- missing VD != non-congestion；無有效速度時壅塞狀態保留 NA。

執行前需求
----------
pip install duckdb pandas pyarrow

預設資料夾
----------
原始 VD：
    D:\\統計實務\\Data\\VD

目前已完成 M03A 的 Master：
    D:\\統計實務\\Data\\MASTER_V1_M03A_2024_2025

輸出：
    D:\\統計實務\\Data\\VD_5MIN_WIDE_2024_2025
    D:\\統計實務\\Data\\MASTER_V2_2024_2025
    D:\\統計實務\\Data\\MASTER_V2_QC
"""

from __future__ import annotations

from pathlib import Path
import time

import duckdb
import pandas as pd


# ============================================================
# 1. 路徑與執行設定
# ============================================================

DATA_ROOT = Path(r"D:\統計實務\Data")

# 組員的 lane-level VD CSV 根目錄。
# 程式會 recursive 搜尋所有 CSV，只接受符合 raw schema 的檔案。
VD_RAW_ROOT = DATA_ROOT / "VD"

# VD 整理後資料
VD_OUT_ROOT = DATA_ROOT / "VD_5MIN_WIDE_2024_2025"

# 目前已完成 M03A merge 的 Master
MASTER_BASE_ROOT = DATA_ROOT / "MASTER_V1_M03A_2024_2025"

# 最終 V2
MASTER_V2_ROOT = DATA_ROOT / "MASTER_V2_2024_2025"

# QC
QC_ROOT = DATA_ROOT / "MASTER_V2_QC"
TEMP_ROOT = QC_ROOT / "_duckdb_temp"

START_TIME = "2024-01-01 00:00:00"
END_TIME = "2025-12-31 23:55:00"

# 組員設計：每 5 分鐘 target time 用附近 1-minute observation 代表。
# 最多接受前後 4 分鐘。
MAX_ABS_LAG_MIN = 4.0

# 只做 REVIEW flag，不自動刪除。
SPEED_REVIEW_MAX = 180.0

# 這次因為要覆蓋前一版 VDID mapping 錯誤的輸出，預設 True。
# 確認新版 QC PASS 後，可改回 False。
FORCE_REBUILD_VD = True
FORCE_REBUILD_MASTER_V2 = True


# ============================================================
# 2. VD Sensor Map
#    注意：VDID 必須使用 raw data 的完整名稱（含 M-LOOP / M-RS）
# ============================================================

SENSORS = [
    # ---------------- N3 North ----------------
    {
        "RouteDirection": "N3_Daxi_Longtan_N",
        "VDID": "VD-N3-N-61.000-M-LOOP",
        "Alias": "VD_N3_N_61p000",
        "MileKM": 61.000,
        "ETagMileKM": 64.8,
    },
    {
        "RouteDirection": "N3_Daxi_Longtan_N",
        "VDID": "VD-N3-N-61.670-M-RS",
        "Alias": "VD_N3_N_61p670",
        "MileKM": 61.670,
        "ETagMileKM": 64.8,
    },
    {
        "RouteDirection": "N3_Daxi_Longtan_N",
        "VDID": "VD-N3-N-63.990-M-LOOP",
        "Alias": "VD_N3_N_63p990",
        "MileKM": 63.990,
        "ETagMileKM": 64.8,
    },
    {
        "RouteDirection": "N3_Daxi_Longtan_N",
        "VDID": "VD-N3-N-65.990-M-RS",
        "Alias": "VD_N3_N_65p990",
        "MileKM": 65.990,
        "ETagMileKM": 64.8,
    },
    {
        "RouteDirection": "N3_Daxi_Longtan_N",
        "VDID": "VD-N3-N-69.510-M-LOOP",
        "Alias": "VD_N3_N_69p510",
        "MileKM": 69.510,
        "ETagMileKM": 64.8,
    },

    # ---------------- N3 South ----------------
    {
        "RouteDirection": "N3_Daxi_Longtan_S",
        "VDID": "VD-N3-S-61.400-M-LOOP",
        "Alias": "VD_N3_S_61p400",
        "MileKM": 61.400,
        "ETagMileKM": 64.8,
    },
    {
        "RouteDirection": "N3_Daxi_Longtan_S",
        "VDID": "VD-N3-S-62.008-M-LOOP",
        "Alias": "VD_N3_S_62p008",
        "MileKM": 62.008,
        "ETagMileKM": 64.8,
    },
    {
        "RouteDirection": "N3_Daxi_Longtan_S",
        "VDID": "VD-N3-S-64.500-M-RS",
        "Alias": "VD_N3_S_64p500",
        "MileKM": 64.500,
        "ETagMileKM": 64.8,
    },
    {
        "RouteDirection": "N3_Daxi_Longtan_S",
        "VDID": "VD-N3-S-66.020-M-LOOP",
        "Alias": "VD_N3_S_66p020",
        "MileKM": 66.020,
        "ETagMileKM": 64.8,
    },
    {
        "RouteDirection": "N3_Daxi_Longtan_S",
        "VDID": "VD-N3-S-67.000-M-LOOP",
        "Alias": "VD_N3_S_67p000",
        "MileKM": 67.000,
        "ETagMileKM": 64.8,
    },
    {
        "RouteDirection": "N3_Daxi_Longtan_S",
        "VDID": "VD-N3-S-69.700-M-RS",
        "Alias": "VD_N3_S_69p700",
        "MileKM": 69.700,
        "ETagMileKM": 64.8,
    },

    # ---------------- N5 North ----------------
    {
        "RouteDirection": "N5_Pinglin_Toucheng_N",
        "VDID": "VD-N5-N-28.420-M-LOOP",
        "Alias": "VD_N5_N_28p420",
        "MileKM": 28.420,
        "ETagMileKM": 28.7,
    },
    {
        "RouteDirection": "N5_Pinglin_Toucheng_N",
        "VDID": "VD-N5-N-29.000-M-LOOP",
        "Alias": "VD_N5_N_29p000",
        "MileKM": 29.000,
        "ETagMileKM": 28.7,
    },

    # ---------------- N5 South ----------------
    {
        "RouteDirection": "N5_Pinglin_Toucheng_S",
        "VDID": "VD-N5-S-28.236-M-LOOP",
        "Alias": "VD_N5_S_28p236",
        "MileKM": 28.236,
        "ETagMileKM": 28.7,
    },
]


# ============================================================
# 3. Master 檔名
# ============================================================

MASTER_FILES = {
    "N3_Daxi_Longtan_N": "N3_Daxi_Longtan_N_2024_2025_MASTER_V1_M03A.parquet",
    "N3_Daxi_Longtan_S": "N3_Daxi_Longtan_S_2024_2025_MASTER_V1_M03A.parquet",
    "N5_Pinglin_Toucheng_N": "N5_Pinglin_Toucheng_N_2024_2025_MASTER_V1_M03A.parquet",
    "N5_Pinglin_Toucheng_S": "N5_Pinglin_Toucheng_S_2024_2025_MASTER_V1_M03A.parquet",
}


# ============================================================
# 4. Raw VD schema
# ============================================================

REQUIRED_COLUMNS = {
    "File_Time",
    "DataCollectTime",
    "VDID",
    "Status",
    "LaneID",
    "Speed_kmh",
    "Occupancy",
}


# ============================================================
# 5. Helpers
# ============================================================

def ensure_dirs():
    for p in [VD_OUT_ROOT, MASTER_V2_ROOT, QC_ROOT, TEMP_ROOT]:
        p.mkdir(parents=True, exist_ok=True)


def sql_path(path: Path) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


def sql_quote(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def file_list_sql(files: list[Path]) -> str:
    return "[" + ",".join(sql_quote(sql_path(f)) for f in files) + "]"


def make_connection() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(database=":memory:")
    con.execute(f"SET temp_directory='{sql_path(TEMP_ROOT)}'")
    con.execute("SET preserve_insertion_order=false")
    return con


def make_sensor_map() -> pd.DataFrame:
    df = pd.DataFrame(SENSORS).copy()

    def role(row):
        route = row["RouteDirection"]
        mile = row["MileKM"]
        target = row["ETagMileKM"]

        # 南向通常沿里程增加：target 前方（較小里程）為 upstream。
        if route.endswith("_S"):
            return "UPSTREAM" if mile < target else "DOWNSTREAM"

        # 北向通常沿里程減少：target 前方（較大里程）為 upstream。
        return "UPSTREAM" if mile > target else "DOWNSTREAM"

    df["Role"] = df.apply(role, axis=1)
    df["DistanceToETagKM"] = (df["MileKM"] - df["ETagMileKM"]).abs()

    # 靜態最近 sensor：僅供 metadata / 檢查。
    df["StaticClosest"] = False
    for route_key, idx in df.groupby("RouteDirection")["DistanceToETagKM"].idxmin().items():
        df.loc[idx, "StaticClosest"] = True

    out = QC_ROOT / "00_vd_sensor_map.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return df


# ============================================================
# 6. 掃描 raw VD CSV
# ============================================================

def find_raw_vd_files() -> list[Path]:
    if not VD_RAW_ROOT.exists():
        raise FileNotFoundError(f"VD_RAW_ROOT 不存在：{VD_RAW_ROOT}")

    all_csv = sorted(VD_RAW_ROOT.rglob("*.csv"))
    accepted = []
    rejected = []

    print("\n[1] 掃描 VD CSV...")

    for f in all_csv:
        try:
            cols = set(pd.read_csv(f, nrows=0).columns)
            if REQUIRED_COLUMNS.issubset(cols):
                accepted.append(f)
            else:
                rejected.append(f)
        except Exception:
            rejected.append(f)

    print(f"  accepted raw VD CSV : {len(accepted):,}")
    print(f"  ignored other CSV   : {len(rejected):,}")

    pd.DataFrame({"File": [str(x) for x in accepted]}).to_csv(
        QC_ROOT / "00a_vd_raw_files_used.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if not accepted:
        raise RuntimeError(
            "找不到符合 raw VD schema 的 CSV。\n"
            f"請確認：{VD_RAW_ROOT}\n"
            f"必要欄位：{sorted(REQUIRED_COLUMNS)}"
        )

    return accepted


# ============================================================
# 7. Preflight：避免再發生 VDID mapping 全部錯掉但程式仍跑完
# ============================================================

def preflight_check_vdids(files: list[Path], sensor_map: pd.DataFrame):
    print("\n[2] VDID preflight check...")

    con = make_connection()
    fs = file_list_sql(files)

    raw_ids = con.execute(
        f"""
        SELECT DISTINCT TRIM(CAST(VDID AS VARCHAR)) AS VDID
        FROM read_csv_auto(
            {fs},
            header=true,
            union_by_name=true,
            all_varchar=true
        )
        WHERE VDID IS NOT NULL
        ORDER BY VDID
        """
    ).fetchdf()

    con.close()

    raw_set = set(raw_ids["VDID"].dropna().astype(str))
    expected_set = set(sensor_map["VDID"].dropna().astype(str))

    missing_expected = sorted(expected_set - raw_set)
    extra_raw = sorted(raw_set - expected_set)

    pd.DataFrame({"VDID": sorted(raw_set)}).to_csv(
        QC_ROOT / "00b_raw_unique_vdids.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(f"  raw unique VDID : {len(raw_set):,}")
    print(f"  expected VDID   : {len(expected_set):,}")

    if extra_raw:
        print(f"  raw 中另有 {len(extra_raw):,} 支不在研究 sensor map，將忽略。")

    if missing_expected:
        print("\n  ERROR：以下研究 VDID 在 raw data 中完全找不到：")
        for x in missing_expected:
            print("   -", x)
        raise RuntimeError("Sensor map 與 raw VDID 不一致，停止執行。")

    print("  VDID check: PASS")


# ============================================================
# 8. 建立 clean long：一列 = Time5 x VDID
# ============================================================

def build_vd_long(files: list[Path], sensor_map: pd.DataFrame) -> Path:
    out = VD_OUT_ROOT / "VD_LONG_CLEAN_2024_2025.parquet"

    if out.exists() and not FORCE_REBUILD_VD:
        print("\n[3] VD long 已存在，跳過：", out)
        return out

    if out.exists():
        out.unlink()

    print("\n[3] 建立 VD_LONG_CLEAN...")

    con = make_connection()
    con.register("sensor_map_df", sensor_map)

    fs = file_list_sql(files)
    valid_vdids = ",".join(sql_quote(x) for x in sensor_map["VDID"])

    sql = f"""
    COPY (
        WITH raw AS (
            SELECT
                TRY_CAST(File_Time AS TIMESTAMP) AS Time5,

                -- DataCollectTime 原始值可能是 ISO +08:00。
                -- 研究全部使用台灣 local naive time，因此取 YYYY-MM-DD HH:MM:SS。
                TRY_CAST(
                    SUBSTR(
                        REPLACE(CAST(DataCollectTime AS VARCHAR), 'T', ' '),
                        1,
                        19
                    )
                    AS TIMESTAMP
                ) AS DataCollectTimeLocal,

                TRIM(CAST(VDID AS VARCHAR)) AS VDID,
                TRY_CAST(NULLIF(TRIM(CAST(Status AS VARCHAR)), '') AS INTEGER) AS Status,
                TRIM(CAST(LaneID AS VARCHAR)) AS LaneID,
                TRY_CAST(Speed_kmh AS DOUBLE) AS LaneSpeed,
                TRY_CAST(Occupancy AS DOUBLE) AS LaneOccupancy

            FROM read_csv_auto(
                {fs},
                header=true,
                union_by_name=true,
                all_varchar=true
            )

            WHERE TRIM(CAST(VDID AS VARCHAR)) IN ({valid_vdids})
        ),

        lane_clean AS (
            SELECT
                *,
                CASE WHEN LaneSpeed >= 0 THEN LaneSpeed ELSE NULL END AS LaneSpeedClean,
                CASE WHEN LaneOccupancy >= 0 THEN LaneOccupancy ELSE NULL END AS LaneOccupancyClean
            FROM raw
            WHERE Time5 BETWEEN TIMESTAMP '{START_TIME}' AND TIMESTAMP '{END_TIME}'
        ),

        station AS (
            SELECT
                Time5,
                VDID,
                MAX(DataCollectTimeLocal) AS DataCollectTimeLocal,
                MAX(Status) AS Status,

                COUNT(DISTINCT LaneID) AS LaneCount,
                COUNT(*) FILTER (WHERE LaneSpeedClean IS NOT NULL) AS ValidSpeedLaneCount,

                AVG(LaneSpeedClean) AS MeanLaneSpeed,
                AVG(LaneOccupancyClean) AS MeanOccupancy,
                MAX(LaneOccupancyClean) AS MaxOccupancy,

                SUM(LaneOccupancyClean)
                    FILTER (
                        WHERE LaneSpeedClean IS NOT NULL
                          AND LaneOccupancyClean > 0
                    ) AS PositiveOccupancySum,

                -- 組員原始概念：occupancy-weighted lane speed。
                -- 如果 occupancy 全為 0，不把 lane speed=0 當成壅塞速度，改留 NA。
                CASE
                    WHEN COALESCE(
                        SUM(LaneOccupancyClean)
                            FILTER (
                                WHERE LaneSpeedClean IS NOT NULL
                                  AND LaneOccupancyClean > 0
                            ),
                        0
                    ) > 0
                    THEN
                        SUM(LaneSpeedClean * LaneOccupancyClean)
                            FILTER (
                                WHERE LaneSpeedClean IS NOT NULL
                                  AND LaneOccupancyClean > 0
                            )
                        /
                        SUM(LaneOccupancyClean)
                            FILTER (
                                WHERE LaneSpeedClean IS NOT NULL
                                  AND LaneOccupancyClean > 0
                            )
                    ELSE NULL
                END AS SensorSpeedRaw,

                BOOL_AND(COALESCE(LaneSpeedClean, 0) = 0) AS AllLaneSpeedZero,
                BOOL_AND(COALESCE(LaneOccupancyClean, 0) = 0) AS AllLaneOccupancyZero

            FROM lane_clean
            GROUP BY Time5, VDID
        ),

        calc AS (
            SELECT
                s.*,
                DATE_DIFF('second', DataCollectTimeLocal, Time5) / 60.0 AS LagMinutes,
                ABS(DATE_DIFF('second', DataCollectTimeLocal, Time5) / 60.0) AS AbsLagMinutes,

                CASE WHEN Status IN (1, 2, 3) THEN TRUE ELSE FALSE END AS StatusAbnormal,
                (Status IS NULL) AS StatusUnknown,

                (AllLaneSpeedZero AND AllLaneOccupancyZero) AS NoTrafficLike
            FROM station s
        )

        SELECT
            m.RouteDirection,
            m.VDID,
            m.Alias,
            m.MileKM,
            m.ETagMileKM,
            m.Role,
            m.DistanceToETagKM,
            m.StaticClosest,

            c.Time5,
            c.DataCollectTimeLocal,
            c.Status,
            c.StatusUnknown,
            c.StatusAbnormal,

            c.LaneCount,
            c.ValidSpeedLaneCount,
            c.MeanLaneSpeed,
            c.MeanOccupancy,
            c.MaxOccupancy,
            c.PositiveOccupancySum,
            c.SensorSpeedRaw,
            c.AllLaneSpeedZero,
            c.AllLaneOccupancyZero,
            c.NoTrafficLike,

            c.LagMinutes,
            c.AbsLagMinutes,

            TRUE AS Observed,

            CASE
                WHEN c.SensorSpeedRaw IS NOT NULL
                 AND c.AbsLagMinutes <= {MAX_ABS_LAG_MIN}
                 AND c.StatusAbnormal = FALSE
                THEN TRUE
                ELSE FALSE
            END AS ValidSnapshot,

            CASE
                WHEN c.SensorSpeedRaw IS NOT NULL
                 AND c.AbsLagMinutes <= {MAX_ABS_LAG_MIN}
                 AND c.StatusAbnormal = FALSE
                THEN c.SensorSpeedRaw
                ELSE NULL
            END AS Speed,

            CASE
                WHEN c.SensorSpeedRaw IS NOT NULL
                 AND c.AbsLagMinutes <= {MAX_ABS_LAG_MIN}
                 AND c.StatusAbnormal = FALSE
                THEN c.MeanOccupancy
                ELSE NULL
            END AS Occupancy,

            CASE
                WHEN c.SensorSpeedRaw IS NOT NULL
                 AND c.AbsLagMinutes <= {MAX_ABS_LAG_MIN}
                 AND c.StatusAbnormal = FALSE
                THEN (c.SensorSpeedRaw < 40)
                ELSE NULL
            END AS IsCongested40,

            (c.SensorSpeedRaw > {SPEED_REVIEW_MAX}) AS SpeedReviewGT180

        FROM calc c
        INNER JOIN sensor_map_df m
            ON c.VDID = m.VDID

        ORDER BY m.RouteDirection, c.Time5, m.MileKM
    )
    TO '{sql_path(out)}'
    (FORMAT PARQUET, COMPRESSION ZSTD)
    """

    t0 = time.time()
    con.execute(sql)
    elapsed = (time.time() - t0) / 60.0

    # Key uniqueness
    dup = con.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT RouteDirection, Time5, VDID, COUNT(*) AS n
            FROM read_parquet('{sql_path(out)}')
            GROUP BY 1,2,3
            HAVING COUNT(*) > 1
        ) x
        """
    ).fetchone()[0]

    if dup > 0:
        con.close()
        raise RuntimeError(f"VD long 出現 {dup} 個重複 RouteDirection x Time5 x VDID key。")

    # Coverage QC
    coverage = con.execute(
        f"""
        SELECT
            RouteDirection,
            VDID,
            Alias,
            MileKM,
            Role,
            DistanceToETagKM,
            StaticClosest,

            COUNT(*) AS ObservedRows,
            COUNT(*) FILTER (WHERE ValidSnapshot) AS ValidRows,
            100.0 * COUNT(*) FILTER (WHERE ValidSnapshot) / NULLIF(COUNT(*),0) AS ValidPct,

            COUNT(*) FILTER (WHERE AbsLagMinutes > {MAX_ABS_LAG_MIN}) AS StaleRows,
            COUNT(*) FILTER (WHERE StatusAbnormal) AS AbnormalStatusRows,
            COUNT(*) FILTER (WHERE StatusUnknown) AS UnknownStatusRows,
            COUNT(*) FILTER (WHERE NoTrafficLike) AS NoTrafficLikeRows,
            COUNT(*) FILTER (WHERE SpeedReviewGT180) AS SpeedGT180Rows,

            MIN(Time5) AS FirstTime,
            MAX(Time5) AS LastTime,
            MEDIAN(AbsLagMinutes) AS MedianAbsLagMin,
            QUANTILE_CONT(AbsLagMinutes, 0.95) AS P95AbsLagMin,
            MAX(AbsLagMinutes) AS MaxAbsLagMin

        FROM read_parquet('{sql_path(out)}')
        GROUP BY
            RouteDirection, VDID, Alias, MileKM, Role,
            DistanceToETagKM, StaticClosest
        ORDER BY RouteDirection, MileKM
        """
    ).fetchdf()

    coverage.to_csv(
        QC_ROOT / "01_vd_sensor_coverage.csv",
        index=False,
        encoding="utf-8-sig",
    )

    stale = con.execute(
        f"""
        SELECT *
        FROM read_parquet('{sql_path(out)}')
        WHERE AbsLagMinutes > {MAX_ABS_LAG_MIN}
        ORDER BY AbsLagMinutes DESC
        """
    ).fetchdf()
    stale.to_csv(QC_ROOT / "02_vd_stale_rows.csv", index=False, encoding="utf-8-sig")

    high = con.execute(
        f"""
        SELECT *
        FROM read_parquet('{sql_path(out)}')
        WHERE SpeedReviewGT180
        ORDER BY SensorSpeedRaw DESC
        """
    ).fetchdf()
    high.to_csv(QC_ROOT / "03_vd_speed_gt180.csv", index=False, encoding="utf-8-sig")

    print(f"  VD long 完成：{elapsed:.1f} min")
    print(f"  key duplicate：{dup}")
    print(f"  sensor coverage rows：{len(coverage):,}")
    print(f"  stale rows：{len(stale):,}")
    print(f"  speed > {SPEED_REVIEW_MAX:g} rows：{len(high):,}")

    con.close()
    return out


# ============================================================
# 9. 建立一個方向的 wide table
#    一列 = Time5；每支 VD 變成 columns
# ============================================================

def build_one_wide(route_key: str, sensor_map: pd.DataFrame, long_path: Path) -> Path:
    route_sensors = (
        sensor_map.loc[sensor_map["RouteDirection"] == route_key]
        .sort_values("MileKM")
        .copy()
    )

    out = VD_OUT_ROOT / f"{route_key}_2024_2025_VD_5MIN_WIDE.parquet"

    if out.exists() and not FORCE_REBUILD_VD:
        print("  wide 已存在，跳過：", out)
        return out

    if out.exists():
        out.unlink()

    print(f"\n[4] 建立 VD wide：{route_key}")

    # Sensor-specific columns
    sensor_exprs = []
    for _, row in route_sensors.iterrows():
        alias = row["Alias"]

        sensor_exprs.extend([
            f"MAX(CASE WHEN x.Alias={sql_quote(alias)} THEN x.DataCollectTimeLocal END) AS \"{alias}_DataTime\"",
            f"MAX(CASE WHEN x.Alias={sql_quote(alias)} THEN x.Speed END) AS \"{alias}_Speed\"",
            f"MAX(CASE WHEN x.Alias={sql_quote(alias)} THEN x.Occupancy END) AS \"{alias}_Occupancy\"",
            f"MAX(CASE WHEN x.Alias={sql_quote(alias)} THEN x.LagMinutes END) AS \"{alias}_LagMin\"",
            f"MAX(CASE WHEN x.Alias={sql_quote(alias)} THEN x.Status END) AS \"{alias}_Status\"",
            f"COALESCE(BOOL_OR(CASE WHEN x.Alias={sql_quote(alias)} THEN x.Observed ELSE FALSE END), FALSE) AS \"{alias}_Observed\"",
            f"COALESCE(BOOL_OR(CASE WHEN x.Alias={sql_quote(alias)} THEN x.ValidSnapshot ELSE FALSE END), FALSE) AS \"{alias}_Valid\"",
        ])

    sensor_sql = ",\n                ".join(sensor_exprs)

    con = make_connection()

    # 這裡 primary 是「該 Time5 有效資料中，離 eTag 最近的 VD」，因此可隨 availability 動態變動。
    sql = f"""
    COPY (
        WITH grid AS (
            SELECT UNNEST(
                GENERATE_SERIES(
                    TIMESTAMP '{START_TIME}',
                    TIMESTAMP '{END_TIME}',
                    INTERVAL 5 MINUTE
                )
            ) AS Time5
        ),

        x AS (
            SELECT *
            FROM read_parquet('{sql_path(long_path)}')
            WHERE RouteDirection = {sql_quote(route_key)}
        ),

        primary_ranked AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY Time5
                    ORDER BY
                        DistanceToETagKM ASC,
                        AbsLagMinutes ASC,
                        MileKM ASC,
                        VDID ASC
                ) AS rn
            FROM x
            WHERE ValidSnapshot
        ),

        primary_one AS (
            SELECT
                Time5,
                VDID AS VD_PrimarySensorID,
                Alias AS VD_PrimaryAlias,
                MileKM AS VD_PrimaryMileKM,
                Role AS VD_PrimaryRole,
                Speed AS VD_PrimarySpeed,
                Occupancy AS VD_PrimaryOccupancy,
                LagMinutes AS VD_PrimaryLagMin,
                DataCollectTimeLocal AS VD_PrimaryDataTime,
                TRUE AS VD_PrimaryValid
            FROM primary_ranked
            WHERE rn = 1
        ),

        agg AS (
            SELECT
                g.Time5,

                {sensor_sql},

                COUNT(x.VDID) AS VD_ObservedSensorCount,
                COUNT(x.VDID) FILTER (WHERE x.ValidSnapshot) AS VD_ValidSensorCount,

                MIN(x.Speed) AS VD_MinSpeed,
                MEDIAN(x.Speed) AS VD_MedianSpeed,
                MAX(x.Speed) AS VD_MaxSpeed,

                AVG(x.Occupancy) AS VD_MeanOccupancy,
                MAX(x.Occupancy) AS VD_MaxOccupancy,

                COUNT(x.VDID)
                    FILTER (WHERE x.ValidSnapshot AND x.Speed < 40)
                    AS VD_CongestedSensorCount,

                CASE
                    WHEN COUNT(x.VDID) FILTER (WHERE x.ValidSnapshot) = 0
                    THEN NULL
                    ELSE
                        COUNT(x.VDID) FILTER (WHERE x.ValidSnapshot AND x.Speed < 40) * 1.0
                        / COUNT(x.VDID) FILTER (WHERE x.ValidSnapshot)
                END AS VD_CongestedSensorShare,

                CASE
                    WHEN COUNT(x.VDID) FILTER (WHERE x.ValidSnapshot) = 0
                    THEN NULL
                    ELSE BOOL_OR(x.ValidSnapshot AND x.Speed < 40)
                END AS VD_AnyCongested40,

                CASE
                    WHEN MIN(x.Speed) IS NULL THEN NULL
                    WHEN MIN(x.Speed) >= 80 THEN 1
                    WHEN MIN(x.Speed) >= 60 THEN 2
                    WHEN MIN(x.Speed) >= 40 THEN 3
                    WHEN MIN(x.Speed) >= 20 THEN 4
                    ELSE 5
                END AS VD_WorstSpeedBand,

                COUNT(x.VDID) FILTER (WHERE x.AbsLagMinutes > {MAX_ABS_LAG_MIN})
                    AS VD_StaleSensorCount,

                COUNT(x.VDID) FILTER (WHERE x.StatusUnknown)
                    AS VD_UnknownStatusSensorCount,

                COUNT(x.VDID) FILTER (WHERE x.StatusAbnormal)
                    AS VD_AbnormalStatusSensorCount,

                COUNT(x.VDID) FILTER (WHERE x.NoTrafficLike)
                    AS VD_NoTrafficLikeSensorCount,

                MAX(x.AbsLagMinutes) AS VD_MaxAbsLagMin

            FROM grid g
            LEFT JOIN x
                ON g.Time5 = x.Time5
            GROUP BY g.Time5
        )

        SELECT
            {sql_quote(route_key)} AS RouteDirection,
            a.*,

            p.VD_PrimarySensorID,
            p.VD_PrimaryAlias,
            p.VD_PrimaryMileKM,
            p.VD_PrimaryRole,
            p.VD_PrimarySpeed,
            p.VD_PrimaryOccupancy,
            p.VD_PrimaryLagMin,
            p.VD_PrimaryDataTime,
            COALESCE(p.VD_PrimaryValid, FALSE) AS VD_PrimaryValid

        FROM agg a
        LEFT JOIN primary_one p
            ON a.Time5 = p.Time5
        ORDER BY a.Time5
    )
    TO '{sql_path(out)}'
    (FORMAT PARQUET, COMPRESSION ZSTD)
    """

    con.execute(sql)

    n_rows = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{sql_path(out)}')"
    ).fetchone()[0]

    dup = con.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT Time5, COUNT(*) AS n
            FROM read_parquet('{sql_path(out)}')
            GROUP BY Time5
            HAVING COUNT(*) > 1
        ) x
        """
    ).fetchone()[0]

    expected = 731 * 288
    print(f"  rows={n_rows:,} (expected {expected:,})")
    print(f"  duplicate Time5={dup}")

    if dup > 0:
        con.close()
        raise RuntimeError(f"{route_key}: VD wide 出現 duplicate Time5。")

    if n_rows != expected:
        print("  WARNING：wide grid row 數不是 210,528，請檢查研究期間。")

    con.close()
    return out


def build_all_wide(sensor_map: pd.DataFrame, long_path: Path) -> dict[str, Path]:
    out = {}
    for route_key in MASTER_FILES:
        out[route_key] = build_one_wide(route_key, sensor_map, long_path)
    return out


# ============================================================
# 10. Merge VD wide -> MASTER V2
# ============================================================

def merge_one_master(route_key: str, wide_path: Path) -> dict:
    master_in = MASTER_BASE_ROOT / MASTER_FILES[route_key]
    master_out = MASTER_V2_ROOT / f"{route_key}_2024_2025_MASTER_V2.parquet"

    if not master_in.exists():
        raise FileNotFoundError(f"找不到 Master base：{master_in}")

    if not wide_path.exists():
        raise FileNotFoundError(f"找不到 VD wide：{wide_path}")

    if master_out.exists() and not FORCE_REBUILD_MASTER_V2:
        print("\n[5] V2 已存在，跳過：", master_out)
        return {
            "RouteDirection": route_key,
            "Status": "SKIP_EXISTING",
            "OutputFile": str(master_out),
        }

    if master_out.exists():
        master_out.unlink()

    print(f"\n[5] Merge MASTER V2：{route_key}")

    con = make_connection()
    mi = sql_path(master_in)
    wi = sql_path(wide_path)
    mo = sql_path(master_out)

    # Master schema guard
    master_cols = [
        r[0]
        for r in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{mi}')"
        ).fetchall()
    ]

    if "Time5" not in master_cols:
        con.close()
        raise RuntimeError(f"{master_in.name} 沒有 Time5。")

    existing_vd = [c for c in master_cols if c.startswith("VD_")]
    if existing_vd:
        con.close()
        raise RuntimeError(
            f"Base Master 已經有 VD 欄位，不應重複 merge。\n前幾個：{existing_vd[:10]}"
        )

    # Wide key 必須唯一
    wide_dup = con.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT Time5, COUNT(*) AS n
            FROM read_parquet('{wi}')
            GROUP BY Time5
            HAVING COUNT(*) > 1
        ) x
        """
    ).fetchone()[0]

    if wide_dup > 0:
        con.close()
        raise RuntimeError(f"{route_key}: VD wide Time5 duplicate={wide_dup}")

    input_rows = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{mi}')"
    ).fetchone()[0]

    wide_cols = [
        r[0]
        for r in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{wi}')"
        ).fetchall()
    ]
    wide_cols = [c for c in wide_cols if c not in {"RouteDirection", "Time5"}]

    add_cols = ",\n                ".join(
        f'v."{c}" AS "{c}"' for c in wide_cols
    )

    sql = f"""
    COPY (
        SELECT
            m.*,
            {add_cols}
        FROM read_parquet('{mi}') m
        LEFT JOIN read_parquet('{wi}') v
            ON m.Time5 = v.Time5
    )
    TO '{mo}'
    (
        FORMAT PARQUET,
        COMPRESSION ZSTD,
        ROW_GROUP_SIZE 250000
    )
    """

    t0 = time.time()
    con.execute(sql)
    elapsed_min = (time.time() - t0) / 60.0

    output_rows = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{mo}')"
    ).fetchone()[0]

    if input_rows != output_rows:
        con.close()
        raise RuntimeError(
            f"{route_key}: merge 後 row count 改變：{input_rows:,} -> {output_rows:,}"
        )

    qc = con.execute(
        f"""
        SELECT
            COUNT(*) AS Rows,
            COUNT(*) FILTER (WHERE VD_ObservedSensorCount > 0) AS RowsWithVDObservation,
            COUNT(*) FILTER (WHERE VD_ValidSensorCount > 0) AS RowsWithValidVD,
            COUNT(*) FILTER (WHERE VD_PrimaryValid) AS RowsWithValidPrimaryVD,
            COUNT(*) FILTER (WHERE VD_AnyCongested40 = TRUE) AS RowsAnyVDCongested40,
            COUNT(*) FILTER (WHERE VD_AnyCongested40 = FALSE) AS RowsAnyVDNotCongested40,
            COUNT(*) FILTER (WHERE VD_AnyCongested40 IS NULL) AS RowsVDStateUnknown
        FROM read_parquet('{mo}')
        """
    ).fetchdf().iloc[0]

    result = {
        "RouteDirection": route_key,
        "Status": "PASS",
        "InputRows": int(input_rows),
        "OutputRows": int(output_rows),
        "RowCountStatus": "PASS",
        "RowsWithVDObservation": int(qc["RowsWithVDObservation"]),
        "RowsWithValidVD": int(qc["RowsWithValidVD"]),
        "RowsWithValidPrimaryVD": int(qc["RowsWithValidPrimaryVD"]),
        "RowsAnyVDCongested40": int(qc["RowsAnyVDCongested40"]),
        "RowsAnyVDNotCongested40": int(qc["RowsAnyVDNotCongested40"]),
        "RowsVDStateUnknown": int(qc["RowsVDStateUnknown"]),
        "VDObservationPct": 100.0 * qc["RowsWithVDObservation"] / input_rows if input_rows else 0.0,
        "ValidVDPct": 100.0 * qc["RowsWithValidVD"] / input_rows if input_rows else 0.0,
        "PrimaryValidPct": 100.0 * qc["RowsWithValidPrimaryVD"] / input_rows if input_rows else 0.0,
        "ElapsedMinutes": elapsed_min,
        "OutputFile": str(master_out),
    }

    print(f"  before={input_rows:,}")
    print(f"  after ={output_rows:,}")
    print("  row-count QC=PASS")
    print(f"  VD observation={result['VDObservationPct']:.3f}%")
    print(f"  any valid VD ={result['ValidVDPct']:.3f}%")
    print(f"  primary valid={result['PrimaryValidPct']:.3f}%")
    print(f"  time={elapsed_min:.1f} min")

    # 這個防呆很重要：不允許再出現前一版「全部 0% 但程式照樣完成」。
    if result["RowsWithVDObservation"] == 0:
        con.close()
        raise RuntimeError(
            f"{route_key}: merge 完後 VD observation 仍為 0。"
            " 這通常代表 VDID / Time5 / raw path 有問題，停止。"
        )

    con.close()
    return result


def merge_all_masters(wide_paths: dict[str, Path]) -> pd.DataFrame:
    results = []

    for route_key in MASTER_FILES:
        result = merge_one_master(route_key, wide_paths[route_key])
        results.append(result)

        pd.DataFrame(results).to_csv(
            QC_ROOT / "04_master_v2_merge_qc.csv",
            index=False,
            encoding="utf-8-sig",
        )

    return pd.DataFrame(results)


# ============================================================
# 11. 欄位定義與 README
# ============================================================

def write_metadata(sensor_map: pd.DataFrame):
    readme = f"""
MASTER V2 DATA DEFINITION
=========================

Observation unit
----------------
MASTER_V2 一列仍然代表一筆通過研究瓶頸的 M06B trip。

Composition
-----------
MASTER_V2 = MASTER_V1 + M03A + VD snapshot context

也就是主要包含：
- M06B：trip / VehicleID / OD / TripLength / route path
- M05A：相鄰 ETC 門架區間交通狀態
- M03A：研究瓶頸 ETC 門架 5-minute volume
- VD：瓶頸附近固定偵測器的局部 traffic state
- Weather
- Calendar / Holiday

VD time definition
------------------
VD 並非 5-minute average。

本研究使用 5-minute Target Time（File_Time）作共同分析格點，
並使用距離該 Target Time 最近、且 abs(lag) <= {MAX_ABS_LAG_MIN:g} 分鐘的
1-minute VD observation 作為該格點的 snapshot。

Target_Time      = Time5 / File_Time（台灣本地時間）
DataCollectTime  = 實際 VD observation time（台灣本地時間）
LagMin           = Target_Time - DataCollectTime

VD speed
--------
同一支 VD 可能有多車道。
SensorSpeedRaw 採 occupancy-weighted lane speed：

    sum(speed_lane * occupancy_lane) / sum(occupancy_lane)

僅使用 occupancy > 0 的 lane。
如果所有 lane occupancy 都為 0，Speed 留 NA，避免把 speed=0 誤判為壅塞。

VD validity
-----------
ValidSnapshot = TRUE 需要：
1. SensorSpeedRaw 可用
2. abs(LagMin) <= {MAX_ABS_LAG_MIN:g}
3. Status 不是 explicit abnormal code 1 / 2 / 3

Status 缺失會標記 StatusUnknown，但不直接視為設備故障。

Congestion operational definition
---------------------------------
IsCongested40 = TRUE if valid Speed < 40 km/h。
這是本研究 operational threshold，不宣稱為唯一官方壅塞定義。

如果沒有有效 speed：
    IsCongested40 = NA

因此：
    missing VD != non-congestion

Primary VD
----------
每個 Time5 的 VD_PrimarySensorID 為：
「當下 ValidSnapshot=TRUE 的 sensors 中，距離目標 eTag 門架最近者」。

因此 Primary VD 可因 sensor availability 而隨時間改變。
各支 VD 的 sensor-specific columns 仍全部保留。

Sensor-specific wide columns
----------------------------
例如：
    VD_N3_N_63p990_DataTime
    VD_N3_N_63p990_Speed
    VD_N3_N_63p990_Occupancy
    VD_N3_N_63p990_LagMin
    VD_N3_N_63p990_Status
    VD_N3_N_63p990_Observed
    VD_N3_N_63p990_Valid

Corridor summary
----------------
    VD_ObservedSensorCount
    VD_ValidSensorCount
    VD_MinSpeed
    VD_MedianSpeed
    VD_MaxSpeed
    VD_MeanOccupancy
    VD_MaxOccupancy
    VD_CongestedSensorCount
    VD_CongestedSensorShare
    VD_AnyCongested40
    VD_WorstSpeedBand
    VD_StaleSensorCount
    VD_UnknownStatusSensorCount
    VD_AbnormalStatusSensorCount
    VD_NoTrafficLikeSensorCount
    VD_MaxAbsLagMin

Repeated traffic context in trip Master
---------------------------------------
同一 Route / Direction / Time5 的多筆 M06B trip 會共享相同 VD / M03A / M05A context。
這是預期的資料設計，不是 duplicate error。

若未來做 congestion nowcasting / Next15 / Next30 / Next60，
會另外建立 MODEL_5MIN_V1：

    一列 = Route x Direction x Time5

而 MASTER_V2 繼續保留供 VehicleID / trip / OD 分析。
""".strip()

    (MASTER_V2_ROOT / "README_MASTER_V2.txt").write_text(readme, encoding="utf-8")

    dictionary = pd.DataFrame([
        ["Time5", "Master / VD", "5-minute analysis target time, Taiwan local time"],
        ["VD_*_DataTime", "VD", "Actual local DataCollectTime used for the sensor snapshot"],
        ["VD_*_Speed", "VD", "QC-approved occupancy-weighted lane speed (km/h)"],
        ["VD_*_Occupancy", "VD", "Mean lane occupancy for the selected sensor snapshot"],
        ["VD_*_LagMin", "VD", "Target Time minus actual DataCollectTime, minutes"],
        ["VD_*_Status", "VD", "Original detector status when available"],
        ["VD_*_Observed", "VD", "Whether a raw snapshot exists for this sensor/time"],
        ["VD_*_Valid", "VD", f"Usable snapshot: speed exists, abs(lag)<={MAX_ABS_LAG_MIN:g}, no explicit abnormal status"],
        ["VD_PrimarySensorID", "VD", "Nearest valid VD to target eTag gantry at that Time5"],
        ["VD_PrimarySpeed", "VD", "Speed of dynamic primary VD"],
        ["VD_PrimaryOccupancy", "VD", "Occupancy of dynamic primary VD"],
        ["VD_ObservedSensorCount", "VD", "Number of target sensors with any raw observation at Time5"],
        ["VD_ValidSensorCount", "VD", "Number of target sensors with a valid traffic-state snapshot"],
        ["VD_MinSpeed", "VD", "Minimum valid sensor speed across corridor sensors"],
        ["VD_MedianSpeed", "VD", "Median valid sensor speed across corridor sensors"],
        ["VD_MaxOccupancy", "VD", "Maximum valid sensor occupancy across corridor sensors"],
        ["VD_CongestedSensorCount", "VD", "Count of valid sensors with speed < 40 km/h"],
        ["VD_CongestedSensorShare", "VD", "Share of valid sensors with speed < 40 km/h"],
        ["VD_AnyCongested40", "VD", "TRUE if any valid sensor has speed < 40; NA if no valid speed exists"],
        ["VD_WorstSpeedBand", "VD", "Band based on minimum valid sensor speed: >=80, 60-80, 40-60, 20-40, <20"],
    ], columns=["Field", "Source", "Definition"])

    dictionary.to_csv(
        MASTER_V2_ROOT / "DATA_DICTIONARY_V2_ADDITIONS.csv",
        index=False,
        encoding="utf-8-sig",
    )

    sensor_map.to_csv(
        MASTER_V2_ROOT / "VD_SENSOR_MAP.csv",
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# 12. Main
# ============================================================

def main():
    ensure_dirs()

    print("=" * 90)
    print("BUILD MASTER V2 WITH VD - FIXED VERSION")
    print("=" * 90)
    print("VD raw       :", VD_RAW_ROOT)
    print("Master base  :", MASTER_BASE_ROOT)
    print("VD output    :", VD_OUT_ROOT)
    print("Master V2 out:", MASTER_V2_ROOT)
    print("QC           :", QC_ROOT)
    print("=" * 90)

    sensor_map = make_sensor_map()
    files = find_raw_vd_files()

    # 關鍵防呆：mapping 不對就立刻停，不再讓 0% match 默默跑到底。
    preflight_check_vdids(files, sensor_map)

    long_path = build_vd_long(files, sensor_map)
    wide_paths = build_all_wide(sensor_map, long_path)
    merge_qc = merge_all_masters(wide_paths)

    write_metadata(sensor_map)

    print("\n" + "=" * 90)
    print("COMPLETE")
    print("=" * 90)
    print("\n請回傳以下 QC 檔：")
    print("1. 00_vd_sensor_map.csv")
    print("2. 01_vd_sensor_coverage.csv")
    print("3. 02_vd_stale_rows.csv")
    print("4. 03_vd_speed_gt180.csv")
    print("5. 04_master_v2_merge_qc.csv")

    if not merge_qc.empty:
        print("\nMASTER V2 merge summary:")
        print(
            merge_qc[
                [
                    "RouteDirection",
                    "InputRows",
                    "OutputRows",
                    "VDObservationPct",
                    "ValidVDPct",
                    "PrimaryValidPct",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
