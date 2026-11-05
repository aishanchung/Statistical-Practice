# -*- coding: utf-8 -*-
"""
build_v2_lite_n3_n.py

目的
----
只針對國道 3 號北向：
    V1_LITE + VD_N3_N_LITE -> V2_LITE

V1_LITE 原本 15 欄全部保留，再新增 5 支 VD 的：
    Speed + Occupancy
共新增 10 欄，最終約 25 欄。

重要設計
--------
1. 最終 V2_LITE 不保留 Time5 / MergeTime。
2. 最終 V2_LITE 不保留 VDType，避免欄位膨脹。
3. VDType 留在獨立的 VD_N3_N_LITE.parquet，供 QC / 資料追溯使用。
4. VD 負數狀態碼會原樣進入 V2_LITE：
      -1    = Occupancy 全 0，沒有偵測到車
      -99   = VD 存在，但官方速度無效 / 缺測
      -999  = XML 存在，但該 VD 不在 XML
      -9999 = 官方 XML 不存在 / 無法取得
5. 合併鍵使用 BottleneckTime 向上對齊到下一個 5 分鐘格點，
   與先前 Master Time5 的 ceil("5min") 邏輯一致。
6. VD_N3_N_LITE 中 RECOVERED_P1 (+1 分鐘找回) 也會被合併進來。
   後續若建立「即時預測」模型，需回到 VDType 排除未來資訊。

需求
----
pip install duckdb
"""

from pathlib import Path
import csv
import duckdb


# ============================================================
# 1. 路徑
# ============================================================

DATA_ROOT = Path(r"D:\統計實務\Data")

V1_LITE_PATH = (
    DATA_ROOT
    / "MASTER_V1_N3_LITE_2024_2025"
    / "N3_Daxi_Longtan_N_2024_2025_MASTER_V1_LITE.parquet"
)

VD_LITE_PATH = (
    DATA_ROOT
    / "VD"
    / "VD_N3_N_LITE_2024_2025"
    / "VD_N3_N_2024_2025_LITE.parquet"
)

OUT_ROOT = DATA_ROOT / "MASTER_V2_LITE_2024_2025"
OUT_PATH = OUT_ROOT / "N3_Daxi_Longtan_N_2024_2025_MASTER_V2_LITE.parquet"
QC_PATH = OUT_ROOT / "01_N3_N_V2_LITE_merge_qc.csv"

TEMP_ROOT = OUT_ROOT / "_duckdb_temp"

FORCE_REBUILD = True


# ============================================================
# 2. N3 北向 5 支 VD 與輸出欄位
# ============================================================

VD_MAP = [
    ("VD-N3-N-61.000-M-LOOP", "61p000"),
    ("VD-N3-N-61.670-M-RS",   "61p670"),
    ("VD-N3-N-63.990-M-LOOP", "63p990"),
    ("VD-N3-N-65.990-M-RS",   "65p990"),
    ("VD-N3-N-69.510-M-LOOP", "69p510"),
]


# ============================================================
# 3. Helpers
# ============================================================

def sql_path(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"找不到檔案：{path}")


# ============================================================
# 4. Main
# ============================================================

def main():
    print("=" * 100)
    print("BUILD N3 NORTH V2_LITE")
    print("=" * 100)

    require_file(V1_LITE_PATH)
    require_file(VD_LITE_PATH)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)

    if OUT_PATH.exists() and FORCE_REBUILD:
        OUT_PATH.unlink()

    con = duckdb.connect(database=":memory:")
    con.execute(f"SET temp_directory='{sql_path(TEMP_ROOT)}'")
    con.execute("SET preserve_insertion_order=false")

    # --------------------------------------------------------
    # V1_LITE schema
    # --------------------------------------------------------
    v1_cols = [
        row[0]
        for row in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{sql_path(V1_LITE_PATH)}')"
        ).fetchall()
    ]

    if "BottleneckTime" not in v1_cols:
        raise RuntimeError("V1_LITE 缺少 BottleneckTime，無法建立 5 分鐘合併鍵。")

    # --------------------------------------------------------
    # VD_LITE schema / key QC
    # --------------------------------------------------------
    vd_cols = [
        row[0]
        for row in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{sql_path(VD_LITE_PATH)}')"
        ).fetchall()
    ]

    required_vd_cols = {"TargetTime", "VDID", "Speed", "Occupancy", "VDType"}
    missing_vd_cols = required_vd_cols - set(vd_cols)
    if missing_vd_cols:
        raise RuntimeError(f"VD_LITE 缺少必要欄位：{sorted(missing_vd_cols)}")

    dup_count = con.execute(f"""
        SELECT COUNT(*)
        FROM (
            SELECT TargetTime, VDID, COUNT(*) AS n
            FROM read_parquet('{sql_path(VD_LITE_PATH)}')
            GROUP BY TargetTime, VDID
            HAVING COUNT(*) > 1
        )
    """).fetchone()[0]

    if dup_count != 0:
        raise RuntimeError(f"VD_LITE 有 {dup_count:,} 組重複 TargetTime x VDID。")

    # --------------------------------------------------------
    # 建立一列一個 TargetTime 的 VD wide view
    # 只帶 Speed / Occupancy，不帶 VDType。
    # --------------------------------------------------------
    wide_exprs = []
    vd_output_cols = []

    for vdid, alias in VD_MAP:
        speed_col = f"VD_{alias}_Speed"
        occ_col = f"VD_{alias}_Occupancy"
        vd_output_cols.extend([speed_col, occ_col])

        vdid_sql = vdid.replace("'", "''")
        wide_exprs.append(
            f"MAX(CASE WHEN VDID = '{vdid_sql}' THEN Speed END) AS {qident(speed_col)}"
        )
        wide_exprs.append(
            f"MAX(CASE WHEN VDID = '{vdid_sql}' THEN Occupancy END) AS {qident(occ_col)}"
        )

    con.execute(f"""
        CREATE TEMP VIEW vd_wide AS
        SELECT
            TargetTime,
            {', '.join(wide_exprs)}
        FROM read_parquet('{sql_path(VD_LITE_PATH)}')
        GROUP BY TargetTime
    """)

    # 每個時間理論上 5 支都有一列（即使是負數狀態碼）。
    bad_vd_times = con.execute(f"""
        SELECT COUNT(*)
        FROM (
            SELECT TargetTime, COUNT(*) AS n
            FROM read_parquet('{sql_path(VD_LITE_PATH)}')
            GROUP BY TargetTime
            HAVING COUNT(*) <> 5
        )
    """).fetchone()[0]

    if bad_vd_times != 0:
        raise RuntimeError(f"VD_LITE 有 {bad_vd_times:,} 個時間點不是完整 5 支 VD。")

    # --------------------------------------------------------
    # Row counts before merge
    # --------------------------------------------------------
    input_rows = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{sql_path(V1_LITE_PATH)}')"
    ).fetchone()[0]

    vd_times = con.execute("SELECT COUNT(*) FROM vd_wide").fetchone()[0]

    print(f"V1_LITE rows : {input_rows:,}")
    print(f"VD time rows : {vd_times:,}")
    print(f"V1 columns   : {len(v1_cols)}")
    print(f"VD add cols  : {len(vd_output_cols)}")

    # --------------------------------------------------------
    # V1 的 merge key：ceil 到 5 分鐘。
    # 最後不輸出 _MergeTime。
    # --------------------------------------------------------
    base_select = ",\n                ".join(
        f"v.{qident(c)}" for c in v1_cols
    )
    vd_select = ",\n                ".join(
        f"d.{qident(c)}" for c in vd_output_cols
    )

    merge_sql = f"""
        WITH v AS (
            SELECT
                *,
                CASE
                    WHEN BottleneckTime = time_bucket(INTERVAL '5 minutes', BottleneckTime)
                    THEN BottleneckTime
                    ELSE time_bucket(INTERVAL '5 minutes', BottleneckTime) + INTERVAL '5 minutes'
                END AS _MergeTime
            FROM read_parquet('{sql_path(V1_LITE_PATH)}')
        )
        SELECT
                {base_select},
                {vd_select}
        FROM v
        LEFT JOIN vd_wide d
            ON v._MergeTime = d.TargetTime
    """

    print("\nWriting V2_LITE Parquet...")

    con.execute(f"""
        COPY (
            {merge_sql}
        )
        TO '{sql_path(OUT_PATH)}'
        (
            FORMAT PARQUET,
            COMPRESSION ZSTD,
            ROW_GROUP_SIZE 250000
        )
    """)

    # --------------------------------------------------------
    # QC
    # --------------------------------------------------------
    output_rows = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{sql_path(OUT_PATH)}')"
    ).fetchone()[0]

    output_cols = [
        row[0]
        for row in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{sql_path(OUT_PATH)}')"
        ).fetchall()
    ]

    if output_rows != input_rows:
        raise RuntimeError(
            f"Merge row count changed: {input_rows:,} -> {output_rows:,}"
        )

    expected_cols = v1_cols + vd_output_cols
    if output_cols != expected_cols:
        raise RuntimeError(
            "V2_LITE 欄位不符合預期。\n"
            f"Expected: {expected_cols}\nActual: {output_cols}"
        )

    # 因 VD_LITE 本身建成完整時間格點，正常研究範圍內應可找到 VD row。
    # 若所有 VD 欄位都是 NULL，通常是 BottleneckTime ceil 後落在 VD 範圍之外，
    # 或 merge key 定義不一致。
    first_speed_col = vd_output_cols[0]
    unmatched_rows = con.execute(f"""
        SELECT COUNT(*)
        FROM read_parquet('{sql_path(OUT_PATH)}')
        WHERE {qident(first_speed_col)} IS NULL
    """).fetchone()[0]

    null_bottleneck_rows = con.execute(f"""
        SELECT COUNT(*)
        FROM read_parquet('{sql_path(V1_LITE_PATH)}')
        WHERE BottleneckTime IS NULL
    """).fetchone()[0]

    qc_rows = [
        ["N3_Daxi_Longtan_N", input_rows, output_rows, len(v1_cols), len(output_cols), vd_times, unmatched_rows, null_bottleneck_rows, "PASS" if input_rows == output_rows else "FAIL"]
    ]

    with open(QC_PATH, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "RouteDirection",
            "InputRows",
            "OutputRows",
            "InputColumns",
            "OutputColumns",
            "VDTargetTimes",
            "RowsWithoutVDTimeMatch",
            "RowsWithNullBottleneckTime",
            "RowCountStatus",
        ])
        writer.writerows(qc_rows)

    print("\nQC")
    print(f"Input rows            : {input_rows:,}")
    print(f"Output rows           : {output_rows:,}")
    print(f"Output columns        : {len(output_cols)}")
    print(f"Rows without VD match : {unmatched_rows:,}")
    print(f"Row count             : PASS")
    print(f"\nOutput: {OUT_PATH}")
    print(f"QC    : {QC_PATH}")

    con.close()


if __name__ == "__main__":
    main()
