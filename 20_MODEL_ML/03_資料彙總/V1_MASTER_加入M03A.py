# -*- coding: utf-8 -*-
"""
Prepare 2024-2025 M03A target-gantry 5-minute volumes and left-join them
into the four MASTER_V1 parquet files.

Input M03A layout example:
D:\\<project>\\Data\\M03A\\extracted\\2024\\12\\20241202\\M03A\\20241202\\09\\TDCS_M03A_20241202_095500.csv

Raw M03A columns (no header):
TimeInterval, GantryID, Direction, VehicleType, Volume

Outputs are written to NEW folders; MASTER_V1 is never overwritten.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import time

import duckdb
import pandas as pd


# ============================================================
# 1. Configuration
# ============================================================

# D:\\統計實務\\Data written with Unicode escapes so the source is portable.
DATA_ROOT = Path("D:/" + "".join(chr(x) for x in [0x7d71, 0x8a08, 0x5be6, 0x52d9]) + "/Data")

M03_RAW_ROOT = DATA_ROOT / "M03A" / "extracted"
M03_OUT_ROOT = DATA_ROOT / "M03A_TARGET_2024_2025"
MASTER_ROOT = DATA_ROOT / "MASTER_V1_2024_2025"
MASTER_OUT_ROOT = DATA_ROOT / "MASTER_V1_M03A_2024_2025"
QC_ROOT = DATA_ROOT / "M03A_MERGE_QC"
TEMP_ROOT = QC_ROOT / "_duckdb_temp"

START_DATE = "2024-01-01"
END_DATE = "2025-12-31"

RUN_PREPARE_M03 = True
RUN_MERGE_MASTER = True

FORCE_REBUILD_M03 = False
FORCE_REBUILD_MASTER = False


TARGETS = {
    "N3_Daxi_Longtan_N": {
        "gantry": "03F0648N",
        "direction": "N",
        "master_file": "N3_Daxi_Longtan_N_2024_2025_MASTER_V1.parquet",
    },
    "N3_Daxi_Longtan_S": {
        "gantry": "03F0648S",
        "direction": "S",
        "master_file": "N3_Daxi_Longtan_S_2024_2025_MASTER_V1.parquet",
    },
    "N5_Pinglin_Toucheng_N": {
        "gantry": "05F0287N",
        "direction": "N",
        "master_file": "N5_Pinglin_Toucheng_N_2024_2025_MASTER_V1.parquet",
    },
    "N5_Pinglin_Toucheng_S": {
        "gantry": "05F0287S",
        "direction": "S",
        "master_file": "N5_Pinglin_Toucheng_S_2024_2025_MASTER_V1.parquet",
    },
}

GANTRY_TO_KEY = {cfg["gantry"]: key for key, cfg in TARGETS.items()}
TARGET_GANTRIES = list(GANTRY_TO_KEY)


# ============================================================
# 2. Helpers
# ============================================================

def ensure_dirs():
    for p in [M03_OUT_ROOT, MASTER_OUT_ROOT, QC_ROOT, TEMP_ROOT]:
        p.mkdir(parents=True, exist_ok=True)


def make_connection():
    con = duckdb.connect(database=":memory:")
    temp = TEMP_ROOT.as_posix().replace("'", "''")
    con.execute(f"SET temp_directory='{temp}'")
    con.execute("SET preserve_insertion_order=false")
    return con


def sql_quote(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def sql_path(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def sql_file_list(files: list[Path]) -> str:
    values = [sql_quote(p.as_posix()) for p in files]
    return "[" + ",".join(values) + "]"


def iter_dates(start: str, end: str):
    d0 = datetime.strptime(start, "%Y-%m-%d").date()
    d1 = datetime.strptime(end, "%Y-%m-%d").date()
    d = d0
    while d <= d1:
        yield d
        d += timedelta(days=1)


def target_output_path(key: str) -> Path:
    return M03_OUT_ROOT / f"{key}_2024_2025_M03A.parquet"


def all_m03_outputs_exist() -> bool:
    return all(target_output_path(key).exists() for key in TARGETS)


# ============================================================
# 3. Locate one day of extracted M03A files
# ============================================================

def find_day_files(d) -> list[Path]:
    ymd = d.strftime("%Y%m%d")
    day_root = M03_RAW_ROOT / f"{d.year:04d}" / f"{d.month:02d}" / ymd

    if not day_root.exists():
        return []

    return sorted(day_root.rglob(f"TDCS_M03A_{ymd}_*.csv"))


# ============================================================
# 4. Read and pivot one day
# ============================================================

def read_one_day(con, d, files):
    expected_intervals = 288 * len(TARGET_GANTRIES)

    if not files:
        return pd.DataFrame(), {
            "Date": d.isoformat(),
            "SourceFiles": 0,
            "ExpectedFiles": 288,
            "SourceFileStatus": "MISSING_DAY",
            "RawTargetRows": 0,
            "InvalidRows": 0,
            "DuplicateKeyExtraRows": 0,
            "NegativeVolumeRows": 0,
            "TargetIntervals": 0,
            "ExpectedTargetIntervals": expected_intervals,
            "MissingTargetIntervals": expected_intervals,
        }

    file_list_sql = sql_file_list(files)
    gantry_sql = ",".join(sql_quote(x) for x in TARGET_GANTRIES)

    con.execute("DROP VIEW IF EXISTS m03raw")

    con.execute(f"""
        CREATE TEMP VIEW m03raw AS
        SELECT
            TRY_CAST(TimeInterval AS TIMESTAMP) AS M03_Time5,
            TRIM(GantryID) AS M03_GantryID,
            TRIM(Direction) AS M03_Direction,
            TRIM(VehicleType) AS VehicleType,
            TRY_CAST(Volume AS BIGINT) AS Volume
        FROM read_csv(
            {file_list_sql},
            header=false,
            auto_detect=false,
            columns={{
                'TimeInterval':'VARCHAR',
                'GantryID':'VARCHAR',
                'Direction':'VARCHAR',
                'VehicleType':'VARCHAR',
                'Volume':'VARCHAR'
            }}
        )
        WHERE TRIM(GantryID) IN ({gantry_sql})
    """)

    raw_target_rows = con.execute("SELECT COUNT(*) FROM m03raw").fetchone()[0]

    invalid_rows = con.execute("""
        SELECT COUNT(*)
        FROM m03raw
        WHERE M03_Time5 IS NULL
           OR M03_GantryID IS NULL
           OR M03_Direction IS NULL
           OR VehicleType IS NULL
           OR Volume IS NULL
    """).fetchone()[0]

    negative_rows = con.execute("""
        SELECT COUNT(*)
        FROM m03raw
        WHERE Volume < 0
    """).fetchone()[0]

    duplicate_extra = con.execute("""
        SELECT COALESCE(SUM(n - 1), 0)
        FROM (
            SELECT M03_Time5, M03_GantryID, VehicleType, COUNT(*) AS n
            FROM m03raw
            WHERE M03_Time5 IS NOT NULL
            GROUP BY 1,2,3
            HAVING COUNT(*) > 1
        ) x
    """).fetchone()[0]

    day_df = con.execute("""
        SELECT
            M03_Time5,
            M03_GantryID,
            ANY_VALUE(M03_Direction) AS M03_Direction,
            MAX(CASE WHEN VehicleType='31' THEN Volume END) AS M03_Volume_31,
            MAX(CASE WHEN VehicleType='32' THEN Volume END) AS M03_Volume_32,
            MAX(CASE WHEN VehicleType='41' THEN Volume END) AS M03_Volume_41,
            MAX(CASE WHEN VehicleType='42' THEN Volume END) AS M03_Volume_42,
            MAX(CASE WHEN VehicleType='5'  THEN Volume END) AS M03_Volume_5,
            COUNT(DISTINCT VehicleType) AS M03_ClassCount
        FROM m03raw
        WHERE M03_Time5 IS NOT NULL
          AND Volume IS NOT NULL
        GROUP BY M03_Time5, M03_GantryID
        ORDER BY M03_Time5, M03_GantryID
    """).fetchdf()

    if not day_df.empty:
        volume_cols = [
            "M03_Volume_31",
            "M03_Volume_32",
            "M03_Volume_41",
            "M03_Volume_42",
            "M03_Volume_5",
        ]

        # Only compute Total if all five class values are present.
        day_df["M03_Volume_Total"] = day_df[volume_cols].sum(axis=1, min_count=5)
        day_df["RouteDirection"] = day_df["M03_GantryID"].map(GANTRY_TO_KEY)
        day_df["SourceDate"] = d.isoformat()

        day_df = day_df[[
            "RouteDirection",
            "M03_Time5",
            "M03_GantryID",
            "M03_Direction",
            "M03_Volume_31",
            "M03_Volume_32",
            "M03_Volume_41",
            "M03_Volume_42",
            "M03_Volume_5",
            "M03_Volume_Total",
            "M03_ClassCount",
            "SourceDate",
        ]]

    n_intervals = len(day_df)

    qc = {
        "Date": d.isoformat(),
        "SourceFiles": len(files),
        "ExpectedFiles": 288,
        "SourceFileStatus": "COMPLETE" if len(files) == 288 else "INCOMPLETE",
        "RawTargetRows": int(raw_target_rows),
        "InvalidRows": int(invalid_rows),
        "DuplicateKeyExtraRows": int(duplicate_extra),
        "NegativeVolumeRows": int(negative_rows),
        "TargetIntervals": int(n_intervals),
        "ExpectedTargetIntervals": int(expected_intervals),
        "MissingTargetIntervals": int(max(expected_intervals - n_intervals, 0)),
    }

    return day_df, qc


# ============================================================
# 5. Prepare all 2024-2025 M03A target data
# ============================================================

def prepare_m03a():
    if not FORCE_REBUILD_M03 and all_m03_outputs_exist():
        print("[M03A] Prepared target files already exist. Skip preprocessing.")
        return

    print("=" * 90)
    print("STEP A - PREPARE M03A")
    print("Raw:", M03_RAW_ROOT)
    print("Out:", M03_OUT_ROOT)

    con = make_connection()
    all_parts = []
    qc_rows = []

    dates = list(iter_dates(START_DATE, END_DATE))

    for i, d in enumerate(dates, 1):
        files = find_day_files(d)
        day_df, qc = read_one_day(con, d, files)
        qc_rows.append(qc)

        if not day_df.empty:
            all_parts.append(day_df)

        if (
            i == 1
            or i % 10 == 0
            or qc["SourceFileStatus"] != "COMPLETE"
            or qc["InvalidRows"] > 0
            or qc["DuplicateKeyExtraRows"] > 0
            or i == len(dates)
        ):
            print(
                f"[{i:3d}/{len(dates)}] {d} | "
                f"files={qc['SourceFiles']:3d} | "
                f"intervals={qc['TargetIntervals']:4d} | "
                f"dup={qc['DuplicateKeyExtraRows']} | "
                f"invalid={qc['InvalidRows']}"
            )

    con.close()

    if not all_parts:
        raise RuntimeError("No target M03A data were found. Check M03_RAW_ROOT.")

    all_df = pd.concat(all_parts, ignore_index=True)

    dup = all_df.duplicated(
        subset=["RouteDirection", "M03_Time5"],
        keep=False,
    )

    if dup.any():
        dup_path = QC_ROOT / "m03_duplicate_time5_rows.csv"
        all_df.loc[dup].to_csv(dup_path, index=False, encoding="utf-8-sig")
        raise RuntimeError(f"Duplicate RouteDirection x Time5 found. See {dup_path}")

    combined_path = M03_OUT_ROOT / "M03A_4TARGETS_2024_2025.parquet"
    all_df.to_parquet(combined_path, index=False, compression="zstd")

    summary_rows = []

    expected = pd.date_range(
        START_DATE,
        f"{END_DATE} 23:55:00",
        freq="5min",
    )

    for key, cfg in TARGETS.items():
        z = all_df.loc[all_df["RouteDirection"] == key].copy()
        z = z.sort_values("M03_Time5")

        out = target_output_path(key)
        z.to_parquet(out, index=False, compression="zstd")

        actual = pd.DatetimeIndex(pd.to_datetime(z["M03_Time5"]))
        missing_n = len(expected.difference(actual))

        summary_rows.append({
            "RouteDirection": key,
            "Gantry": cfg["gantry"],
            "Direction": cfg["direction"],
            "Rows": len(z),
            "ExpectedTime5": len(expected),
            "MissingTime5": int(missing_n),
            "MinTime5": z["M03_Time5"].min() if len(z) else None,
            "MaxTime5": z["M03_Time5"].max() if len(z) else None,
            "IncompleteClassRows": int((z["M03_ClassCount"] != 5).sum()),
            "NegativeTotalRows": int((z["M03_Volume_Total"] < 0).sum()),
            "OutputFile": str(out),
        })

        print(
            f"  {key}: rows={len(z):,}, "
            f"missing Time5={missing_n:,}, "
            f"incomplete class rows={summary_rows[-1]['IncompleteClassRows']:,}"
        )

    pd.DataFrame(qc_rows).to_csv(
        QC_ROOT / "01_m03_daily_source_qc.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(summary_rows).to_csv(
        QC_ROOT / "02_m03_target_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("Saved:", combined_path)


# ============================================================
# 6. Merge one target M03A file into one MASTER_V1 file
# ============================================================

def merge_one_master(key: str, cfg: dict) -> dict:
    master_in = MASTER_ROOT / cfg["master_file"]
    m03_in = target_output_path(key)

    master_out = MASTER_OUT_ROOT / cfg["master_file"].replace(
        "_MASTER_V1.parquet",
        "_MASTER_V1_M03A.parquet",
    )

    if not master_in.exists():
        raise FileNotFoundError(master_in)

    if not m03_in.exists():
        raise FileNotFoundError(m03_in)

    if master_out.exists() and not FORCE_REBUILD_MASTER:
        print(f"[{key}] Output exists; skip: {master_out}")
        return {
            "RouteDirection": key,
            "Status": "SKIP_EXISTING",
            "InputRows": None,
            "OutputRows": None,
            "MatchedRows": None,
            "UnmatchedRows": None,
            "MatchedPct": None,
            "M03VolumeMismatchRows": None,
            "OutputFile": str(master_out),
        }

    if master_out.exists():
        master_out.unlink()

    print("-" * 90)
    print("MERGE:", key)
    print("Master:", master_in)
    print("M03A  :", m03_in)
    print("Out   :", master_out)

    con = make_connection()

    mi = sql_path(master_in)
    vi = sql_path(m03_in)
    mo = sql_path(master_out)

    master_cols = [
        row[0]
        for row in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{mi}')"
        ).fetchall()
    ]

    if "Time5" not in master_cols:
        con.close()
        raise RuntimeError(f"{master_in.name} has no Time5 column.")

    existing_m03 = [c for c in master_cols if c.startswith("M03_")]
    if existing_m03:
        con.close()
        raise RuntimeError(
            f"{master_in.name} already has M03 columns: {existing_m03}"
        )

    input_rows = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{mi}')"
    ).fetchone()[0]

    m03_dup = con.execute(f"""
        SELECT COUNT(*)
        FROM (
            SELECT M03_Time5
            FROM read_parquet('{vi}')
            GROUP BY M03_Time5
            HAVING COUNT(*) > 1
        ) x
    """).fetchone()[0]

    if m03_dup > 0:
        con.close()
        raise RuntimeError(f"{key}: M03_Time5 has {m03_dup} duplicate keys.")

    # Master.Time5 is already the 5-minute TDCS key used by M05A.
    # M03A uses the same 5-minute report timestamp, so join directly on Time5.
    copy_sql = f"""
        COPY (
            SELECT
                m.*,
                v.M03_Time5,
                v.M03_GantryID,
                v.M03_Direction,
                v.M03_Volume_31,
                v.M03_Volume_32,
                v.M03_Volume_41,
                v.M03_Volume_42,
                v.M03_Volume_5,
                v.M03_Volume_Total,
                CASE CAST(m.VehicleType AS VARCHAR)
                    WHEN '31' THEN v.M03_Volume_31
                    WHEN '32' THEN v.M03_Volume_32
                    WHEN '41' THEN v.M03_Volume_41
                    WHEN '42' THEN v.M03_Volume_42
                    WHEN '5'  THEN v.M03_Volume_5
                    ELSE NULL
                END AS M03_Volume_OwnType,
                (v.M03_Time5 IS NOT NULL) AS M03_Matched
            FROM read_parquet('{mi}') AS m
            LEFT JOIN read_parquet('{vi}') AS v
                ON m.Time5 = v.M03_Time5
        )
        TO '{mo}'
        (
            FORMAT PARQUET,
            COMPRESSION ZSTD,
            ROW_GROUP_SIZE 250000
        )
    """

    t0 = time.time()
    con.execute(copy_sql)
    elapsed = time.time() - t0

    output_rows = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{mo}')"
    ).fetchone()[0]

    matched_rows, unmatched_rows = con.execute(f"""
        SELECT
            COUNT(*) FILTER (WHERE M03_Matched = TRUE),
            COUNT(*) FILTER (WHERE M03_Matched = FALSE)
        FROM read_parquet('{mo}')
    """).fetchone()

    volume_mismatch = con.execute(f"""
        SELECT COUNT(*)
        FROM read_parquet('{mo}')
        WHERE M03_Matched = TRUE
          AND M03_Volume_Total IS DISTINCT FROM
              (M03_Volume_31 + M03_Volume_32 + M03_Volume_41
               + M03_Volume_42 + M03_Volume_5)
    """).fetchone()[0]

    con.close()

    if input_rows != output_rows:
        raise RuntimeError(
            f"{key}: row count changed after join: {input_rows} -> {output_rows}"
        )

    match_pct = 100.0 * matched_rows / output_rows if output_rows else 0.0

    print(f"  Input rows : {input_rows:,}")
    print(f"  Output rows: {output_rows:,}")
    print(f"  M03 matched: {matched_rows:,} ({match_pct:.5f}%)")
    print(f"  Unmatched  : {unmatched_rows:,}")
    print(f"  Volume mismatch: {volume_mismatch:,}")
    print(f"  Time: {elapsed / 60:.1f} min")

    return {
        "RouteDirection": key,
        "Status": "PASS",
        "InputRows": int(input_rows),
        "OutputRows": int(output_rows),
        "MatchedRows": int(matched_rows),
        "UnmatchedRows": int(unmatched_rows),
        "MatchedPct": match_pct,
        "M03VolumeMismatchRows": int(volume_mismatch),
        "ElapsedMinutes": elapsed / 60.0,
        "OutputFile": str(master_out),
    }


# ============================================================
# 7. Merge all four MASTER files
# ============================================================

def merge_masters():
    print("=" * 90)
    print("STEP B - MERGE M03A INTO MASTER_V1")
    print("Master input :", MASTER_ROOT)
    print("Master output:", MASTER_OUT_ROOT)

    results = []

    for key, cfg in TARGETS.items():
        result = merge_one_master(key, cfg)
        results.append(result)

        pd.DataFrame(results).to_csv(
            QC_ROOT / "03_master_m03_merge_qc.csv",
            index=False,
            encoding="utf-8-sig",
        )


# ============================================================
# 8. Main
# ============================================================

def main():
    ensure_dirs()

    print("=" * 90)
    print("M03A -> MASTER_V1")
    print("M03 raw      :", M03_RAW_ROOT)
    print("M03 prepared :", M03_OUT_ROOT)
    print("Master input :", MASTER_ROOT)
    print("Master output:", MASTER_OUT_ROOT)
    print("QC           :", QC_ROOT)
    print("=" * 90)

    if RUN_PREPARE_M03:
        prepare_m03a()

    if RUN_MERGE_MASTER:
        merge_masters()

    print("=" * 90)
    print("COMPLETE")
    print("Please send these QC files back for review:")
    print("1. 01_m03_daily_source_qc.csv")
    print("2. 02_m03_target_summary.csv")
    print("3. 03_master_m03_merge_qc.csv")
    print("=" * 90)


if __name__ == "__main__":
    main()
