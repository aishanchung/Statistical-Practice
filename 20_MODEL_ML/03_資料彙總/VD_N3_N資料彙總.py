# -*- coding: utf-8 -*-
"""
build_vd_n3_n_lite.py

目的
----
以同學已整理好的 N3 北向 VD CSV 為主體，搭配 missing archive audit 結果，
建立一份乾淨、固定完整時間格點的 VD_N3_N_LITE.parquet。

正式資料一列 = 一個 TargetTime x VDID，只保留 5 欄：
    TargetTime, VDID, Speed, Occupancy, VDType

Speed / Occupancy 特殊碼
------------------------
-1    : 有 VD 觀測，但所有可用車道 Occupancy = 0（視為沒有偵測到車）
-99   : VD 存在，但官方速度資料無效 / 缺測
-999  : 官方 XML 存在，但該 VDID 不在 XML 中
-9999 : 官方歷史 XML 本身不存在或無法取得

VDType
------
OBSERVED         : 原本同學整理檔中已有的資料
RECOVERED_EXACT  : 原整理檔缺，但回查官方同分鐘找到有效資料
RECOVERED_M1     : 原整理檔缺，使用 TargetTime - 1 分鐘找到的有效資料
RECOVERED_P1     : 原整理檔缺，使用 TargetTime + 1 分鐘找到的有效資料
OFFICIAL_NA      : VD 存在，但官方速度無效
VD_ABSENT        : XML 存在，但該 VD 不在 XML
FILE_UNAVAILABLE : 官方 XML 不存在 / 無法取得

注意
----
1. TargetTime 是台灣本地時間的分析時間鍵，不把字尾 Z 當 UTC 轉換。
2. Speed 使用 Occupancy-weighted speed。
3. Occupancy 使用可用車道 Occupancy 的算術平均。
4. 本程式只處理國道 3 號北向 5 支 VD。
5. RECOVERED_P1 使用未來 1 分鐘觀測；做「即時預測」模型時應另外排除或處理。
"""

from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# 1. 路徑：請依你的電腦修改這三個輸入路徑
# ============================================================

from pathlib import Path

DATA_ROOT = Path(r"D:\統計實務\Data\VD")

SOURCE_VD_CSV = (
    DATA_ROOT
    / "N3_Daxi_Longtan_N_2024_2025_MASTER.csv"
)

AUDIT_CSV = (
    DATA_ROOT
    / "02_missing_archive_audit.csv"
)

RECOVERED_CSV = (
    DATA_ROOT
    / "03_recovered_lane_candidates.csv"
)

OUT_ROOT = DATA_ROOT / "VD_N3_N_LITE_2024_2025"
OUT_PARQUET = OUT_ROOT / "VD_N3_N_2024_2025_LITE.parquet"
OUT_SUMMARY = OUT_ROOT / "01_vdtype_summary.csv"
OUT_CODES = OUT_ROOT / "02_value_codebook.csv"


# ============================================================
# 2. 研究設定
# ============================================================

START_TIME = pd.Timestamp("2024-01-01 00:00:00")
END_TIME = pd.Timestamp("2025-12-31 23:55:00")

TARGET_VDS = [
    "VD-N3-N-61.000-M-LOOP",
    "VD-N3-N-61.670-M-RS",
    "VD-N3-N-63.990-M-LOOP",
    "VD-N3-N-65.990-M-RS",
    "VD-N3-N-69.510-M-LOOP",
]

SPEED_COLS = [f"VD_Speed_kmh_Lane{i}" for i in range(5)]
OCC_COLS = [f"VD_Occupancy_Lane{i}" for i in range(5)]


# ============================================================
# 3. Helpers
# ============================================================

def parse_local_time(series: pd.Series) -> pd.Series:
    """移除錯誤標上的 Z，但不做 UTC +8 轉換。"""
    s = series.astype("string").str.strip().str.replace(r"Z$", "", regex=True)
    return pd.to_datetime(s, errors="coerce")


def weighted_station_values(speed_df: pd.DataFrame, occ_df: pd.DataFrame):
    """
    向量化計算每列 VD 的代表 Speed / Occupancy。

    Speed:
      - 只使用 Speed >= 0 且 Occupancy > 0 的車道做 occupancy-weighted speed
      - 若有可用車道，但所有可用 Occupancy 都為 0 => Speed = -1, Occupancy = 0
      - 其他無法形成可用速度 => Speed = -99

    Occupancy:
      - 對 Occupancy >= 0 的可用車道取算術平均
      - 完全無可用 Occupancy => -99
    """
    s = speed_df.apply(pd.to_numeric, errors="coerce").astype(float)
    o = occ_df.apply(pd.to_numeric, errors="coerce").astype(float)

    valid_occ = o.ge(0)
    positive_weight = s.ge(0) & o.gt(0)

    weights = o.where(positive_weight).sum(axis=1, min_count=1)
    numerator = (s.where(positive_weight) * o.where(positive_weight)).sum(axis=1, min_count=1)

    occ_mean = o.where(valid_occ).mean(axis=1)
    has_valid_occ = valid_occ.any(axis=1)
    all_valid_occ_zero = has_valid_occ & o.where(valid_occ).fillna(0).sum(axis=1).eq(0)
    has_positive_weight = weights.fillna(0).gt(0)

    out_speed = pd.Series(-99.0, index=s.index, dtype="float64")
    out_occ = occ_mean.fillna(-99.0).astype(float)

    out_speed.loc[has_positive_weight] = (
        numerator.loc[has_positive_weight] / weights.loc[has_positive_weight]
    )

    out_speed.loc[~has_positive_weight & all_valid_occ_zero] = -1.0
    out_occ.loc[~has_positive_weight & all_valid_occ_zero] = 0.0

    return out_speed, out_occ


def aggregate_candidate_group(g: pd.DataFrame):
    """由 audit 回抓的 lane-level rows 計算一個 VD 代表值。"""
    speed = pd.to_numeric(g["Speed_kmh"], errors="coerce")
    occ = pd.to_numeric(g["Occupancy"], errors="coerce")

    valid_occ = occ.ge(0)
    positive = speed.ge(0) & occ.gt(0)

    if positive.any() and occ.loc[positive].sum() > 0:
        weighted_speed = float((speed.loc[positive] * occ.loc[positive]).sum() / occ.loc[positive].sum())
        occ_mean = float(occ.loc[valid_occ].mean()) if valid_occ.any() else -99.0
        return weighted_speed, occ_mean

    if valid_occ.any() and float(occ.loc[valid_occ].sum()) == 0:
        return -1.0, 0.0

    occ_mean = float(occ.loc[valid_occ].mean()) if valid_occ.any() else -99.0
    return -99.0, occ_mean


def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"找不到檔案：{path}")


# ============================================================
# 4. 原整理檔 -> OBSERVED
# ============================================================

def build_observed():
    print("[1/5] 讀取同學整理後的 N3 北向 VD...")

    required_cols = ["Target_Time", "VDID"] + SPEED_COLS + OCC_COLS
    source = pd.read_csv(SOURCE_VD_CSV, usecols=required_cols)

    source["TargetTime"] = parse_local_time(source["Target_Time"])
    source = source[source["VDID"].isin(TARGET_VDS)].copy()

    if source["TargetTime"].isna().any():
        raise RuntimeError("SOURCE_VD_CSV 有無法解析的 Target_Time。")

    if source.duplicated(["TargetTime", "VDID"]).any():
        dup = source[source.duplicated(["TargetTime", "VDID"], keep=False)].head(20)
        raise RuntimeError(f"來源檔 TargetTime x VDID 有重複：\n{dup}")

    speed, occ = weighted_station_values(source[SPEED_COLS], source[OCC_COLS])

    observed = pd.DataFrame({
        "TargetTime": source["TargetTime"],
        "VDID": source["VDID"].astype("string"),
        "Speed": speed,
        "Occupancy": occ,
        "VDType": "OBSERVED",
    })

    # 理論上原整理檔中若無法形成代表速度，改明確標記 OFFICIAL_NA。
    observed.loc[observed["Speed"].eq(-99), "VDType"] = "OFFICIAL_NA"

    print(f"    原整理檔 rows: {len(observed):,}")
    return observed


# ============================================================
# 5. audit missing -> recovery / sentinel codes
# ============================================================

def build_missing_rows():
    print("[2/5] 讀取 missing archive audit...")

    audit = pd.read_csv(AUDIT_CSV)
    recovered = pd.read_csv(RECOVERED_CSV)

    audit["TargetTime"] = pd.to_datetime(audit["Target_Time"], errors="coerce")
    recovered["TargetTime"] = pd.to_datetime(recovered["Target_Time"], errors="coerce")

    audit = audit[audit["VDID"].isin(TARGET_VDS)].copy()
    recovered = recovered[recovered["VDID"].isin(TARGET_VDS)].copy()

    if audit["TargetTime"].isna().any():
        raise RuntimeError("AUDIT_CSV 有無法解析的 Target_Time。")

    # 建立候選資料索引，避免每一列 audit 都掃完整 recovered CSV。
    recovered["MatchOffsetMin"] = pd.to_numeric(recovered["MatchOffsetMin"], errors="coerce")
    candidate_groups = {
        (t, vd, int(offset)): g.copy()
        for (t, vd, offset), g in recovered.dropna(subset=["MatchOffsetMin"]).groupby(
            ["TargetTime", "VDID", "MatchOffsetMin"], sort=False
        )
    }

    out = []

    for row in audit.itertuples(index=False):
        t = row.TargetTime
        vd = row.VDID
        audit_class = row.AuditClass
        recoverable = bool(row.RecoverableWithinOriginalRule)
        offset_raw = row.BestRecoverableOffsetMin

        # ----------------------------------------------------
        # A. 可以從 exact / -1 / +1 找回真正觀測值
        # ----------------------------------------------------
        if recoverable and pd.notna(offset_raw):
            offset = int(offset_raw)
            key = (t, vd, offset)

            if key not in candidate_groups:
                raise RuntimeError(f"Audit 說可回補，但 recovered candidates 找不到：{key}")

            speed, occ = aggregate_candidate_group(candidate_groups[key])

            if offset == 0:
                vdtype = "RECOVERED_EXACT"
            elif offset == -1:
                vdtype = "RECOVERED_M1"
            elif offset == 1:
                vdtype = "RECOVERED_P1"
            else:
                raise RuntimeError(f"未預期的 recovery offset: {offset}")

            # 若回抓候選實際仍完全無法形成速度，保守改回 OFFICIAL_NA。
            if speed == -99:
                vdtype = "OFFICIAL_NA"

            out.append((t, vd, speed, occ, vdtype))
            continue

        # ----------------------------------------------------
        # B. VD 存在，但官方速度本身無效
        # ----------------------------------------------------
        if audit_class == "EXACT_VD_PRESENT_INVALID_SPEED":
            key = (t, vd, 0)

            occ_value = -99.0
            if key in candidate_groups:
                g = candidate_groups[key]
                occ_series = pd.to_numeric(g["Occupancy"], errors="coerce")
                valid_occ = occ_series[occ_series.ge(0)]
                if len(valid_occ) > 0:
                    occ_value = float(valid_occ.mean())

            out.append((t, vd, -99.0, occ_value, "OFFICIAL_NA"))
            continue

        # ----------------------------------------------------
        # C. XML 有，但該 VD 根本沒有出現在 XML
        # ----------------------------------------------------
        if audit_class == "VD_NOT_IN_EXACT_XML":
            out.append((t, vd, -999.0, -999.0, "VD_ABSENT"))
            continue

        # ----------------------------------------------------
        # D. 官方歷史 XML 本身不存在 / 無法取得
        # ----------------------------------------------------
        if audit_class == "EXACT_ARCHIVE_FILE_MISSING":
            out.append((t, vd, -9999.0, -9999.0, "FILE_UNAVAILABLE"))
            continue

        # ----------------------------------------------------
        # E. FOUND_EXACT_VALID / FOUND_PLUS1_VALID 理論上前面
        #    recoverable 已經處理。若落到這裡直接報錯。
        # ----------------------------------------------------
        raise RuntimeError(
            f"尚未定義的 AuditClass / recovery 狀態："
            f"TargetTime={t}, VDID={vd}, AuditClass={audit_class}, "
            f"Recoverable={recoverable}, Offset={offset_raw}"
        )

    missing_rows = pd.DataFrame(
        out,
        columns=["TargetTime", "VDID", "Speed", "Occupancy", "VDType"]
    )

    print(f"    audit missing rows: {len(missing_rows):,}")
    return missing_rows


# ============================================================
# 6. 合併並建立完整 5-minute x 5 VD grid
# ============================================================

def combine_and_qc(observed, missing_rows):
    print("[3/5] 合併 observed + audit missing...")

    final = pd.concat([observed, missing_rows], ignore_index=True)

    if final.duplicated(["TargetTime", "VDID"]).any():
        dup = final[final.duplicated(["TargetTime", "VDID"], keep=False)].sort_values(
            ["TargetTime", "VDID"]
        ).head(30)
        raise RuntimeError(f"合併後 TargetTime x VDID 重複：\n{dup}")

    # 兩年完整 5-min grid
    expected_times = pd.date_range(START_TIME, END_TIME, freq="5min")
    expected_rows = len(expected_times) * len(TARGET_VDS)

    if len(final) != expected_rows:
        expected = pd.MultiIndex.from_product(
            [expected_times, TARGET_VDS], names=["TargetTime", "VDID"]
        )
        actual = pd.MultiIndex.from_frame(final[["TargetTime", "VDID"]])
        missing_keys = expected.difference(actual)
        extra_keys = actual.difference(expected)
        raise RuntimeError(
            f"完整格點檢查失敗：expected={expected_rows:,}, actual={len(final):,}\n"
            f"missing keys={len(missing_keys):,}, extra keys={len(extra_keys):,}\n"
            f"missing sample={list(missing_keys[:10])}\n"
            f"extra sample={list(extra_keys[:10])}"
        )

    # 型態 / 排序
    order_map = {vd: i for i, vd in enumerate(TARGET_VDS)}
    final["_vd_order"] = final["VDID"].map(order_map)
    final = final.sort_values(["TargetTime", "_vd_order"]).drop(columns="_vd_order").reset_index(drop=True)

    final["Speed"] = pd.to_numeric(final["Speed"], errors="raise").astype("float32")
    final["Occupancy"] = pd.to_numeric(final["Occupancy"], errors="raise").astype("float32")
    final["VDID"] = final["VDID"].astype("string")
    final["VDType"] = final["VDType"].astype("string")

    allowed_types = {
        "OBSERVED",
        "RECOVERED_EXACT",
        "RECOVERED_M1",
        "RECOVERED_P1",
        "OFFICIAL_NA",
        "VD_ABSENT",
        "FILE_UNAVAILABLE",
    }
    unknown = set(final["VDType"].dropna().unique()) - allowed_types
    if unknown:
        raise RuntimeError(f"出現未知 VDType：{unknown}")

    print(f"    完整 TargetTime 數: {final['TargetTime'].nunique():,}")
    print(f"    VD 數: {final['VDID'].nunique():,}")
    print(f"    最終 rows: {len(final):,} (PASS)")

    return final


# ============================================================
# 7. 輸出
# ============================================================

def save_outputs(final):
    print("[4/5] 輸出 Parquet 與 QC summary...")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    try:
        final.to_parquet(
            OUT_PARQUET,
            index=False,
            compression="zstd",
        )
    except ImportError as e:
        raise RuntimeError(
            "輸出 Parquet 需要 pyarrow。請先執行：pip install pyarrow"
        ) from e

    summary = (
        final.groupby(["VDID", "VDType"], dropna=False)
        .size()
        .rename("Rows")
        .reset_index()
    )
    summary["PctWithinVD"] = (
        summary["Rows"]
        / summary.groupby("VDID")["Rows"].transform("sum")
        * 100
    )
    summary.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")

    codebook = pd.DataFrame([
        ["Speed / Occupancy", ">= 0", "正常可用數值"],
        ["Speed", "-1", "有 VD 觀測，但所有可用車道 Occupancy = 0；視為沒有偵測到車"],
        ["Speed / Occupancy", "-99", "VD 存在，但官方該變數無效 / 缺測"],
        ["Speed / Occupancy", "-999", "官方 XML 存在，但該 VDID 不在 XML 中"],
        ["Speed / Occupancy", "-9999", "官方歷史 XML 不存在或無法取得"],
        ["VDType", "OBSERVED", "原同學整理檔中已有"],
        ["VDType", "RECOVERED_EXACT", "回查官方同分鐘補回"],
        ["VDType", "RECOVERED_M1", "使用目標時間前 1 分鐘補回"],
        ["VDType", "RECOVERED_P1", "使用目標時間後 1 分鐘補回；未來即時預測需注意資訊洩漏"],
        ["VDType", "OFFICIAL_NA", "VD 存在，但官方速度無效"],
        ["VDType", "VD_ABSENT", "XML 存在，但該 VD 不在 XML"],
        ["VDType", "FILE_UNAVAILABLE", "官方 XML 不存在 / 無法取得"],
    ], columns=["Field", "Code", "Meaning"])
    codebook.to_csv(OUT_CODES, index=False, encoding="utf-8-sig")

    print("\nVDType summary:")
    print(final["VDType"].value_counts().to_string())
    print(f"\nParquet: {OUT_PARQUET}")
    print(f"Summary: {OUT_SUMMARY}")
    print(f"Codebook: {OUT_CODES}")


# ============================================================
# 8. Main
# ============================================================

def main():
    print("=" * 100)
    print("BUILD VD_N3_N_LITE 2024-2025")
    print("=" * 100)

    for p in [SOURCE_VD_CSV, AUDIT_CSV, RECOVERED_CSV]:
        require_file(p)

    observed = build_observed()
    missing_rows = build_missing_rows()
    final = combine_and_qc(observed, missing_rows)
    save_outputs(final)

    print("[5/5] COMPLETE")


if __name__ == "__main__":
    main()
