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

BASE_ROOT = Path(r"D:\統計實務\Data")
VD_ROOT = BASE_ROOT /"原始檔案"/ "VD"

# 目前三個輸入檔都放在 D:\統計實務\Data\VD\
SOURCE_VD_CSV = VD_ROOT / "N3_Daxi_Longtan_N_2024_2025_MASTER.csv"
AUDIT_CSV = BASE_ROOT / "VD_N3_N_LITE" /"VD_N3_N_MISSING_AUDIT_2024_2025"/ "02_missing_archive_audit.csv"
RECOVERED_CSV = BASE_ROOT / "VD_N3_N_LITE" /"VD_N3_N_MISSING_AUDIT_2024_2025"/"03_recovered_lane_candidates.csv"

OUT_ROOT = BASE_ROOT / "VD_N3_N_LITE_2024_2025"
OUT_PARQUET = OUT_ROOT / "VD_N3_N_2024_2025_LITE.parquet"
OUT_SUMMARY = OUT_ROOT / "01_vdtype_summary.csv"
OUT_CODES = OUT_ROOT / "02_value_codebook.csv"
OUT_SOURCE_QC = OUT_ROOT / "03_source_speed_recalc_qc.csv"


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

    注意：Speed 與 Occupancy 的欄位名稱不同，不能直接用 pandas DataFrame
    做 s.ge(0) & o.gt(0)。pandas 會依欄名對齊，會讓 Lane0~Lane4 錯位。
    這裡改用 NumPy，依欄位順序一一對應 Lane0~Lane4。

    Speed:
      - Speed >= 0 且 Occupancy > 0 的車道才參與 occupancy-weighted speed
      - 若有可用 Occupancy，但所有可用 Occupancy 都為 0 => Speed = -1, Occupancy = 0
      - 其他無法形成可用速度 => Speed = -99

    Occupancy:
      - 對 Occupancy >= 0 的可用車道取算術平均
      - 完全無可用 Occupancy => -99
    """
    s = speed_df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    o = occ_df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    valid_occ = np.isfinite(o) & (o >= 0)
    positive_weight = np.isfinite(s) & (s >= 0) & valid_occ & (o > 0)

    weights = np.where(positive_weight, o, 0.0).sum(axis=1)
    numerator = np.where(positive_weight, s * o, 0.0).sum(axis=1)

    valid_occ_count = valid_occ.sum(axis=1)
    occ_sum = np.where(valid_occ, o, 0.0).sum(axis=1)
    occ_mean = np.divide(
        occ_sum,
        valid_occ_count,
        out=np.full(len(s), np.nan, dtype=float),
        where=valid_occ_count > 0,
    )

    out_speed = np.full(len(s), -99.0, dtype=float)
    out_occ = np.where(np.isfinite(occ_mean), occ_mean, -99.0)

    has_positive_weight = weights > 0
    out_speed[has_positive_weight] = (
        numerator[has_positive_weight] / weights[has_positive_weight]
    )

    all_valid_occ_zero = (valid_occ_count > 0) & (occ_sum == 0)
    no_traffic = (~has_positive_weight) & all_valid_occ_zero
    out_speed[no_traffic] = -1.0
    out_occ[no_traffic] = 0.0

    return (
        pd.Series(out_speed, index=speed_df.index, dtype="float64"),
        pd.Series(out_occ, index=speed_df.index, dtype="float64"),
    )


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

    required_cols = ["Target_Time", "VDID", "VD_Weighted_Avg_Speed"] + SPEED_COLS + OCC_COLS
    source = pd.read_csv(SOURCE_VD_CSV, usecols=required_cols)

    source["TargetTime"] = parse_local_time(source["Target_Time"])
    source = source[source["VDID"].isin(TARGET_VDS)].copy()

    if source["TargetTime"].isna().any():
        raise RuntimeError("SOURCE_VD_CSV 有無法解析的 Target_Time。")

    if source.duplicated(["TargetTime", "VDID"]).any():
        dup = source[source.duplicated(["TargetTime", "VDID"], keep=False)].head(20)
        raise RuntimeError(f"來源檔 TargetTime x VDID 有重複：\n{dup}")

    speed, occ = weighted_station_values(source[SPEED_COLS], source[OCC_COLS])

    # ----------------------------------------------------
    # 強制驗證：重算代表速度必須與同學整理檔既有欄位一致
    # N3 北向 Lane4 在目前整理檔皆為缺值，因此應可直接一致。
    # ----------------------------------------------------
    reference_speed = pd.to_numeric(source["VD_Weighted_Avg_Speed"], errors="coerce")

    calc_np = speed.to_numpy(dtype=float)
    ref_np = reference_speed.to_numpy(dtype=float)

    both_normal = (ref_np >= 0) & (calc_np >= 0)
    both_minus1 = (ref_np == -1) & (calc_np == -1)
    normal_equal = np.zeros(len(source), dtype=bool)
    normal_equal[both_normal] = np.isclose(
        ref_np[both_normal],
        calc_np[both_normal],
        atol=1e-6,
        rtol=1e-8,
    )
    matched = normal_equal | both_minus1
    comparable = np.isfinite(ref_np)
    mismatch = comparable & ~matched

    if mismatch.any():
        idx = np.flatnonzero(mismatch)[:20]
        sample = source.iloc[idx][[
            "Target_Time", "VDID", "VD_Weighted_Avg_Speed"
        ]].copy()
        sample["RecalculatedSpeed"] = calc_np[idx]
        raise RuntimeError(
            f"重算 VD Speed 與來源整理檔不一致，共 {int(mismatch.sum()):,} 筆。\n"
            f"Sample:\n{sample.to_string(index=False)}"
        )

    observed = pd.DataFrame({
        "TargetTime": source["TargetTime"],
        "VDID": source["VDID"].astype("string"),
        "Speed": speed,
        "Occupancy": occ,
        "VDType": "OBSERVED",
    })

    # 原整理檔若真的存在無法形成代表速度的列，才標 OFFICIAL_NA。
    # Speed = -1 仍保留 VDType=OBSERVED；-1 本身已表示 Occupancy 全 0。
    observed.loc[observed["Speed"].eq(-99), "VDType"] = "OFFICIAL_NA"

    print(f"    原整理檔 rows: {len(observed):,}")
    print(f"    正常 Speed >= 0 : {(observed['Speed'] >= 0).sum():,}")
    print(f"    Speed = -1      : {(observed['Speed'] == -1).sum():,}")
    print(f"    Speed = -99     : {(observed['Speed'] == -99).sum():,}")

    # N3 北向目前整理檔應以正常值為主；若低於 90%，直接停止，避免錯誤資料流入 V2。
    normal_share = float((observed["Speed"] >= 0).mean())
    if normal_share < 0.90:
        raise RuntimeError(
            f"來源 VD 重算後正常 Speed 比例只有 {normal_share:.2%}，異常偏低，停止輸出。"
        )

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

    source_qc = pd.DataFrame([
        {
            "Metric": "TotalRows",
            "Value": int(len(final)),
            "Note": "完整 2024-2025 x 5 VD rows",
        },
        {
            "Metric": "ObservedRows",
            "Value": int((final["VDType"] == "OBSERVED").sum()),
            "Note": "原同學整理檔已有（含 Speed=-1 的無車狀態）",
        },
        {
            "Metric": "NormalSpeedRows",
            "Value": int((final["Speed"] >= 0).sum()),
            "Note": "Speed >= 0",
        },
        {
            "Metric": "SpeedMinus1Rows",
            "Value": int((final["Speed"] == -1).sum()),
            "Note": "Occupancy 全 0；包含原始或回補資料",
        },
        {
            "Metric": "OfficialNARows",
            "Value": int((final["VDType"] == "OFFICIAL_NA").sum()),
            "Note": "VD 存在但官方速度無效",
        },
        {
            "Metric": "RecoveredRows",
            "Value": int(final["VDType"].isin(["RECOVERED_EXACT", "RECOVERED_M1", "RECOVERED_P1"]).sum()),
            "Note": "回查官方資料找回",
        },
    ])
    source_qc.to_csv(OUT_SOURCE_QC, index=False, encoding="utf-8-sig")

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
