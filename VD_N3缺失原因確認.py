# -*- coding: utf-8 -*-
"""
audit_n3_north_vd_missing_archive.py

目的
----
只針對「目前整理後的 N3 北向 VD CSV 中缺失的 Target_Time x VDID」
回頭查高速公路局歷史 VD archive，判斷缺失原因。

官方歷史 URL 格式（沿用你們原始下載程式）：
https://tisvcloud.freeway.gov.tw/history/motc20/VD/YYYYMMDD/VDLive_HHMM.xml.gz

原始下載程式的候選分鐘順序：
Target、Target+1 min、Target-1 min

本程式與原始下載程式不同之處：
- 不會在第一個 HTTP 200 後就停止。
- 會針對每一個 missing VD 分別檢查 Target、+1、-1。
- 會分辨：XML 不存在 / VD 不在 XML / VD 有但 Speed 無效 / VD 有有效 lane。
- 不修改原始 CSV，只產生 audit 結果與可回收的 lane-level 候選資料。
- 有 XML cache 與 404 marker，可中斷後續跑。

建議先跑完 audit，不要直接補回 MASTER。
"""

from pathlib import Path
import gzip
import io
import time
import json
import math
import requests
import urllib3
import pandas as pd
import xml.etree.ElementTree as ET


# ============================================================
# 1. 使用者設定
# ============================================================

# 改成你電腦上「整理後 N3 北向 VD CSV」的實際路徑
SOURCE_CSV = Path(
    r"D:\統計實務\Data\VD\N3_Daxi_Longtan_N_2024_2025_MASTER.csv"
)

OUTPUT_ROOT = Path(
    r"D:\統計實務\Data\VD_N3_N_MISSING_AUDIT_2024_2025"
)

CACHE_ROOT = OUTPUT_ROOT / "_archive_cache"

START_TIME = pd.Timestamp("2024-01-01 00:00:00")
END_TIME = pd.Timestamp("2025-12-31 23:55:00")

# N3 北向目前整理檔中應有的 5 支 VD
TARGET_VDS = [
    "VD-N3-N-61.000-M-LOOP",
    "VD-N3-N-61.670-M-RS",
    "VD-N3-N-63.990-M-LOOP",
    "VD-N3-N-65.990-M-RS",
    "VD-N3-N-69.510-M-LOOP",
]

# 完全依照組員原始下載程式：T、T+1、T-1
CHECK_OFFSETS_MIN = [0, 1, -1]

ARCHIVE_BASE = "https://tisvcloud.freeway.gov.tw/history/motc20/VD"

# 原程式 verify=False；這裡保留相同行為。
VERIFY_SSL = False
REQUEST_TIMEOUT = 12
MAX_RETRIES = 3
REQUEST_SLEEP_SEC = 0.15

# 每處理多少個 missing Target_Time 做一次 checkpoint
CHECKPOINT_EVERY_TARGETS = 100

# 是否把找到的 lane-level 候選資料另外輸出成 CSV
SAVE_RECOVERED_LANES = True


# ============================================================
# 2. 初始化
# ============================================================

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
CACHE_ROOT.mkdir(parents=True, exist_ok=True)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (academic VD missing audit; "
        "N3 Daxi-Longtan 2024-2025)"
    )
})


# ============================================================
# 3. 小工具
# ============================================================

def local_name(tag: str) -> str:
    """去除 XML namespace。"""
    return tag.split("}")[-1]


def text_of_direct_child(elem, child_name):
    for child in list(elem):
        if local_name(child.tag) == child_name:
            return child.text.strip() if child.text else None
    return None


def text_of_first_descendant(elem, child_name):
    for child in elem.iter():
        if local_name(child.tag) == child_name:
            return child.text.strip() if child.text else None
    return None


def safe_float(x):
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def parse_target_time_as_local(series: pd.Series) -> pd.Series:
    """
    你的整理檔 Target_Time 可能長成：
      2024-01-01T00:00:00.000Z

    但你們原本抓取時的 File_Time 實際意義是台灣 local clock。
    因此這裡只把尾端 Z 拿掉，不做 UTC -> +08:00 的時區位移。
    """
    s = series.astype(str).str.strip().str.replace(r"Z$", "", regex=True)
    return pd.to_datetime(s, errors="coerce")


def archive_url(dt: pd.Timestamp) -> str:
    date_str = dt.strftime("%Y%m%d")
    time_str = dt.strftime("%H%M")
    return f"{ARCHIVE_BASE}/{date_str}/VDLive_{time_str}.xml.gz"


def cache_paths(dt: pd.Timestamp):
    date_dir = CACHE_ROOT / dt.strftime("%Y%m%d")
    date_dir.mkdir(parents=True, exist_ok=True)
    stem = f"VDLive_{dt.strftime('%H%M')}"
    gz_path = date_dir / f"{stem}.xml.gz"
    missing_marker = date_dir / f"{stem}.404"
    error_marker = date_dir / f"{stem}.error.json"
    return gz_path, missing_marker, error_marker


# ============================================================
# 4. 下載 archive（含 cache / retry）
# ============================================================

def fetch_archive_gz(dt: pd.Timestamp):
    """
    回傳 dict：
    {
      http_status,
      exists,
      source,
      url,
      content,
      error
    }

    content 是 gzip bytes（如果 exists=True）。
    """
    url = archive_url(dt)
    gz_path, missing_marker, error_marker = cache_paths(dt)

    # 已下載成功
    if gz_path.exists():
        return {
            "http_status": 200,
            "exists": True,
            "source": "CACHE",
            "url": url,
            "content": gz_path.read_bytes(),
            "error": None,
        }

    # 已知 404
    if missing_marker.exists():
        return {
            "http_status": 404,
            "exists": False,
            "source": "CACHE_404",
            "url": url,
            "content": None,
            "error": None,
        }

    last_error = None
    last_status = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = SESSION.get(
                url,
                timeout=REQUEST_TIMEOUT,
                verify=VERIFY_SSL,
            )
            last_status = response.status_code

            if response.status_code == 200:
                # 確認真的可以 gzip 解壓，避免把 HTML error page cache 起來
                try:
                    gzip.decompress(response.content)
                except Exception as e:
                    last_error = f"HTTP200_BUT_INVALID_GZIP: {e}"
                else:
                    gz_path.write_bytes(response.content)
                    time.sleep(REQUEST_SLEEP_SEC)
                    return {
                        "http_status": 200,
                        "exists": True,
                        "source": "WEB",
                        "url": url,
                        "content": response.content,
                        "error": None,
                    }

            elif response.status_code == 404:
                missing_marker.write_text(url, encoding="utf-8")
                time.sleep(REQUEST_SLEEP_SEC)
                return {
                    "http_status": 404,
                    "exists": False,
                    "source": "WEB",
                    "url": url,
                    "content": None,
                    "error": None,
                }

            else:
                last_error = f"HTTP_{response.status_code}"

        except requests.RequestException as e:
            last_error = repr(e)

        if attempt < MAX_RETRIES:
            time.sleep(1.0 * attempt)

    payload = {
        "url": url,
        "http_status": last_status,
        "error": last_error,
    }
    error_marker.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "http_status": last_status,
        "exists": False,
        "source": "ERROR",
        "url": url,
        "content": None,
        "error": last_error,
    }


# ============================================================
# 5. 解析一個 VD 在 XML 中的狀態
# ============================================================

def inspect_vd_in_archive(gz_bytes: bytes, wanted_vdid: str, archive_time: pd.Timestamp):
    """
    回傳：
      vd_present
      status
      data_collect_time
      lane_count
      valid_lane_count      (Speed >= 0，完全模擬原始程式可寫入條件)
      invalid_lane_count    (Speed < 0 或無法轉數字)
      lane_rows             (所有 lane，不先丟 -99)
    """
    xml_bytes = gzip.decompress(gz_bytes)
    root = ET.fromstring(xml_bytes)

    # 原始程式 soup.find("DataCollectTime") 是抓文件中第一個 DataCollectTime
    global_data_collect_time = text_of_first_descendant(root, "DataCollectTime")

    found_node = None

    for node in root.iter():
        if local_name(node.tag) != "VDLive":
            continue
        vdid = text_of_direct_child(node, "VDID")
        if vdid is None:
            vdid = text_of_first_descendant(node, "VDID")
        if vdid == wanted_vdid:
            found_node = node
            break

    if found_node is None:
        return {
            "vd_present": False,
            "status": None,
            "data_collect_time": global_data_collect_time,
            "lane_count": 0,
            "valid_lane_count": 0,
            "invalid_lane_count": 0,
            "lane_rows": [],
        }

    status = text_of_direct_child(found_node, "Status")
    if status is None:
        status = text_of_first_descendant(found_node, "Status")

    # 如果 VD 節點自己有 DataCollectTime，優先使用；否則沿用文件層級時間
    vd_data_collect_time = text_of_first_descendant(found_node, "DataCollectTime")
    data_collect_time = vd_data_collect_time or global_data_collect_time

    lane_rows = []

    for lane in found_node.iter():
        if local_name(lane.tag) != "Lane":
            continue

        lane_id = text_of_direct_child(lane, "LaneID")
        lane_type = text_of_direct_child(lane, "LaneType")
        speed_text = text_of_direct_child(lane, "Speed")
        occ_text = text_of_direct_child(lane, "Occupancy")

        speed = safe_float(speed_text)
        occupancy = safe_float(occ_text)

        # 完全對照原始寫入條件：speed >= 0 才會被 writer.writerow
        speed_valid_for_original = speed is not None and speed >= 0

        lane_rows.append({
            "Archive_Time": archive_time,
            "DataCollectTime": data_collect_time,
            "VDID": wanted_vdid,
            "Status": status,
            "LaneID": lane_id,
            "LaneType": lane_type,
            "Speed_kmh": speed,
            "Occupancy": occupancy,
            "SpeedValidForOriginal": speed_valid_for_original,
        })

    valid_lane_count = sum(bool(x["SpeedValidForOriginal"]) for x in lane_rows)
    invalid_lane_count = len(lane_rows) - valid_lane_count

    return {
        "vd_present": True,
        "status": status,
        "data_collect_time": data_collect_time,
        "lane_count": len(lane_rows),
        "valid_lane_count": valid_lane_count,
        "invalid_lane_count": invalid_lane_count,
        "lane_rows": lane_rows,
    }


# ============================================================
# 6. 找目前整理檔真正 missing 的 Target_Time x VDID
# ============================================================

def build_missing_table():
    print("=" * 100)
    print("STEP 1 - FIND MISSING TARGET_TIME x VDID")
    print("=" * 100)
    print("Source:", SOURCE_CSV)

    if not SOURCE_CSV.exists():
        raise FileNotFoundError(
            f"找不到來源 CSV：\n{SOURCE_CSV}\n\n"
            "請先修改程式最上方 SOURCE_CSV。"
        )

    df = pd.read_csv(
        SOURCE_CSV,
        usecols=["Target_Time", "VDID"],
        dtype={"VDID": "string"},
        low_memory=False,
    )

    df["Target_Time"] = parse_target_time_as_local(df["Target_Time"])
    df = df.dropna(subset=["Target_Time", "VDID"])
    df = df[df["VDID"].isin(TARGET_VDS)]

    observed = (
        df[["Target_Time", "VDID"]]
        .drop_duplicates()
        .assign(HasData=1)
    )

    full_times = pd.date_range(START_TIME, END_TIME, freq="5min")
    full_grid = pd.MultiIndex.from_product(
        [full_times, TARGET_VDS],
        names=["Target_Time", "VDID"],
    ).to_frame(index=False)

    check = full_grid.merge(
        observed,
        on=["Target_Time", "VDID"],
        how="left",
    )
    check["HasData"] = check["HasData"].fillna(0).astype("int8")

    missing = (
        check.loc[check["HasData"] == 0, ["Target_Time", "VDID"]]
        .sort_values(["Target_Time", "VDID"])
        .reset_index(drop=True)
    )

    missing_path = OUTPUT_ROOT / "01_missing_target_vd.csv"
    missing.to_csv(missing_path, index=False, encoding="utf-8-sig")

    print(f"Expected 5-min slots / VD : {len(full_times):,}")
    print(f"Missing TargetTime x VDID : {len(missing):,}")
    print(f"Unique missing TargetTime : {missing['Target_Time'].nunique():,}")
    print("Saved:", missing_path)

    print("\nMissing count by VD:")
    print(missing.groupby("VDID").size().to_string())

    return missing


# ============================================================
# 7. Audit 一筆 missing Target_Time x VDID
# ============================================================

def audit_one_missing(target_time: pd.Timestamp, vdid: str, archive_cache: dict):
    row = {
        "Target_Time": target_time,
        "VDID": vdid,
    }

    recovered_lane_rows = []
    per_offset_results = {}

    for offset in CHECK_OFFSETS_MIN:
        archive_time = target_time + pd.Timedelta(minutes=offset)
        archive_key = archive_time.strftime("%Y-%m-%d %H:%M")

        # 同一 archive minute 只下載 / parse gzip 一次
        if archive_key not in archive_cache:
            archive_cache[archive_key] = fetch_archive_gz(archive_time)

        fetched = archive_cache[archive_key]

        prefix = (
            "Exact" if offset == 0
            else "Plus1" if offset == 1
            else "Minus1"
        )

        row[f"{prefix}_ArchiveTime"] = archive_time
        row[f"{prefix}_URL"] = fetched["url"]
        row[f"{prefix}_HTTPStatus"] = fetched["http_status"]
        row[f"{prefix}_FileExists"] = bool(fetched["exists"])
        row[f"{prefix}_FetchSource"] = fetched["source"]
        row[f"{prefix}_FetchError"] = fetched["error"]

        if not fetched["exists"]:
            row[f"{prefix}_VDPresent"] = False
            row[f"{prefix}_VDStatus"] = None
            row[f"{prefix}_DataCollectTime"] = None
            row[f"{prefix}_LaneCount"] = 0
            row[f"{prefix}_ValidLaneCount"] = 0
            row[f"{prefix}_InvalidLaneCount"] = 0

            per_offset_results[offset] = {
                "file_exists": False,
                "vd_present": False,
                "valid_lane_count": 0,
                "invalid_lane_count": 0,
            }
            continue

        try:
            vdinfo = inspect_vd_in_archive(
                fetched["content"],
                wanted_vdid=vdid,
                archive_time=archive_time,
            )
        except Exception as e:
            row[f"{prefix}_VDPresent"] = False
            row[f"{prefix}_VDStatus"] = None
            row[f"{prefix}_DataCollectTime"] = None
            row[f"{prefix}_LaneCount"] = 0
            row[f"{prefix}_ValidLaneCount"] = 0
            row[f"{prefix}_InvalidLaneCount"] = 0
            row[f"{prefix}_ParseError"] = repr(e)

            per_offset_results[offset] = {
                "file_exists": True,
                "vd_present": False,
                "valid_lane_count": 0,
                "invalid_lane_count": 0,
            }
            continue

        row[f"{prefix}_VDPresent"] = bool(vdinfo["vd_present"])
        row[f"{prefix}_VDStatus"] = vdinfo["status"]
        row[f"{prefix}_DataCollectTime"] = vdinfo["data_collect_time"]
        row[f"{prefix}_LaneCount"] = vdinfo["lane_count"]
        row[f"{prefix}_ValidLaneCount"] = vdinfo["valid_lane_count"]
        row[f"{prefix}_InvalidLaneCount"] = vdinfo["invalid_lane_count"]

        per_offset_results[offset] = {
            "file_exists": True,
            "vd_present": bool(vdinfo["vd_present"]),
            "valid_lane_count": int(vdinfo["valid_lane_count"]),
            "invalid_lane_count": int(vdinfo["invalid_lane_count"]),
        }

        # 把找到的 lane 原始候選存下來；之後真正 backfill 時再決定用哪分鐘
        for lane_row in vdinfo["lane_rows"]:
            x = dict(lane_row)
            x["Target_Time"] = target_time
            x["MatchOffsetMin"] = offset
            x["CandidateType"] = prefix
            recovered_lane_rows.append(x)

    # --------------------------------------------------------
    # MissingReason 分類
    # --------------------------------------------------------
    exact = per_offset_results.get(0, {})
    plus1 = per_offset_results.get(1, {})
    minus1 = per_offset_results.get(-1, {})

    exact_valid = exact.get("valid_lane_count", 0) > 0
    plus1_valid = plus1.get("valid_lane_count", 0) > 0
    minus1_valid = minus1.get("valid_lane_count", 0) > 0

    exact_present = exact.get("vd_present", False)
    exact_file_exists = exact.get("file_exists", False)

    if exact_valid:
        audit_class = "FOUND_EXACT_VALID"
        likely_reason = (
            "官方同分鐘有可用 VD；原整理檔卻缺失。"
            "優先懷疑當初下載/寫檔/月份斷點流程漏抓。"
        )
        best_offset = 0

    elif exact_present and exact.get("invalid_lane_count", 0) > 0:
        if plus1_valid:
            audit_class = "EXACT_INVALID_BUT_PLUS1_VALID"
            likely_reason = (
                "同分鐘 VD 存在，但沒有 Speed>=0 的 lane；"
                "原程式會因 speed<0 不寫入。+1 分鐘有可用資料。"
            )
            best_offset = 1
        elif minus1_valid:
            audit_class = "EXACT_INVALID_BUT_MINUS1_VALID"
            likely_reason = (
                "同分鐘 VD 存在，但沒有 Speed>=0 的 lane；"
                "原程式會因 speed<0 不寫入。-1 分鐘有可用資料。"
            )
            best_offset = -1
        else:
            audit_class = "EXACT_VD_PRESENT_INVALID_SPEED"
            likely_reason = (
                "官方同分鐘有 VD 節點，但 lane Speed 全部無效/負值；"
                "這符合原程式被 speed>=0 條件排除的情況。"
            )
            best_offset = None

    elif plus1_valid:
        audit_class = "FOUND_PLUS1_VALID"
        likely_reason = (
            "同分鐘沒有可用 VD，但 +1 分鐘有。"
            "原程式若同分鐘 XML 本身 HTTP 200，會先 break，"
            "因此可能沒有繼續搜尋 +1。"
        )
        best_offset = 1

    elif minus1_valid:
        audit_class = "FOUND_MINUS1_VALID"
        likely_reason = (
            "同分鐘與 +1 沒有可用 VD，但 -1 分鐘有。"
            "原程式可能因較前面的候選 XML HTTP 200 而提前 break。"
        )
        best_offset = -1

    elif not exact_file_exists:
        audit_class = "EXACT_ARCHIVE_FILE_MISSING"
        likely_reason = (
            "官方同分鐘歷史 XML 不存在或無法取得；"
            "且 +/-1 分鐘也未找到可用 VD。"
        )
        best_offset = None

    elif not exact_present:
        audit_class = "VD_NOT_IN_EXACT_XML"
        likely_reason = (
            "官方同分鐘 XML 存在，但該 VDID 不在 XML 中；"
            "且 +/-1 分鐘也未找到可用資料。"
        )
        best_offset = None

    else:
        audit_class = "NOT_FOUND_VALID_WITHIN_PLUS_MINUS_1"
        likely_reason = (
            "Target、+1、-1 都沒有找到符合原始 speed>=0 寫入條件的 VD。"
        )
        best_offset = None

    row["AuditClass"] = audit_class
    row["LikelyReason"] = likely_reason
    row["BestRecoverableOffsetMin"] = best_offset
    row["RecoverableWithinOriginalRule"] = best_offset is not None

    return row, recovered_lane_rows


# ============================================================
# 8. Checkpoint / summary
# ============================================================

def write_checkpoint(audit_rows, lane_rows):
    if audit_rows:
        pd.DataFrame(audit_rows).to_csv(
            OUTPUT_ROOT / "02_missing_archive_audit_CHECKPOINT.csv",
            index=False,
            encoding="utf-8-sig",
        )

    if SAVE_RECOVERED_LANES and lane_rows:
        pd.DataFrame(lane_rows).to_csv(
            OUTPUT_ROOT / "03_recovered_lane_candidates_CHECKPOINT.csv",
            index=False,
            encoding="utf-8-sig",
        )


def save_final(audit_rows, lane_rows):
    audit = pd.DataFrame(audit_rows)
    audit = audit.sort_values(["Target_Time", "VDID"]).reset_index(drop=True)

    audit_path = OUTPUT_ROOT / "02_missing_archive_audit.csv"
    audit.to_csv(audit_path, index=False, encoding="utf-8-sig")

    if SAVE_RECOVERED_LANES and lane_rows:
        lanes = pd.DataFrame(lane_rows)
        lanes = lanes.sort_values(
            ["Target_Time", "VDID", "MatchOffsetMin", "LaneID"],
            na_position="last",
        ).reset_index(drop=True)

        lane_path = OUTPUT_ROOT / "03_recovered_lane_candidates.csv"
        lanes.to_csv(lane_path, index=False, encoding="utf-8-sig")
    else:
        lane_path = None

    # Summary by VD x reason
    summary_vd = (
        audit.groupby(["VDID", "AuditClass"], dropna=False)
        .size()
        .reset_index(name="Count")
    )
    summary_vd["PctWithinVDMissing"] = (
        100.0
        * summary_vd["Count"]
        / summary_vd.groupby("VDID")["Count"].transform("sum")
    )

    summary_vd_path = OUTPUT_ROOT / "04_summary_by_vd_reason.csv"
    summary_vd.to_csv(summary_vd_path, index=False, encoding="utf-8-sig")

    # Overall reason summary
    summary_reason = (
        audit.groupby("AuditClass", dropna=False)
        .size()
        .reset_index(name="Count")
        .sort_values("Count", ascending=False)
    )
    summary_reason["Pct"] = 100.0 * summary_reason["Count"] / len(audit)

    summary_reason_path = OUTPUT_ROOT / "05_summary_by_reason.csv"
    summary_reason.to_csv(summary_reason_path, index=False, encoding="utf-8-sig")

    # Recoverable summary
    recoverable = audit["RecoverableWithinOriginalRule"].fillna(False).astype(bool)
    print("\n" + "=" * 100)
    print("AUDIT SUMMARY")
    print("=" * 100)
    print(f"Missing rows audited : {len(audit):,}")
    print(f"Recoverable T/+1/-1  : {recoverable.sum():,} ({100*recoverable.mean():.2f}%)")
    print("\nReason summary:")
    print(summary_reason.to_string(index=False))

    print("\nSaved:")
    print(" -", audit_path)
    if lane_path is not None:
        print(" -", lane_path)
    print(" -", summary_vd_path)
    print(" -", summary_reason_path)


# ============================================================
# 9. Main
# ============================================================

def main():
    print("=" * 100)
    print("N3 NORTH VD MISSING -> OFFICIAL ARCHIVE AUDIT")
    print("=" * 100)
    print("Source CSV :", SOURCE_CSV)
    print("Output     :", OUTPUT_ROOT)
    print("Offsets    :", CHECK_OFFSETS_MIN)
    print("Rule       : Speed >= 0 == would have been written by original downloader")

    missing = build_missing_table()

    audit_rows = []
    recovered_lane_rows = []
    archive_cache = {}

    grouped = list(missing.groupby("Target_Time", sort=True))
    total_targets = len(grouped)

    print("\n" + "=" * 100)
    print("STEP 2 - CHECK ONLY MISSING TIMES AGAINST OFFICIAL ARCHIVE")
    print("=" * 100)
    print(f"Unique missing Target_Time to audit: {total_targets:,}")

    for target_i, (target_time, g) in enumerate(grouped, start=1):
        for vdid in g["VDID"].tolist():
            audit_row, lane_rows = audit_one_missing(
                target_time=target_time,
                vdid=vdid,
                archive_cache=archive_cache,
            )
            audit_rows.append(audit_row)
            recovered_lane_rows.extend(lane_rows)

        if target_i % 25 == 0 or target_i == total_targets:
            print(
                f"[{target_i:,}/{total_targets:,}] "
                f"Target={target_time} | audited rows={len(audit_rows):,}"
            )

        if target_i % CHECKPOINT_EVERY_TARGETS == 0:
            write_checkpoint(audit_rows, recovered_lane_rows)

    save_final(audit_rows, recovered_lane_rows)

    print("\n" + "=" * 100)
    print("COMPLETE")
    print("=" * 100)
    print("這支程式只做 audit，沒有修改原本整理後 CSV。")


if __name__ == "__main__":
    main()
