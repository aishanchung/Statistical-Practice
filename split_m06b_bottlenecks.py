from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import csv
import re

try:
    import polars as pl
except ImportError as e:
    raise SystemExit(
        "尚未安裝 polars。請先在虛擬環境執行：\n"
        "    python -m pip install polars\n"
        "再重新執行本程式。"
    ) from e

# ============================================================
# 使用者設定
# ============================================================
INPUT_ROOT = Path(r"D:\M06B")
OUTPUT_ROOT = Path(r"D:\M06B_BOTTLENECK")
START_DATE = "20240101"
END_DATE = "20251231"
OVERWRITE = False

# 已確認部分原始壓縮檔本身可能就是空檔，例如 2024-01-16 13、14 時。
# True：存在但為空的來源檔，記錄後繼續，該日仍可視為 preprocessing 完成。
ALLOW_EXISTING_EMPTY_FILES = True

# M06B：無表頭，共 9 欄
COLUMNS = [
    "VehicleID",
    "VehicleType",
    "DetectionTimeO",
    "GantryO",
    "DetectionTimeD",
    "GantryD",
    "TripLength",
    "TripEnd",
    "TripInformation",
]
SCHEMA = {name: pl.String for name in COLUMNS}

# 目前先切兩個邊界清楚的瓶頸，共 4 個方向
TARGETS = {
    "N5_Pinglin_Toucheng_S": {
        "code": "05F0287S",
        "bottleneck": "N5_Pinglin_Toucheng",
        "direction": "S",
        "label": "國5 坪林->頭城",
    },
    "N5_Pinglin_Toucheng_N": {
        "code": "05F0287N",
        "bottleneck": "N5_Pinglin_Toucheng",
        "direction": "N",
        "label": "國5 頭城->坪林",
    },
    "N3_Daxi_Longtan_S": {
        "code": "03F0648S",
        "bottleneck": "N3_Daxi_Longtan",
        "direction": "S",
        "label": "國3 大溪->龍潭",
    },
    "N3_Daxi_Longtan_N": {
        "code": "03F0648N",
        "bottleneck": "N3_Daxi_Longtan",
        "direction": "N",
        "label": "國3 龍潭->大溪",
    },
}

SUMMARY_FIELDS = [
    "SourceDate",
    "Target",
    "Label",
    "GantryCode",
    "NTrips",
    "NUniqueVehicles",
    "NExpectedHours",
    "NReadableHours",
    "NInputFiles",
    "NSourceEmptyHours",
    "SourceEmptyHours",
    "NMalformedSourceRows",
    "MalformedHours",
    "NMissingHours",
    "MissingHours",
    "Status",
]

ISSUE_FIELDS = [
    "SourceDate",
    "SourceHour",
    "IssueType",
    "FilePath",
    "Note",
]

MALFORMED_FIELDS = [
    "SourceDate",
    "SourceHour",
    "FilePath",
    "LineNumber",
    "FieldCount",
    "FirstField",
    "RowPreview",
    "Action",
]

OUTPUT_SCHEMA = {
    **SCHEMA,
    "Bottleneck": pl.String,
    "Direction": pl.String,
    "BottleneckGantry": pl.String,
    "BottleneckTime": pl.String,
    "SourceDate": pl.String,
    "SourceHour": pl.String,
    "SourceFile": pl.String,
}


def daterange(start_yyyymmdd: str, end_yyyymmdd: str):
    d = datetime.strptime(start_yyyymmdd, "%Y%m%d").date()
    end = datetime.strptime(end_yyyymmdd, "%Y%m%d").date()
    while d <= end:
        yield d
        d += timedelta(days=1)


def output_file(target_key: str, day_str: str) -> Path:
    meta = TARGETS[target_key]
    year, month = day_str[:4], day_str[4:6]
    out_dir = OUTPUT_ROOT / meta["bottleneck"] / meta["direction"] / year / month
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{target_key}_{day_str}.parquet"


def load_csv_rows(path: Path, key_fields: tuple[str, ...]) -> dict[tuple[str, ...], dict]:
    rows: dict[tuple[str, ...], dict] = {}
    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if all(row.get(k) not in (None, "") for k in key_fields):
                key = tuple(row[k] for k in key_fields)
                rows[key] = row
    return rows


def write_csv_rows(path: Path, rows: dict, fieldnames: list[str], sort_key) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows.values(), key=sort_key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(ordered)


def add_issue(
    issue_rows: dict[tuple[str, str, str, str], dict],
    day_str: str,
    hour: str,
    issue_type: str,
    file_path: str,
    note: str,
) -> None:
    key = (day_str, hour, issue_type, file_path)
    issue_rows[key] = {
        "SourceDate": day_str,
        "SourceHour": hour,
        "IssueType": issue_type,
        "FilePath": file_path,
        "Note": note,
    }


def remove_issue(
    issue_rows: dict[tuple[str, str, str, str], dict],
    day_str: str,
    hour: str,
    issue_type: str,
    file_path: str,
) -> None:
    issue_rows.pop((day_str, hour, issue_type, file_path), None)


def empty_output_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=OUTPUT_SCHEMA)


def row_preview(row: list[str], max_chars: int = 600) -> str:
    text = " | ".join(row)
    if len(text) > max_chars:
        return text[:max_chars] + " ..."
    return text


def safe_fallback_read_csv(
    file_path: Path,
    day_str: str,
    hh: str,
    malformed_rows: dict[tuple[str, str, str, str], dict],
) -> tuple[pl.DataFrame, int]:
    """
    只有正常 Polars 讀取失敗時才使用。

    原則：
    - 正常 9 欄 physical row：保留。
    - 非 9 欄 physical row：不猜、不修，自動隔離並記錄。
    - 若整個檔案沒有任何正常 9 欄資料，視為 fallback 失敗。
    """
    good_rows: list[list[str]] = []
    n_bad = 0

    with file_path.open(
        "r",
        encoding="utf-8-sig",
        errors="replace",
        newline="",
    ) as f:
        reader = csv.reader(f)
        for line_no, row in enumerate(reader, start=1):
            if len(row) == len(COLUMNS):
                good_rows.append(row)
                continue

            n_bad += 1
            key = (day_str, hh, str(file_path), str(line_no))
            malformed_rows[key] = {
                "SourceDate": day_str,
                "SourceHour": hh,
                "FilePath": str(file_path),
                "LineNumber": line_no,
                "FieldCount": len(row),
                "FirstField": row[0] if row else "",
                "RowPreview": row_preview(row),
                "Action": "EXCLUDED_PHYSICAL_ROW",
            }

    if not good_rows:
        raise ValueError(
            f"fallback 後找不到任何正常 {len(COLUMNS)} 欄資料；"
            f"malformed_rows={n_bad}"
        )

    # 用 orient='row' 明確表示每個 list 是一列。
    df = pl.DataFrame(good_rows, schema=COLUMNS, orient="row").cast(SCHEMA, strict=False)
    return df, n_bad


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    done_dir = OUTPUT_ROOT / "_done"
    done_dir.mkdir(parents=True, exist_ok=True)

    log_dir = OUTPUT_ROOT / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    summary_path = log_dir / "daily_summary.csv"
    issue_path = log_dir / "source_issues.csv"
    malformed_path = log_dir / "malformed_rows.csv"

    summary_rows = load_csv_rows(summary_path, ("SourceDate", "Target"))

    # 相容舊版 summary。
    for row in summary_rows.values():
        try:
            n_missing_old = int(row.get("NMissingHours") or 0)
        except ValueError:
            n_missing_old = 0
        try:
            n_empty_old = int(row.get("NSourceEmptyHours") or row.get("NEmptyFiles") or 0)
        except ValueError:
            n_empty_old = 0

        row.setdefault("NExpectedHours", "24")
        if not row.get("NReadableHours"):
            row["NReadableHours"] = str(max(0, 24 - n_missing_old - n_empty_old))
        row.setdefault("NSourceEmptyHours", str(n_empty_old))
        row.setdefault("SourceEmptyHours", "")
        row.setdefault("NMalformedSourceRows", "0")
        row.setdefault("MalformedHours", "")
        row.setdefault("MissingHours", "")
        if row.get("Status") == "OK":
            row["Status"] = "COMPLETE"

    issue_rows = load_csv_rows(
        issue_path,
        ("SourceDate", "SourceHour", "IssueType", "FilePath"),
    )
    malformed_rows = load_csv_rows(
        malformed_path,
        ("SourceDate", "SourceHour", "FilePath", "LineNumber"),
    )

    for day in daterange(START_DATE, END_DATE):
        day_str = day.strftime("%Y%m%d")
        done_marker = done_dir / f"{day_str}.done"

        if done_marker.exists() and not OVERWRITE:
            print(f"[SKIP] {day_str} 已完成")
            continue

        print(f"\n========== {day_str} ==========")

        daily_parts: dict[str, list[pl.DataFrame]] = {k: [] for k in TARGETS}

        n_input_files = 0
        readable_hours: set[str] = set()
        source_empty_hours: set[str] = set()
        malformed_hours: set[str] = set()
        missing_hours: set[str] = set()
        unexpected_empty = False
        n_malformed_source_rows = 0

        for hour in range(24):
            hh = f"{hour:02d}"
            hour_dir = INPUT_ROOT / day_str / hh
            files = sorted(hour_dir.glob(f"TDCS_M06B_{day_str}_{hh}*.csv"))

            if not files:
                missing_hours.add(hh)
                print(f"  [MISS] {day_str} {hh}: 找不到 CSV")
                add_issue(
                    issue_rows,
                    day_str,
                    hh,
                    "MISSING_FILE",
                    str(hour_dir),
                    "該小時找不到符合命名規則的 M06B CSV。",
                )
                continue

            hour_has_readable_file = False
            hour_has_empty_file = False

            for file_path in files:
                n_input_files += 1

                # 檔案存在但為 0 byte：來源端空檔。
                if file_path.stat().st_size == 0:
                    hour_has_empty_file = True
                    print(f"  [SOURCE EMPTY] {file_path}")
                    add_issue(
                        issue_rows,
                        day_str,
                        hh,
                        "SOURCE_EMPTY",
                        str(file_path),
                        "檔案存在但大小為 0 byte；視為來源端空檔，記錄後繼續。",
                    )
                    continue

                print(f"  [READ] {file_path}")

                n_bad_this_file = 0
                used_fallback = False

                try:
                    df = pl.read_csv(
                        file_path,
                        has_header=False,
                        schema=SCHEMA,
                        ignore_errors=False,
                        truncate_ragged_lines=False,
                        low_memory=True,
                    )
                    # 若先前版本留下 READ_ERROR，現在正常讀取成功就移除舊紀錄。
                    remove_issue(issue_rows, day_str, hh, "READ_ERROR", str(file_path))
                except pl.exceptions.NoDataError:
                    hour_has_empty_file = True
                    print(f"  [SOURCE EMPTY] {file_path}（無可讀資料）")
                    add_issue(
                        issue_rows,
                        day_str,
                        hh,
                        "SOURCE_EMPTY",
                        str(file_path),
                        "檔案非 0 byte，但 Polars 判定無可讀資料。",
                    )
                    continue
                except Exception as e:
                    # Polars 遇到 ragged / malformed row 時，改用嚴格 fallback：
                    # 只保留恰好 9 欄的 physical rows，其他全部隔離。
                    print(f"  [FALLBACK] Polars 讀取失敗，改用逐列 QC：{e}")
                    try:
                        df, n_bad_this_file = safe_fallback_read_csv(
                            file_path,
                            day_str,
                            hh,
                            malformed_rows,
                        )
                        used_fallback = True
                        remove_issue(issue_rows, day_str, hh, "READ_ERROR", str(file_path))

                        if n_bad_this_file > 0:
                            malformed_hours.add(hh)
                            n_malformed_source_rows += n_bad_this_file
                            add_issue(
                                issue_rows,
                                day_str,
                                hh,
                                "MALFORMED_ROWS",
                                str(file_path),
                                f"共 {n_bad_this_file} 個 physical rows 非 9 欄；"
                                "已隔離，不進入瓶頸資料；其餘正常 rows 照常保留。",
                            )
                            print(
                                f"  [QC OK] fallback 成功：保留 {df.height:,} 筆正常資料，"
                                f"隔離 {n_bad_this_file} 筆 malformed rows"
                            )
                    except Exception as fallback_e:
                        print(f"  [READ ERROR] fallback 仍失敗：{fallback_e}")
                        add_issue(
                            issue_rows,
                            day_str,
                            hh,
                            "READ_ERROR",
                            str(file_path),
                            f"Polars error={repr(e)}; fallback error={repr(fallback_e)}",
                        )
                        missing_hours.add(hh)
                        continue

                hour_has_readable_file = True

                info = pl.col("TripInformation")

                for key, meta in TARGETS.items():
                    code = meta["code"]
                    hit = df.filter(info.str.contains(code, literal=True))
                    if hit.height == 0:
                        continue

                    # 從 TripInformation 中抽出「通過該瓶頸門架的時間」。
                    time_pattern = (
                        rf"(\d{{4}}-\d{{2}}-\d{{2}} \d{{2}}:\d{{2}}:\d{{2}})"
                        rf"\+{re.escape(code)}"
                    )

                    hit = hit.with_columns(
                        pl.lit(meta["bottleneck"]).alias("Bottleneck"),
                        pl.lit(meta["direction"]).alias("Direction"),
                        pl.lit(code).alias("BottleneckGantry"),
                        pl.col("TripInformation")
                        .str.extract(time_pattern, group_index=1)
                        .alias("BottleneckTime"),
                        pl.lit(day_str).alias("SourceDate"),
                        pl.lit(hh).alias("SourceHour"),
                        pl.lit(file_path.name).alias("SourceFile"),
                    )

                    daily_parts[key].append(hit)

                if used_fallback and n_bad_this_file == 0:
                    # 理論上少見，但保留訊息方便 debug。
                    print("  [QC] fallback 啟動但沒有偵測到非 9 欄 physical row")

                del df

            if hour_has_readable_file:
                readable_hours.add(hh)
            elif hour_has_empty_file:
                source_empty_hours.add(hh)
                if not ALLOW_EXISTING_EMPTY_FILES:
                    unexpected_empty = True

        # ------------------------------------------------------------
        # 日期完整性判定
        # ------------------------------------------------------------
        if missing_hours:
            status = "INCOMPLETE_MISSING_OR_UNREADABLE_FILE"
            day_complete = False
        elif unexpected_empty:
            status = "INCOMPLETE_SOURCE_EMPTY"
            day_complete = False
        elif source_empty_hours and malformed_hours:
            status = "COMPLETE_WITH_SOURCE_EMPTY_AND_MALFORMED_ROWS"
            day_complete = True
        elif source_empty_hours:
            status = "COMPLETE_WITH_SOURCE_EMPTY"
            day_complete = True
        elif malformed_hours:
            status = "COMPLETE_WITH_MALFORMED_ROWS"
            day_complete = True
        else:
            status = "COMPLETE"
            day_complete = True

        empty_hours_text = ",".join(sorted(source_empty_hours))
        malformed_hours_text = ",".join(sorted(malformed_hours))
        missing_hours_text = ",".join(sorted(missing_hours))

        # ------------------------------------------------------------
        # 每個瓶頸方向寫出一天一個 parquet
        # ------------------------------------------------------------
        for key, meta in TARGETS.items():
            if daily_parts[key]:
                day_df = pl.concat(daily_parts[key], how="vertical_relaxed")
                n_trips = day_df.height
                n_vehicles = day_df.select(pl.col("VehicleID").n_unique()).item()
            else:
                day_df = empty_output_frame()
                n_trips = 0
                n_vehicles = 0

            if day_complete:
                out_path = output_file(key, day_str)
                day_df.write_parquet(out_path, compression="zstd")

            summary_rows[(day_str, key)] = {
                "SourceDate": day_str,
                "Target": key,
                "Label": meta["label"],
                "GantryCode": meta["code"],
                "NTrips": n_trips,
                "NUniqueVehicles": n_vehicles,
                "NExpectedHours": 24,
                "NReadableHours": len(readable_hours),
                "NInputFiles": n_input_files,
                "NSourceEmptyHours": len(source_empty_hours),
                "SourceEmptyHours": empty_hours_text,
                "NMalformedSourceRows": n_malformed_source_rows,
                "MalformedHours": malformed_hours_text,
                "NMissingHours": len(missing_hours),
                "MissingHours": missing_hours_text,
                "Status": status,
            }

            print(
                f"[DAY] {day_str} | {meta['label']} | "
                f"trips={n_trips:,} | unique vehicles={n_vehicles:,} | "
                f"readable_hours={len(readable_hours)} | "
                f"source_empty={empty_hours_text or '-'} | "
                f"malformed_rows={n_malformed_source_rows} "
                f"(hours={malformed_hours_text or '-'}) | "
                f"missing={missing_hours_text or '-'} | {status}"
            )

        # 每天即時更新紀錄，中途停止也不會丟失前面結果。
        write_csv_rows(
            summary_path,
            summary_rows,
            SUMMARY_FIELDS,
            sort_key=lambda r: (r["SourceDate"], r["Target"]),
        )
        write_csv_rows(
            issue_path,
            issue_rows,
            ISSUE_FIELDS,
            sort_key=lambda r: (
                r["SourceDate"],
                r["SourceHour"],
                r["IssueType"],
                r["FilePath"],
            ),
        )
        write_csv_rows(
            malformed_path,
            malformed_rows,
            MALFORMED_FIELDS,
            sort_key=lambda r: (
                r["SourceDate"],
                r["SourceHour"],
                r["FilePath"],
                int(r["LineNumber"]),
            ),
        )

        if day_complete:
            done_marker.write_text(
                f"status={status}\n"
                f"source_empty_hours={empty_hours_text}\n"
                f"malformed_source_rows={n_malformed_source_rows}\n"
                f"malformed_hours={malformed_hours_text}\n"
                f"completed_at={datetime.now().isoformat(timespec='seconds')}\n",
                encoding="utf-8",
            )

            print(f"[DONE] {day_str} 已完成並建立 done marker。")
        else:
            print(
                f"[WARN] {day_str} 尚未視為完成："
                f"source_empty={empty_hours_text or '-'}, "
                f"malformed={malformed_hours_text or '-'}, "
                f"missing/unreadable={missing_hours_text or '-'}。"
                "不建立 done marker，也不正式寫出該日 parquet。"
            )

    print("\n處理完成。")
    print(f"輸出位置：{OUTPUT_ROOT}")
    print(f"每日統計：{summary_path}")
    print(f"來源問題紀錄：{issue_path}")
    print(f"異常 physical rows：{malformed_path}")


if __name__ == "__main__":
    main()
