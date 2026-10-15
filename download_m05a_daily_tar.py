# -*- coding: utf-8 -*-
"""
download_m05a_daily_tar.py
==========================

下載 2024-2025 TDCS M05A 每日 tar.gz，直接從壓縮檔中抽出四個研究方向，
保留完整 5 分鐘解析度，輸出每日 Parquet。

已由 2024-01-01 官方 M05A 原始檔確認的四個 target：
- 國3 大溪→龍潭 : 03F0559S -> 03F0648S
- 國3 龍潭→大溪 : 03F0648N -> 03F0559N
- 國5 坪林→頭城 : 05F0055S -> 05F0287S
- 國5 頭城→坪林 : 05F0287N -> 05F0055N

每個正常完整日，每個方向理論上約有：
288 個 5 分鐘時點 × 5 車種 = 1440 rows

執行：
    python download_m05a_daily_tar.py --test
    python download_m05a_daily_tar.py
    python download_m05a_daily_tar.py --start 2024-01-01 --end 2025-12-31

安裝：
    pip install requests pandas pyarrow tqdm
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import ssl
import tarfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.poolmanager import PoolManager


BASE_URL = "https://tisvcloud.freeway.gov.tw/history/TDCS/M05A"


@dataclass(frozen=True)
class Target:
    key: str
    route: str
    direction: str
    label: str
    gantry_from: str
    gantry_to: str


TARGETS = [
    Target(
        "N3_Daxi_Longtan_S",
        "N3_Daxi_Longtan",
        "S",
        "國3 大溪→龍潭",
        "03F0559S",
        "03F0648S",
    ),
    Target(
        "N3_Daxi_Longtan_N",
        "N3_Daxi_Longtan",
        "N",
        "國3 龍潭→大溪",
        "03F0648N",
        "03F0559N",
    ),
    Target(
        "N5_Pinglin_Toucheng_S",
        "N5_Pinglin_Toucheng",
        "S",
        "國5 坪林→頭城",
        "05F0055S",
        "05F0287S",
    ),
    Target(
        "N5_Pinglin_Toucheng_N",
        "N5_Pinglin_Toucheng",
        "N",
        "國5 頭城→坪林",
        "05F0287N",
        "05F0055N",
    ),
]

PAIR_MAP = {
    (t.gantry_from, t.gantry_to): t
    for t in TARGETS
}


class RelaxedStrictTLSAdapter(HTTPAdapter):
    """Python 3.13/3.14 相容：保留 TLS 驗證，只移除 VERIFY_X509_STRICT。"""

    def init_poolmanager(
        self,
        connections,
        maxsize,
        block=False,
        **pool_kwargs,
    ):
        context = ssl.create_default_context()

        if hasattr(ssl, "VERIFY_X509_STRICT"):
            context.verify_flags &= ~ssl.VERIFY_X509_STRICT

        self.poolmanager = PoolManager(
            num_pools=connections,
            maxsize=maxsize,
            block=block,
            ssl_context=context,
            **pool_kwargs,
        )


def make_session():
    s = requests.Session()
    s.mount("https://", RelaxedStrictTLSAdapter())
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/120 Safari/537.36"
        )
    })
    return s


def archive_url(date: pd.Timestamp) -> str:
    ymd = date.strftime("%Y%m%d")
    return f"{BASE_URL}/M05A_{ymd}.tar.gz"


def output_path(root: Path, target: Target, date: pd.Timestamp) -> Path:
    return (
        root
        / "M05A"
        / target.route
        / target.direction
        / date.strftime("%Y")
        / date.strftime("%m")
        / f"TDCS_M05A_{target.key}_{date.strftime('%Y%m%d')}.parquet"
    )


def done_path(root: Path, date: pd.Timestamp) -> Path:
    return root / "_done" / "M05A" / f"{date.strftime('%Y%m%d')}.done.json"


def log_path(root: Path, date: pd.Timestamp) -> Path:
    return root / "_logs" / "M05A" / f"{date.strftime('%Y%m%d')}.json"


def write_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)

    os.replace(tmp, path)


def atomic_parquet(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False, compression="zstd")
    os.replace(tmp, path)


def download_archive(
    session: requests.Session,
    date: pd.Timestamp,
    timeout: int,
    retries: int,
) -> bytes:
    url = archive_url(date)
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            r = session.get(
                url,
                timeout=(10, timeout),
                allow_redirects=True,
            )

            if r.status_code == 404:
                raise FileNotFoundError(f"404: {url}")

            r.raise_for_status()

            if not r.content:
                raise RuntimeError("HTTP 200 but archive is empty")

            return r.content

        except FileNotFoundError:
            raise

        except Exception as e:
            last_error = e

            if attempt < retries:
                time.sleep(
                    min(8.0, 0.8 * (2 ** (attempt - 1)))
                    + random.uniform(0, 0.3)
                )

    raise RuntimeError(
        f"Download failed after {retries} attempts: {url}\n{last_error}"
    )


def parse_archive(content: bytes, date: pd.Timestamp):
    rows_by_target = {t.key: [] for t in TARGETS}
    csv_count = 0
    malformed = 0

    with tarfile.open(
        fileobj=io.BytesIO(content),
        mode="r:gz",
    ) as tar:

        members = [
            m for m in tar.getmembers()
            if m.isfile() and m.name.lower().endswith(".csv")
        ]

        csv_count = len(members)

        for member in members:
            f = tar.extractfile(member)

            if f is None:
                continue

            text = io.TextIOWrapper(
                f,
                encoding="utf-8-sig",
                errors="replace",
                newline="",
            )

            reader = csv.reader(text)

            for row in reader:
                if len(row) < 6:
                    malformed += 1
                    continue

                timestamp = row[0].strip()
                g_from = row[1].strip()
                g_to = row[2].strip()
                vehicle_type = row[3].strip()
                speed = row[4].strip()
                volume = row[5].strip()

                target = PAIR_MAP.get((g_from, g_to))

                if target is None:
                    continue

                rows_by_target[target.key].append({
                    "TimeStamp": timestamp,
                    "GantryFrom": g_from,
                    "GantryTo": g_to,
                    "VehicleType": vehicle_type,
                    "Speed": speed,
                    "Volume": volume,
                    "TargetKey": target.key,
                    "Route": target.route,
                    "Direction": target.direction,
                    "Bottleneck": target.label,
                    "SourceDate": date.strftime("%Y-%m-%d"),
                    "SourceMember": member.name,
                })

    return rows_by_target, csv_count, malformed


def normalize_df(rows):
    columns = [
        "TimeStamp",
        "GantryFrom",
        "GantryTo",
        "VehicleType",
        "Speed",
        "Volume",
        "TargetKey",
        "Route",
        "Direction",
        "Bottleneck",
        "SourceDate",
        "SourceMember",
    ]

    if not rows:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(rows, columns=columns)

    df["TimeStamp"] = pd.to_datetime(
        df["TimeStamp"],
        errors="coerce",
    )

    df["VehicleType"] = df["VehicleType"].astype(str)

    df["Speed"] = pd.to_numeric(
        df["Speed"],
        errors="coerce",
    )

    df["Volume"] = pd.to_numeric(
        df["Volume"],
        errors="coerce",
    )

    df = df.sort_values(
        ["TimeStamp", "VehicleType"]
    ).reset_index(drop=True)

    return df


def process_day(
    session,
    date,
    root,
    timeout,
    retries,
    overwrite,
    require_complete,
):
    done = done_path(root, date)

    if done.exists() and not overwrite:
        return {
            "date": str(date.date()),
            "status": "SKIP_DONE",
        }

    content = download_archive(
        session=session,
        date=date,
        timeout=timeout,
        retries=retries,
    )

    rows_by_target, csv_count, malformed = parse_archive(
        content,
        date,
    )

    row_counts = {
        t.key: len(rows_by_target[t.key])
        for t in TARGETS
    }

    # 重要防呆：完全抓不到 target 時，絕不寫 done。
    if sum(row_counts.values()) == 0:
        raise RuntimeError(
            f"{date.date()} archive downloaded, "
            "but all four target row counts are 0."
        )

    # 完整日理論上有 288 CSV。
    # 若使用者要求完整檢查，任何方向不是 1440 rows 都標記異常。
    expected_csv = 288
    expected_rows_each = 1440

    completeness_ok = (
        csv_count == expected_csv
        and all(
            n == expected_rows_each
            for n in row_counts.values()
        )
    )

    # 先全部驗證，再寫檔，避免部分方向先被寫出。
    dfs = {}

    for t in TARGETS:
        dfs[t.key] = normalize_df(
            rows_by_target[t.key]
        )

    for t in TARGETS:
        atomic_parquet(
            dfs[t.key],
            output_path(root, t, date),
        )

    summary = {
        "date": str(date.date()),
        "archive_url": archive_url(date),
        "archive_bytes": len(content),
        "csv_count": csv_count,
        "malformed_rows": malformed,
        "saved_rows": row_counts,
        "expected_csv": expected_csv,
        "expected_rows_each": expected_rows_each,
        "completeness_ok": completeness_ok,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }

    if require_complete and not completeness_ok:
        summary["status"] = "INCOMPLETE_NOT_DONE"
        write_json(summary, log_path(root, date))
        return summary

    summary["status"] = (
        "DONE"
        if completeness_ok
        else "DONE_WITH_INCOMPLETE_SOURCE"
    )

    write_json(summary, log_path(root, date))
    write_json(summary, done)

    return summary


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--start",
        default="2024-01-01",
    )

    p.add_argument(
        "--end",
        default="2025-12-31",
    )

    p.add_argument(
        "--out",
        default=r"D:\TDCS_BOTTLENECK",
    )

    p.add_argument(
        "--timeout",
        type=int,
        default=90,
    )

    p.add_argument(
        "--retries",
        type=int,
        default=3,
    )

    p.add_argument(
        "--overwrite",
        action="store_true",
    )

    p.add_argument(
        "--test",
        action="store_true",
        help="只跑 start 指定的第一天",
    )

    p.add_argument(
        "--require-complete",
        action="store_true",
        help=(
            "若一天不是 288 個 CSV 或任一方向不是 1440 rows，"
            "則不標記 DONE"
        ),
    )

    return p.parse_args()


def main():
    args = parse_args()

    start = pd.Timestamp(args.start).normalize()
    end = pd.Timestamp(args.end).normalize()

    if args.test:
        dates = pd.DatetimeIndex([start])
    else:
        dates = pd.date_range(
            start,
            end,
            freq="D",
        )

    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("TDCS M05A DAILY TAR DOWNLOADER")
    print("=" * 80)
    print("Date range :", start.date(), "~", end.date())
    print("Days       :", len(dates))
    print("Output     :", root)
    print("Archive    : 1 tar.gz / day")
    print()

    print("Targets:")
    for t in TARGETS:
        print(
            f"  {t.label}: "
            f"{t.gantry_from} -> {t.gantry_to}"
        )

    print()

    session = make_session()
    results = []

    for date in tqdm(
        dates,
        desc="M05A days",
        unit="day",
    ):
        try:
            res = process_day(
                session=session,
                date=date,
                root=root,
                timeout=args.timeout,
                retries=args.retries,
                overwrite=args.overwrite,
                require_complete=args.require_complete,
            )

            results.append(res)

            if res.get("status") not in {
                "DONE",
                "SKIP_DONE",
            }:
                print(
                    f"\n[{date.date()}] "
                    f"{res.get('status')} "
                    f"{res.get('saved_rows')}"
                )

        except FileNotFoundError as e:
            print(f"\n[404] {date.date()}: {e}")

            results.append({
                "date": str(date.date()),
                "status": "404_SOURCE_NOT_FOUND",
            })

        except KeyboardInterrupt:
            print("\n[STOP] 中斷。下次重跑會跳過已完成日期。")
            break

        except Exception as e:
            print(f"\n[ERROR] {date.date()}: {e}")

            results.append({
                "date": str(date.date()),
                "status": "ERROR",
                "error": repr(e),
            })

    if results:
        summary = pd.DataFrame(results)

        summary_path = (
            root
            / "_logs"
            / "M05A"
            / "download_summary_all.csv"
        )

        summary_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        summary.to_csv(
            summary_path,
            index=False,
            encoding="utf-8-sig",
        )

        print("\nSummary:")
        print(
            summary["status"]
            .value_counts(dropna=False)
            .to_string()
        )

        print("\nSaved:")
        print(summary_path)


if __name__ == "__main__":
    main()
