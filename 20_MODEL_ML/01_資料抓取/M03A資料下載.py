# -*- coding: utf-8 -*-

import os
import csv
import time
import tarfile
import shutil
import requests
import ssl
import requests

from urllib3.poolmanager import PoolManager
from datetime import datetime, timedelta
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ============================================================
# 1. 基本設定
# ============================================================

START_DATE = "2024-01-01"
END_DATE   = "2025-12-31"

BASE_URL = "https://tisvcloud.freeway.gov.tw/history/TDCS/M03A/"

# 改成你自己的資料夾
OUTPUT_DIR = r"D:\統計實務\Data\M03A"

# 原始 tar.gz
RAW_DIR = os.path.join(OUTPUT_DIR, "raw")

# 解壓縮後的 CSV
EXTRACT_DIR = os.path.join(OUTPUT_DIR, "extracted")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(EXTRACT_DIR, exist_ok=True)


# ============================================================
# 2. 控制參數
# ============================================================

# 解壓後是否刪除 tar.gz
DELETE_TAR_AFTER_EXTRACT = False

# 每一天下載完休息幾秒
SLEEP_SECONDS = 0.5

# timeout
TIMEOUT = 180


# ============================================================
# 3. 建立可重試的 requests session
# ============================================================
# ============================================================
# SSL Adapter
# ============================================================

class RelaxedStrictTLSAdapter(HTTPAdapter):
    """
    保留 TLS 憑證驗證，
    但關閉 VERIFY_X509_STRICT。

    用來處理 TISVCloud 在新版 Python / OpenSSL
    可能出現的：
    Missing Subject Key Identifier
    """

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


# ============================================================
# 建立 Session
# ============================================================

def create_session():

    session = requests.Session()

    session.mount(
        "https://",
        RelaxedStrictTLSAdapter()
    )

    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/120 Safari/537.36"
        )
    })

    return session


# ============================================================
# 4. 產生日期
# ============================================================

def generate_dates(start_date, end_date):

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end   = datetime.strptime(end_date, "%Y-%m-%d")

    current = start

    while current <= end:

        yield current

        current += timedelta(days=1)


# ============================================================
# 5. 下載單日 tar.gz
# ============================================================

def download_one_day(session, date_obj):

    ymd = date_obj.strftime("%Y%m%d")
    yyyy = date_obj.strftime("%Y")
    mm = date_obj.strftime("%m")

    filename = f"M03A_{ymd}.tar.gz"

    url = BASE_URL + filename

    # -----------------------------
    # 原始檔依 年/月 分資料夾
    # -----------------------------

    raw_month_dir = os.path.join(
        RAW_DIR,
        yyyy,
        mm
    )

    os.makedirs(
        raw_month_dir,
        exist_ok=True
    )

    tar_path = os.path.join(
        raw_month_dir,
        filename
    )

    # -----------------------------
    # 解壓資料夾
    # -----------------------------

    day_extract_dir = os.path.join(
        EXTRACT_DIR,
        yyyy,
        mm,
        ymd
    )

    os.makedirs(
        day_extract_dir,
        exist_ok=True
    )

    done_file = os.path.join(
        day_extract_dir,
        "_DONE.txt"
    )

    # ========================================================
    # 已經完整處理過 → 跳過
    # ========================================================

    if os.path.exists(done_file):

        print(
            f"[SKIP] {ymd} 已完成"
        )

        return True, "already_done"


    # ========================================================
    # Step 1：下載
    # ========================================================

    try:

        if not os.path.exists(tar_path):

            print(
                f"[DOWNLOAD] {ymd}"
            )

            print(
                url
            )

            response = session.get(
                url,
                stream=True,
                timeout=TIMEOUT
            )

            # 網站真的沒有這天
            if response.status_code == 404:

                print(
                    f"[404] {ymd} 無檔案"
                )

                return False, "404_not_found"

            response.raise_for_status()

            # 先存成暫存檔
            temp_path = tar_path + ".part"

            with open(
                temp_path,
                "wb"
            ) as f:

                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):

                    if chunk:

                        f.write(chunk)

            # 完整下載後才改名
            os.replace(
                temp_path,
                tar_path
            )

        else:

            print(
                f"[FOUND] {ymd} tar.gz 已存在"
            )


        # ====================================================
        # Step 2：解壓縮
        # ====================================================

        print(
            f"[EXTRACT] {ymd}"
        )

        with tarfile.open(
            tar_path,
            "r:gz"
        ) as tar:

            tar.extractall(
                day_extract_dir
            )


        # ====================================================
        # Step 3：建立完成標記
        # ====================================================

        with open(
            done_file,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(
                f"{ymd} completed\n"
            )


        # ====================================================
        # Step 4：選擇是否刪除 tar.gz
        # ====================================================

        if DELETE_TAR_AFTER_EXTRACT:

            os.remove(
                tar_path
            )

            print(
                f"[DELETE TAR] {ymd}"
            )


        print(
            f"[DONE] {ymd}"
        )

        return True, "success"


    except Exception as e:

        print(
            f"[ERROR] {ymd}"
        )

        print(
            e
        )

        return False, str(e)


# ============================================================
# 6. 主程式
# ============================================================

def main():

    session = create_session()

    dates = list(
        generate_dates(
            START_DATE,
            END_DATE
        )
    )

    total = len(dates)

    print("=" * 60)

    print(
        f"下載期間：{START_DATE} ~ {END_DATE}"
    )

    print(
        f"共 {total} 天"
    )

    print(
        f"輸出位置：{OUTPUT_DIR}"
    )

    print("=" * 60)


    failed = []


    for i, date_obj in enumerate(
        dates,
        start=1
    ):

        ymd = date_obj.strftime(
            "%Y%m%d"
        )

        print()
        print("=" * 60)

        print(
            f"進度 {i}/{total}"
        )

        print(
            f"日期 {ymd}"
        )

        print("=" * 60)


        success, message = download_one_day(
            session,
            date_obj
        )


        if not success:

            failed.append({
                "Date": ymd,
                "Message": message
            })


        time.sleep(
            SLEEP_SECONDS
        )


    # ========================================================
    # 7. 儲存失敗清單
    # ========================================================

    failed_file = os.path.join(
        OUTPUT_DIR,
        "failed_dates.csv"
    )


    with open(
        failed_file,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Date",
                "Message"
            ]
        )

        writer.writeheader()

        writer.writerows(
            failed
        )


    print()
    print("=" * 60)

    print(
        "全部處理完成"
    )

    print(
        f"失敗日期：{len(failed)}"
    )

    print(
        f"失敗紀錄：{failed_file}"
    )

    print("=" * 60)


# ============================================================
# 8. 執行
# ============================================================

if __name__ == "__main__":

    main()