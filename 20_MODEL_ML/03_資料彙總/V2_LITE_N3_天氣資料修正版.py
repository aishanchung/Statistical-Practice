from pathlib import Path

import duckdb
import pandas as pd


# ============================================================
# 0. 基本設定
# ============================================================

# ------------------------------------------------------------
# 舊 V2_LITE
# ------------------------------------------------------------

V2_PATH = Path(
    r"D:\統計實務\Data\V2_LITE"
    r"\N3_Daxi_Longtan_N_V2_LITE.parquet"
)


# ------------------------------------------------------------
# 天氣 Excel
#
# 等組員把缺少的 2023/12/31 16:00~23:00 UTC 補進去後，
# 改成她最後給你的檔案路徑。
# ------------------------------------------------------------

WEATHER_FILES = [
    Path(
        r"D:\統計實務\Data\原始檔案\Freeway_Weather_2024_2025.xlsx"
    ),

    # 如果組員另外給你一個補抓檔案，
    # 可以直接加第二個，不需要手動合併 Excel：
    #
     Path(
         r"D:\統計實務\Data\原始檔案\Freeway_Weather_2023_extra.xlsx"
     ),
]


# N3 大溪龍潭
WEATHER_SHEET = "國道3號_大溪龍潭"


# ------------------------------------------------------------
# 新輸出
#
# 不要直接覆蓋原 V2，先另存。
# QC 都確認沒問題後，再決定要不要把它當正式 V2_LITE。
# ------------------------------------------------------------

OUT_DIR = Path(
    r"D:\統計實務\Data\V2_LITE_WEATHER"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUT_PATH = (
    OUT_DIR
    / "N3_Daxi_Longtan_N_2024_2025_V2_LITE_WEATHER.parquet"
)

QC_PATH = (
    OUT_DIR
    / "N3_N_weather_remerge_QC.csv"
)

MISSING_HOUR_PATH = (
    OUT_DIR
    / "N3_N_weather_unmatched_hours.csv"
)


# ============================================================
# 1. V2 裡舊的四個天氣欄位
# ============================================================

OLD_WEATHER_COLS = [
    "Weather_Temperature_C",
    "Weather_Dewpoint_C",
    "Weather_Precipitation_mm",
    "Weather_WindSpeed_mps",
]


# ============================================================
# 2. 讀取天氣
# ============================================================

def load_weather():

    print("=" * 100)
    print("STEP 1：讀取天氣資料")
    print("=" * 100)

    all_weather = []

    for weather_file in WEATHER_FILES:

        if not weather_file.exists():

            raise FileNotFoundError(
                f"找不到天氣檔案：\n{weather_file}"
            )

        print()
        print(
            f"讀取：{weather_file}"
        )

        x = pd.read_excel(
            weather_file,
            sheet_name=WEATHER_SHEET,
            engine="openpyxl",
        )

        expected = [
            "DateTime",
            "Temperature_C",
            "Dewpoint_C",
            "Precipitation_mm",
            "WindSpeed_mps",
        ]

        missing = [
            col
            for col in expected
            if col not in x.columns
        ]

        if missing:

            raise ValueError(
                f"{weather_file.name} 缺少欄位：{missing}"
            )

        x = x[expected].copy()

        # ----------------------------------------------------
        # 原始 DateTime = UTC / 倫敦標準時間
        # 固定 +8 小時轉台灣時間
        #
        # 不使用 Europe/London，
        # 因為我們不要套用英國夏令時間。
        # ----------------------------------------------------

        x["DateTime_UTC"] = pd.to_datetime(
            x["DateTime"],
            errors="coerce",
        )

        if x["DateTime_UTC"].isna().any():

            bad = x.loc[
                x["DateTime_UTC"].isna(),
                "DateTime",
            ]

            raise ValueError(
                "天氣 DateTime 有無法解析的資料，例如：\n"
                f"{bad.head(10)}"
            )

        x["WeatherHourTW"] = (
            x["DateTime_UTC"]
            + pd.Timedelta(hours=8)
        )

        x["SourceFile"] = (
            weather_file.name
        )

        all_weather.append(x)

    weather = pd.concat(
        all_weather,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # 如果主檔和補抓檔案有重複 hour，
    # 這裡先檢查。
    # --------------------------------------------------------

    dup = (
        weather
        .duplicated(
            subset=["WeatherHourTW"],
            keep=False,
        )
    )

    if dup.any():

        print()
        print("注意：發現重複 WeatherHourTW。")

        dup_df = (
            weather.loc[
                dup,
                [
                    "WeatherHourTW",
                    "SourceFile",
                ],
            ]
            .sort_values(
                "WeatherHourTW"
            )
        )

        print(
            dup_df.head(20)
        )

        # 如果只是主檔 + 補抓檔重疊，
        # 留最後讀進來的版本。
        weather = (
            weather
            .sort_values(
                [
                    "WeatherHourTW",
                    "SourceFile",
                ]
            )
            .drop_duplicates(
                subset=["WeatherHourTW"],
                keep="last",
            )
            .reset_index(drop=True)
        )

    # --------------------------------------------------------
    # 最後只保留 merge 所需欄位
    # --------------------------------------------------------

    weather = weather[
        [
            "WeatherHourTW",
            "Temperature_C",
            "Dewpoint_C",
            "Precipitation_mm",
            "WindSpeed_mps",
        ]
    ].copy()

    weather = (
        weather
        .sort_values(
            "WeatherHourTW"
        )
        .reset_index(drop=True)
    )

    print()
    print(
        f"Weather rows = {len(weather):,}"
    )

    print(
        "Taiwan weather time range:"
    )

    print(
        weather["WeatherHourTW"].min(),
        "→",
        weather["WeatherHourTW"].max(),
    )

    print()

    print(
        weather.head(10)
    )

    return weather


# ============================================================
# 3. 檢查是否補齊研究期間需要的小時
# ============================================================

def check_weather_coverage(weather):

    print()
    print("=" * 100)
    print("STEP 2：檢查天氣時間完整性")
    print("=" * 100)

    # --------------------------------------------------------
    # V2 研究期間為台灣時間：
    # 2024-01-01 00:00
    # ~
    # 2025-12-31 23:00
    # --------------------------------------------------------

    expected_hours = pd.date_range(
        start="2024-01-01 00:00:00",
        end="2025-12-31 23:00:00",
        freq="h",
    )

    actual_hours = pd.DatetimeIndex(
        weather["WeatherHourTW"]
    )

    missing_hours = (
        expected_hours
        .difference(
            actual_hours
        )
    )

    print(
        f"Expected weather hours = {len(expected_hours):,}"
    )

    print(
        f"Missing weather hours  = {len(missing_hours):,}"
    )

    if len(missing_hours) > 0:

        print()
        print("尚缺的小時：")

        for t in missing_hours[:50]:
            print(t)

        if len(missing_hours) > 50:

            print(
                f"... 還有 {len(missing_hours) - 50} 個"
            )

        # ----------------------------------------------------
        # 我建議如果還缺資料就先停止，
        # 避免不小心產出一份仍有缺洞的正式 V2。
        # ----------------------------------------------------

        raise RuntimeError(
            "天氣資料尚未完整覆蓋 "
            "2024-01-01 00:00 ~ 2025-12-31 23:00 台灣時間。\n"
            "請先補齊再 merge。"
        )

    print()
    print(
        "OK：2024-2025 所有台灣整點天氣皆有資料。"
    )


# ============================================================
# 4. DuckDB 重新 merge
# ============================================================

def rebuild_v2(weather):

    print()
    print("=" * 100)
    print("STEP 3：移除舊 Weather 欄位 + 重新 merge")
    print("=" * 100)

    if not V2_PATH.exists():

        raise FileNotFoundError(
            f"找不到 V2：\n{V2_PATH}"
        )

    con = duckdb.connect()

    # --------------------------------------------------------
    # 小型 weather DataFrame 註冊到 DuckDB
    # --------------------------------------------------------

    con.register(
        "weather_df",
        weather,
    )

    # --------------------------------------------------------
    # 檢查 V2 欄位
    # --------------------------------------------------------

    schema = con.execute(
        f"""
        DESCRIBE
        SELECT *
        FROM read_parquet('{V2_PATH.as_posix()}')
        """
    ).fetchdf()

    v2_cols = (
        schema["column_name"]
        .tolist()
    )

    print()
    print(
        f"Original V2 columns = {len(v2_cols)}"
    )

    # --------------------------------------------------------
    # 確認 BottleneckTime
    # --------------------------------------------------------

    if "BottleneckTime" not in v2_cols:

        raise ValueError(
            "V2 找不到 BottleneckTime，無法重新對天氣。"
        )

    # --------------------------------------------------------
    # 舊 weather 欄位哪些真的存在
    # --------------------------------------------------------

    old_weather_present = [
        c
        for c in OLD_WEATHER_COLS
        if c in v2_cols
    ]

    print(
        "舊 Weather 欄位："
    )

    for c in old_weather_present:
        print(
            "  -",
            c,
        )

    # --------------------------------------------------------
    # DuckDB EXCLUDE
    # --------------------------------------------------------

    if old_weather_present:

        exclude_sql = (
            " EXCLUDE ("
            + ", ".join(
                f'"{c}"'
                for c in old_weather_present
            )
            + ")"
        )

    else:

        exclude_sql = ""

    # --------------------------------------------------------
    # 原始 row count
    # --------------------------------------------------------

    n_before = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet(
            '{V2_PATH.as_posix()}'
        )
        """
    ).fetchone()[0]

    print()
    print(
        f"Rows before = {n_before:,}"
    )

    # --------------------------------------------------------
    # 重新 JOIN
    #
    # BottleneckTime = 台灣時間。
    #
    # 例如：
    # 2024-01-01 17:35:27
    #
    # 對應：
    # 2024-01-01 17:00 的 hourly weather
    # --------------------------------------------------------

    query = f"""
    SELECT

        v.*{exclude_sql},

        w.Temperature_C
            AS Weather_Temperature_C,

        w.Dewpoint_C
            AS Weather_Dewpoint_C,

        w.Precipitation_mm
            AS Weather_Precipitation_mm,

        w.WindSpeed_mps
            AS Weather_WindSpeed_mps

    FROM read_parquet(
        '{V2_PATH.as_posix()}'
    ) AS v

    LEFT JOIN weather_df AS w

        ON date_trunc(
            'hour',
            v.BottleneckTime
        )
        =
        w.WeatherHourTW
    """

    # --------------------------------------------------------
    # 直接輸出 parquet
    # --------------------------------------------------------

    print()
    print(
        "開始寫入新 V2..."
    )

    con.execute(
        f"""
        COPY (
            {query}
        )
        TO '{OUT_PATH.as_posix()}'
        (
            FORMAT PARQUET,
            COMPRESSION ZSTD
        )
        """
    )

    print()
    print(
        f"Output:\n{OUT_PATH}"
    )

    # ========================================================
    # 5. QC
    # ========================================================

    print()
    print("=" * 100)
    print("STEP 4：Merge QC")
    print("=" * 100)

    n_after = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet(
            '{OUT_PATH.as_posix()}'
        )
        """
    ).fetchone()[0]

    missing_weather = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet(
            '{OUT_PATH.as_posix()}'
        )
        WHERE
            Weather_Temperature_C IS NULL
            OR Weather_Dewpoint_C IS NULL
            OR Weather_Precipitation_mm IS NULL
            OR Weather_WindSpeed_mps IS NULL
        """
    ).fetchone()[0]

    # --------------------------------------------------------
    # 特別分開：
    #
    # 1. BottleneckTime 本身 NULL
    # 2. BottleneckTime 有值但沒有 weather
    # --------------------------------------------------------

    null_bottleneck_time = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet(
            '{OUT_PATH.as_posix()}'
        )
        WHERE BottleneckTime IS NULL
        """
    ).fetchone()[0]

    unmatched_non_null_time = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet(
            '{OUT_PATH.as_posix()}'
        )
        WHERE
            BottleneckTime IS NOT NULL
            AND (
                Weather_Temperature_C IS NULL
                OR Weather_Dewpoint_C IS NULL
                OR Weather_Precipitation_mm IS NULL
                OR Weather_WindSpeed_mps IS NULL
            )
        """
    ).fetchone()[0]

    print(
        f"Rows before                  = {n_before:,}"
    )

    print(
        f"Rows after                   = {n_after:,}"
    )

    print(
        f"Row count difference         = {n_after - n_before:,}"
    )

    print(
        f"BottleneckTime NULL          = {null_bottleneck_time:,}"
    )

    print(
        f"Weather missing total        = {missing_weather:,}"
    )

    print(
        f"Weather unmatched non-null t = {unmatched_non_null_time:,}"
    )

    # --------------------------------------------------------
    # Row count 必須完全一樣
    # --------------------------------------------------------

    if n_before != n_after:

        raise RuntimeError(
            "ERROR：重新 merge 後 row 數改變！\n"
            "這通常代表 weather hour 有重複造成 one-to-many join。"
        )

    # --------------------------------------------------------
    # 把 unmatched hour 列出來
    # --------------------------------------------------------

    unmatched_hours = con.execute(
        f"""
        SELECT

            date_trunc(
                'hour',
                BottleneckTime
            ) AS BottleneckHourTW,

            COUNT(*) AS TripRows

        FROM read_parquet(
            '{OUT_PATH.as_posix()}'
        )

        WHERE
            BottleneckTime IS NOT NULL
            AND (
                Weather_Temperature_C IS NULL
                OR Weather_Dewpoint_C IS NULL
                OR Weather_Precipitation_mm IS NULL
                OR Weather_WindSpeed_mps IS NULL
            )

        GROUP BY 1
        ORDER BY 1
        """
    ).fetchdf()

    unmatched_hours.to_csv(
        MISSING_HOUR_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # QC summary
    # --------------------------------------------------------

    qc = pd.DataFrame(
        [
            {
                "Metric":
                    "Rows_before",

                "Value":
                    n_before,
            },

            {
                "Metric":
                    "Rows_after",

                "Value":
                    n_after,
            },

            {
                "Metric":
                    "Row_difference",

                "Value":
                    n_after - n_before,
            },

            {
                "Metric":
                    "BottleneckTime_NULL",

                "Value":
                    null_bottleneck_time,
            },

            {
                "Metric":
                    "Weather_missing_total",

                "Value":
                    missing_weather,
            },

            {
                "Metric":
                    "Weather_unmatched_non_null_time",

                "Value":
                    unmatched_non_null_time,
            },

            {
                "Metric":
                    "Weather_TW_min",

                "Value":
                    str(
                        weather[
                            "WeatherHourTW"
                        ].min()
                    ),
            },

            {
                "Metric":
                    "Weather_TW_max",

                "Value":
                    str(
                        weather[
                            "WeatherHourTW"
                        ].max()
                    ),
            },
        ]
    )

    qc.to_csv(
        QC_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        f"QC:\n{QC_PATH}"
    )

    print()
    print(
        f"Unmatched hours:\n{MISSING_HOUR_PATH}"
    )

    # --------------------------------------------------------
    # 最終欄位檢查
    # --------------------------------------------------------

    final_schema = con.execute(
        f"""
        DESCRIBE
        SELECT *
        FROM read_parquet(
            '{OUT_PATH.as_posix()}'
        )
        """
    ).fetchdf()

    print()
    print(
        f"Final columns = {len(final_schema)}"
    )

    print()

    print(
        final_schema[
            [
                "column_name",
                "column_type",
            ]
        ].to_string(
            index=False
        )
    )

    con.close()


# ============================================================
# 6. Main
# ============================================================

def main():

    print("=" * 100)
    print(
        "N3 N V2_LITE：修正 UTC 天氣時間並重新合併"
    )
    print("=" * 100)

    weather = load_weather()

    check_weather_coverage(
        weather
    )

    rebuild_v2(
        weather
    )

    print()
    print("=" * 100)
    print("ALL DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()