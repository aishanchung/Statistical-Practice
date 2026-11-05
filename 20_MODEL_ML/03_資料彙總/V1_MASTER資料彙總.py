# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


# ============================================================
# 1. 路徑設定
# ============================================================

DATA_ROOT = Path(r"D:\統計實務\Data")

M05_ROOT = DATA_ROOT / "M05A"
M06_ROOT = DATA_ROOT / "M06B"

WEATHER_FILE = (
    DATA_ROOT
    / "Freeway_Weather_2024_2025.xlsx"
)

OUT_ROOT = (
    DATA_ROOT
    / "MASTER_2024_2025"
)

OUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. 分析期間
# ============================================================

START_DATE = pd.Timestamp("2024-01-01")
END_DATE = pd.Timestamp("2025-12-31")

# 用於真正時間過濾
END_EXCLUSIVE = pd.Timestamp("2026-01-01")


# ============================================================
# 3. 四個方向
# ============================================================

ROUTES = [
    ("N3_Daxi_Longtan", "N"),
    ("N3_Daxi_Longtan", "S"),
    ("N5_Pinglin_Toucheng", "N"),
    ("N5_Pinglin_Toucheng", "S"),
]


# 五種 ETC 車種
VEHICLE_TYPES = [
    "31",
    "32",
    "41",
    "42",
    "5",
]


# ============================================================
# 4. 工具函數
# ============================================================

def clean_vehicle_type(series):
    """
    31、31.0、'31' 全部統一成 '31'
    """
    return (
        series.astype("string")
        .str.strip()
        .str.replace(
            r"\.0$",
            "",
            regex=True
        )
    )


def normalize_string_columns(df):
    """
    避免不同日期 object dtype 不一致，
    導致 ParquetWriter schema 不一致。
    """
    for col in df.columns:

        if df[col].dtype == "object":
            df[col] = df[col].astype("string")

    return df


# ============================================================
# 5. 讀天氣
# ============================================================

def load_weather():

    print("\n讀取天氣資料：")
    print(WEATHER_FILE)

    sheets = pd.read_excel(
        WEATHER_FILE,
        sheet_name=None,
        engine="openpyxl"
    )

    # --------------------------------------------------------
    # Excel sheet -> 我們研究中的 route 名稱
    # --------------------------------------------------------

    SHEET_ROUTE_MAP = {
        "國道3號_大溪龍潭": "N3_Daxi_Longtan",
        "國道5號_坪林頭城": "N5_Pinglin_Toucheng",
    }

    required = [
        "DateTime",
        "Temperature_C",
        "Dewpoint_C",
        "Precipitation_mm",
        "WindSpeed_mps",
    ]

    parts = []

    for sheet_name, route in SHEET_ROUTE_MAP.items():

        if sheet_name not in sheets:

            raise ValueError(
                f"天氣 Excel 找不到 sheet：{sheet_name}"
            )

        df = sheets[sheet_name].copy()

        missing = [
            x
            for x in required
            if x not in df.columns
        ]

        if missing:

            raise ValueError(
                f"Sheet [{sheet_name}] "
                f"缺少欄位：{missing}"
            )

        print(
            f"  使用 sheet: {sheet_name}"
            f" -> {route}, "
            f"rows={len(df):,}"
        )

        # ----------------------------------------------------
        # 統一欄位名稱
        # ----------------------------------------------------

        df = df.rename(
            columns={
                "DateTime":
                    "WeatherTime",

                "Temperature_C":
                    "Weather_Temperature_C",

                "Dewpoint_C":
                    "Weather_Dewpoint_C",

                "Precipitation_mm":
                    "Weather_Precipitation_mm",

                "WindSpeed_mps":
                    "Weather_WindSpeed_mps",
            }
        )

        # ----------------------------------------------------
        # 加入 route
        # ----------------------------------------------------

        df["Bottleneck"] = route

        # ----------------------------------------------------
        # 時間
        # ----------------------------------------------------

        df["WeatherTime"] = pd.to_datetime(
            df["WeatherTime"],
            errors="coerce"
        )

        # ----------------------------------------------------
        # 數值欄位
        # ----------------------------------------------------

        weather_numeric = [
            "Weather_Temperature_C",
            "Weather_Dewpoint_C",
            "Weather_Precipitation_mm",
            "Weather_WindSpeed_mps",
        ]

        for col in weather_numeric:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        # ----------------------------------------------------
        # 留正式分析期間
        # ----------------------------------------------------

        df = df[
            (df["WeatherTime"] >= pd.Timestamp("2024-01-01"))
            &
            (df["WeatherTime"] < pd.Timestamp("2026-01-01"))
        ].copy()

        parts.append(df)

    # ========================================================
    # 合併兩個研究路段
    # ========================================================

    weather = pd.concat(
        parts,
        ignore_index=True
    )

    weather = (
        weather
        .dropna(
            subset=[
                "Bottleneck",
                "WeatherTime"
            ]
        )
        .drop_duplicates(
            subset=[
                "Bottleneck",
                "WeatherTime"
            ],
            keep="last"
        )
        .sort_values(
            [
                "Bottleneck",
                "WeatherTime"
            ]
        )
        .reset_index(drop=True)
    )

    # ========================================================
    # QA
    # ========================================================

    print("\n天氣資料整理完成")

    print(
        "總 rows:",
        f"{len(weather):,}"
    )

    print(
        "時間範圍:",
        weather["WeatherTime"].min(),
        "~",
        weather["WeatherTime"].max()
    )

    print("\n各路段筆數：")

    print(
        weather["Bottleneck"]
        .value_counts()
        .to_string()
    )

    # 2024 leap year + 2025 = 731 days
    expected_rows_each = 731 * 24

    for route in SHEET_ROUTE_MAP.values():

        n = (
            weather["Bottleneck"]
            == route
        ).sum()

        if n != expected_rows_each:

            print(
                f"[WARNING] {route}: "
                f"{n:,} rows，"
                f"預期 {expected_rows_each:,}"
            )

    return weather


# ============================================================
# 6. M05A 每日資料 -> 5分鐘 wide format
# ============================================================

def prepare_one_m05(file):

    df = pd.read_parquet(file)

    needed = [
        "TimeStamp",
        "GantryFrom",
        "GantryTo",
        "VehicleType",
        "Speed",
        "Volume",
    ]

    missing = [
        x
        for x in needed
        if x not in df.columns
    ]

    if missing:

        raise ValueError(
            f"{file}\n"
            f"M05A 缺欄位：{missing}"
        )

    df["Time5"] = pd.to_datetime(
        df["TimeStamp"],
        errors="coerce"
    )

    df["VehicleType"] = clean_vehicle_type(
        df["VehicleType"]
    )

    df["Speed"] = pd.to_numeric(
        df["Speed"],
        errors="coerce"
    )

    df["Volume"] = pd.to_numeric(
        df["Volume"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Speed wide
    # --------------------------------------------------------

    speed = df.pivot_table(
        index="Time5",
        columns="VehicleType",
        values="Speed",
        aggfunc="first"
    )

    # --------------------------------------------------------
    # Volume wide
    # --------------------------------------------------------

    volume = df.pivot_table(
        index="Time5",
        columns="VehicleType",
        values="Volume",
        aggfunc="sum"
    )

    time_index = pd.Index(
        sorted(
            df["Time5"]
            .dropna()
            .unique()
        ),
        name="Time5"
    )

    out = pd.DataFrame(
        index=time_index
    )

    # --------------------------------------------------------
    # 保留五種車速、五種流量
    # --------------------------------------------------------

    for vt in VEHICLE_TYPES:

        speed_col = f"M05_Speed_{vt}"
        volume_col = f"M05_Volume_{vt}"

        if vt in speed.columns:
            out[speed_col] = speed[vt]
        else:
            out[speed_col] = np.nan

        if vt in volume.columns:
            out[volume_col] = volume[vt]
        else:
            out[volume_col] = 0.0

    out = out.reset_index()

    # --------------------------------------------------------
    # Gantry pair
    # --------------------------------------------------------

    gantry_from = (
        df["GantryFrom"]
        .dropna()
        .astype(str)
        .iloc[0]
    )

    gantry_to = (
        df["GantryTo"]
        .dropna()
        .astype(str)
        .iloc[0]
    )

    out["M05_GantryFrom"] = gantry_from
    out["M05_GantryTo"] = gantry_to

    # --------------------------------------------------------
    # 全車種 Volume-weighted speed
    #
    # 注意：
    # M05A Speed 是各車種自己的速度統計，
    # 這裡是我們自行建立的 composite speed。
    # --------------------------------------------------------

    numerator = np.zeros(
        len(out),
        dtype=float
    )

    denominator = np.zeros(
        len(out),
        dtype=float
    )

    for vt in VEHICLE_TYPES:

        speed_col = f"M05_Speed_{vt}"
        volume_col = f"M05_Volume_{vt}"

        s = pd.to_numeric(
            out[speed_col],
            errors="coerce"
        )

        v = pd.to_numeric(
            out[volume_col],
            errors="coerce"
        ).fillna(0)

        # Volume=0 的 Speed=0
        # 不應被當作真正的 0 km/h
        valid = (
            s.notna()
            & (v > 0)
        )

        numerator += np.where(
            valid,
            s.fillna(0) * v,
            0
        )

        denominator += np.where(
            valid,
            v,
            0
        )

    out["M05_Speed_Weighted"] = np.where(
        denominator > 0,
        numerator / denominator,
        np.nan
    )

    out["M05_Volume_Total"] = denominator

    out["M05_Matched"] = True

    return out


# ============================================================
# 7. 先把某方向兩年的 M05A 合成
#
# 只有約 731 × 288 = 210,528 rows，
# 放記憶體非常合理。
# ============================================================

def build_m05_master(
    route,
    direction
):

    folder = (
        M05_ROOT
        / route
        / direction
    )

    files = sorted(
        folder.glob(
            "*/*/*.parquet"
        )
    )

    if not files:

        raise FileNotFoundError(
            f"找不到 M05A：{folder}"
        )

    print(
        f"\n建立 M05A master："
        f"{route} {direction}"
    )

    parts = []

    for file in tqdm(
        files,
        desc="M05A",
        leave=False
    ):

        part = prepare_one_m05(file)

        parts.append(part)

    m05 = pd.concat(
        parts,
        ignore_index=True
    )

    m05 = (
        m05
        .drop_duplicates(
            subset="Time5",
            keep="last"
        )
        .sort_values("Time5")
        .reset_index(drop=True)
    )

    print(
        f"M05A 5分鐘時點："
        f"{len(m05):,}"
    )

    return m05


# ============================================================
# 8. M06B 檔案路徑
# ============================================================

def get_m06_file(
    route,
    direction,
    date
):

    return (
        M06_ROOT
        / route
        / direction
        / date.strftime("%Y")
        / date.strftime("%m")
        / (
            f"{route}_"
            f"{direction}_"
            f"{date.strftime('%Y%m%d')}"
            f".parquet"
        )
    )


# ============================================================
# 9. 整理一天 M06B
# ============================================================

def prepare_m06_day(
    file,
    route,
    direction
):

    df = pd.read_parquet(file)

    # --------------------------------------------------------
    # Datetime
    # --------------------------------------------------------

    for col in [
        "DetectionTimeO",
        "DetectionTimeD",
        "BottleneckTime",
    ]:

        if col in df.columns:

            df[col] = pd.to_datetime(
                df[col],
                errors="coerce"
            )

    if "BottleneckTime" not in df.columns:

        raise ValueError(
            f"{file}\n"
            "沒有 BottleneckTime"
        )

    # --------------------------------------------------------
    # 限制正式研究期間
    # --------------------------------------------------------

    df = df[
        (df["BottleneckTime"] >= START_DATE)
        &
        (df["BottleneckTime"] < END_EXCLUSIVE)
    ].copy()

    if df.empty:
        return df

    # --------------------------------------------------------
    # M05A 的時間標籤代表剛結束的5分鐘
    #
    # 14:01 -> 14:05
    # 14:04 -> 14:05
    # 14:05 -> 14:05
    # --------------------------------------------------------

    df["Time5"] = (
        df["BottleneckTime"]
        .dt.ceil("5min")
    )

    # 2025/12/31 23:56~23:59
    # 會 ceil 成 2026/01/01 00:00
    # 不納入 2024-2025正式資料
    df = df[
        df["Time5"]
        < END_EXCLUSIVE
    ].copy()

    if "VehicleType" in df.columns:

        df["VehicleType"] = clean_vehicle_type(
            df["VehicleType"]
        )

    if "TripLength" in df.columns:

        df["TripLength"] = pd.to_numeric(
            df["TripLength"],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Weather 是 hourly
    # 已經是台灣時間
    # --------------------------------------------------------

    df["WeatherTime"] = (
        df["BottleneckTime"]
        .dt.floor("h")
    )

    # 明確記錄研究方向
    df["MasterRoute"] = route
    df["MasterDirection"] = direction

    return normalize_string_columns(df)


# ============================================================
# 10. 接 M05A
# ============================================================

def merge_m05(
    df,
    m05
):

    if df.empty:
        return df

    out = df.merge(
        m05,
        on="Time5",
        how="left",
        validate="many_to_one"
    )

    out["M05_Matched"] = (
        out["M05_Matched"]
        .fillna(False)
        .astype(bool)
    )

    # --------------------------------------------------------
    # 該筆 M06B 旅次「自己的車種速度」
    # --------------------------------------------------------

    out["M05_Speed_OwnType"] = np.nan
    out["M05_Volume_OwnType"] = np.nan

    if "VehicleType" in out.columns:

        for vt in VEHICLE_TYPES:

            mask = (
                out["VehicleType"]
                == vt
            )

            out.loc[
                mask,
                "M05_Speed_OwnType"
            ] = out.loc[
                mask,
                f"M05_Speed_{vt}"
            ]

            out.loc[
                mask,
                "M05_Volume_OwnType"
            ] = out.loc[
                mask,
                f"M05_Volume_{vt}"
            ]

    return out


# ============================================================
# 11. 接 Weather
# ============================================================

def merge_weather(
    df,
    weather,
    route
):

    if df.empty:
        return df

    w = weather[
        weather["Bottleneck"]
        == route
    ].copy()

    if w.empty:

        raise ValueError(
            f"天氣資料找不到 Bottleneck={route}"
        )

    # 已經知道 route，
    # 所以 merge 前不需要保留 Bottleneck 欄位
    w = w.drop(
        columns=["Bottleneck"]
    )

    w["Weather_Matched"] = True

    out = df.merge(
        w,
        on="WeatherTime",
        how="left",
        validate="many_to_one"
    )

    out["Weather_Matched"] = (
        out["Weather_Matched"]
        .fillna(False)
        .astype(bool)
    )

    return out


# ============================================================
# 12. 輸出一個方向
#
# 用 ParquetWriter 每天 append 一個 row group，
# 不把兩年所有 M06B 一次塞進 RAM。
# ============================================================

def build_one_direction(
    route,
    direction,
    weather
):

    print("\n")
    print("=" * 80)
    print(
        f"{route} {direction}"
    )
    print("=" * 80)

    # --------------------------------------------------------
    # M05A 約 21萬列，先整份放 RAM
    # --------------------------------------------------------

    m05 = build_m05_master(
        route,
        direction
    )

    output_file = (
        OUT_ROOT
        / (
            f"{route}_"
            f"{direction}_"
            f"2024_2025_MASTER.parquet"
        )
    )

    temp_file = output_file.with_suffix(
        ".tmp.parquet"
    )

    if temp_file.exists():
        temp_file.unlink()

    writer = None
    schema = None
    master_columns = None

    total_rows = 0
    total_m05_match = 0
    total_weather_match = 0

    dates = pd.date_range(
        START_DATE,
        END_DATE,
        freq="D"
    )

    try:

        for date in tqdm(
            dates,
            desc=f"{route}-{direction}",
            unit="day"
        ):

            file = get_m06_file(
                route,
                direction,
                date
            )

            if not file.exists():

                print(
                    f"\n[缺 M06B] {file}"
                )

                continue

            # ------------------------------------------------
            # M06B
            # ------------------------------------------------

            df = prepare_m06_day(
                file,
                route,
                direction
            )

            if df.empty:
                continue

            # ------------------------------------------------
            # + M05A
            # ------------------------------------------------

            df = merge_m05(
                df,
                m05
            )

            # ------------------------------------------------
            # + Weather
            # ------------------------------------------------

            df = merge_weather(
                df,
                weather,
                route
            )

            # ------------------------------------------------
            # Schema 穩定
            # ------------------------------------------------

            df = normalize_string_columns(
                df
            )

            if writer is None:

                master_columns = (
                    df.columns.tolist()
                )

                table = pa.Table.from_pandas(
                    df,
                    preserve_index=False
                )

                schema = table.schema

                writer = pq.ParquetWriter(
                    temp_file,
                    schema,
                    compression="zstd",
                    use_dictionary=True
                )

            else:

                # 若某天意外少欄
                for col in master_columns:

                    if col not in df.columns:
                        df[col] = pd.NA

                # 若某天突然多欄，先不偷偷加入 schema
                extra = [
                    x
                    for x in df.columns
                    if x not in master_columns
                ]

                if extra:

                    print(
                        "\n[注意] 額外欄位未寫入：",
                        extra
                    )

                df = df[
                    master_columns
                ]

                table = pa.Table.from_pandas(
                    df,
                    schema=schema,
                    preserve_index=False,
                    safe=False
                )

            # ------------------------------------------------
            # 寫入當天
            # ------------------------------------------------

            writer.write_table(
                table
            )

            total_rows += len(df)

            total_m05_match += int(
                df["M05_Matched"]
                .sum()
            )

            total_weather_match += int(
                df["Weather_Matched"]
                .sum()
            )

    finally:

        if writer is not None:
            writer.close()

    if writer is None:

        raise RuntimeError(
            f"{route} {direction} 沒有任何資料可輸出"
        )

    # --------------------------------------------------------
    # 全部成功才把 tmp 改成正式檔案
    # --------------------------------------------------------

    if output_file.exists():
        output_file.unlink()

    temp_file.replace(
        output_file
    )

    m05_rate = (
        total_m05_match
        / total_rows
        * 100
    )

    weather_rate = (
        total_weather_match
        / total_rows
        * 100
    )

    print("\n完成：")
    print(output_file)

    print(
        f"總旅次 rows       : "
        f"{total_rows:,}"
    )

    print(
        f"M05A match rate   : "
        f"{m05_rate:.3f}%"
    )

    print(
        f"Weather match rate: "
        f"{weather_rate:.3f}%"
    )


# ============================================================
# 13. MAIN
# ============================================================

def main():

    print("=" * 80)
    print("建立 2024-2025 四方向 Master Data")
    print("=" * 80)

    print("\nM05A:")
    print(M05_ROOT)

    print("\nM06B:")
    print(M06_ROOT)

    print("\nWeather:")
    print(WEATHER_FILE)

    print("\nOutput:")
    print(OUT_ROOT)

    # --------------------------------------------------------
    # Weather 一次讀好
    # --------------------------------------------------------

    weather = load_weather()

    # --------------------------------------------------------
    # 四方向
    # --------------------------------------------------------

    for route, direction in ROUTES:

        build_one_direction(
            route,
            direction,
            weather
        )

    print("\n")
    print("=" * 80)
    print("全部完成")
    print("=" * 80)

    print("\n最後應有四個檔案：")

    for route, direction in ROUTES:

        print(
            OUT_ROOT
            / (
                f"{route}_"
                f"{direction}_"
                f"2024_2025_MASTER.parquet"
            )
        )


if __name__ == "__main__":
    main()