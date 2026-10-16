# -*- coding: utf-8 -*-

"""
build_n3_master_v1_lite.py

用途
----
從：
D:\\統計實務\\Data\\MASTER_V1_M03A_2024_2025

只處理國道 3 號大溪－龍潭：
1. 北向 N
2. 南向 S

每個方向只保留指定的 15 個欄位，
並重新建立只有四分類的 DayType。

每個方向輸出：
1. Parquet
2. CSV

原始 MASTER_V1_M03A 完全不修改。
"""

from pathlib import Path
import time
import duckdb


# ============================================================
# 1. 路徑設定
# ============================================================

DATA_ROOT = Path(r"D:\統計實務\Data")

INPUT_ROOT = (
    DATA_ROOT
    / "MASTER_V1_M03A_2024_2025"
)

OUTPUT_ROOT = (
    DATA_ROOT
    / "MASTER_V1_N3_LITE_2024_2025"
)

TEMP_ROOT = (
    OUTPUT_ROOT
    / "_duckdb_temp"
)


# ============================================================
# 2. 國道 3 號兩個方向
# ============================================================

TARGETS = {

    "N3_Daxi_Longtan_N": {

        "input_file":
            "N3_Daxi_Longtan_N_2024_2025_MASTER_V1_M03A.parquet",

        "output_stem":
            "N3_Daxi_Longtan_N_2024_2025_MASTER_V1_LITE",
    },

    "N3_Daxi_Longtan_S": {

        "input_file":
            "N3_Daxi_Longtan_S_2024_2025_MASTER_V1_M03A.parquet",

        "output_stem":
            "N3_Daxi_Longtan_S_2024_2025_MASTER_V1_LITE",
    },
}


# ============================================================
# 3. 執行設定
# ============================================================

# 如果之前已經輸出過，是否重新覆蓋
FORCE_REBUILD = True

# 你要求兩種都留
WRITE_PARQUET = True
WRITE_CSV = True


# ============================================================
# 4. 最終保留欄位
# ============================================================

FINAL_COLUMNS = [

    "VehicleID",
    "VehicleType",

    "DetectionTimeO",
    "GantryO",

    "DetectionTimeD",
    "GantryD",

    "TripLength",

    "TripInformation",

    "BottleneckTime",

    "Weather_Temperature_C",
    "Weather_Dewpoint_C",
    "Weather_Precipitation_mm",
    "Weather_WindSpeed_mps",

    "WeekdayNum",

    "DayType",
]


# DayType 重建時需要，
# 但最後不會留在輸出資料中。
DAYTYPE_SOURCE_COLUMNS = [

    "IsLongHoliday",
    "IsStandaloneHoliday",
    "IsWeekend",
]


# ============================================================
# 5. Helper
# ============================================================

def ensure_dirs():

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    TEMP_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )


def sql_path(path: Path):

    return (
        path.as_posix()
        .replace("'", "''")
    )


def make_connection():

    con = duckdb.connect(
        database=":memory:"
    )

    con.execute(
        f"""
        SET temp_directory=
        '{sql_path(TEMP_ROOT)}'
        """
    )

    con.execute(
        "SET preserve_insertion_order=false"
    )

    return con


def size_gb(path: Path):

    if not path.exists():
        return None

    return (
        path.stat().st_size
        / 1024
        / 1024
        / 1024
    )


# ============================================================
# 6. 檢查來源欄位
# ============================================================

def check_schema(
    con,
    input_path,
):

    source_cols = [

        row[0]

        for row in con.execute(
            f"""
            DESCRIBE

            SELECT *

            FROM read_parquet(
                '{sql_path(input_path)}'
            )
            """
        ).fetchall()
    ]


    # DayType 最後會重算，所以不要求原本 DayType 一定存在。
    required = [

        x
        for x in FINAL_COLUMNS
        if x != "DayType"

    ] + DAYTYPE_SOURCE_COLUMNS


    missing = [

        x
        for x in required
        if x not in source_cols
    ]


    if missing:

        print(
            "\nERROR：來源檔缺少必要欄位"
        )

        for x in missing:

            print(
                " -",
                x
            )

        raise RuntimeError(
            "Schema check failed."
        )


    print(
        "Schema check: PASS"
    )


# ============================================================
# 7. 精簡版 SELECT
# ============================================================

def make_select_sql(
    input_path,
):

    return f"""

        SELECT

            VehicleID,

            VehicleType,


            DetectionTimeO,

            GantryO,


            DetectionTimeD,

            GantryD,


            TripLength,


            TripInformation,


            BottleneckTime,


            Weather_Temperature_C,

            Weather_Dewpoint_C,

            Weather_Precipitation_mm,

            Weather_WindSpeed_mps,


            WeekdayNum,


            CASE

                WHEN
                    COALESCE(
                        IsLongHoliday,
                        0
                    ) = 1

                THEN
                    'LONGHOLIDAY'


                WHEN
                    COALESCE(
                        IsStandaloneHoliday,
                        0
                    ) = 1

                THEN
                    'STANDALONEHOLIDAY'


                WHEN
                    COALESCE(
                        IsWeekend,
                        0
                    ) = 1

                THEN
                    'WEEKEND'


                ELSE
                    'WEEKDAY'

            END
            AS DayType


        FROM read_parquet(
            '{sql_path(input_path)}'
        )

    """


# ============================================================
# 8. 處理單一方向
# ============================================================

def process_one(
    route_key,
    cfg,
):

    input_path = (
        INPUT_ROOT
        /
        cfg["input_file"]
    )


    parquet_out = (
        OUTPUT_ROOT
        /
        f'{cfg["output_stem"]}.parquet'
    )


    csv_out = (
        OUTPUT_ROOT
        /
        f'{cfg["output_stem"]}.csv'
    )


    print("\n")
    print("=" * 100)

    print(
        "PROCESS:",
        route_key
    )

    print("=" * 100)

    print(
        "Input:",
        input_path
    )


    # --------------------------------------------------------
    # 檢查來源
    # --------------------------------------------------------

    if not input_path.exists():

        raise FileNotFoundError(
            input_path
        )


    con = make_connection()


    check_schema(
        con,
        input_path,
    )


    # --------------------------------------------------------
    # 原始 row 數
    # --------------------------------------------------------

    input_rows = con.execute(
        f"""

        SELECT COUNT(*)

        FROM read_parquet(
            '{sql_path(input_path)}'
        )

        """
    ).fetchone()[0]


    print(
        f"Input rows:"
        f" {input_rows:,}"
    )


    print(
        f"Input size:"
        f" {size_gb(input_path):.3f} GB"
    )


    select_sql = (
        make_select_sql(
            input_path
        )
    )


    # ========================================================
    # 9. Parquet
    # ========================================================

    if WRITE_PARQUET:


        if (
            parquet_out.exists()
            and FORCE_REBUILD
        ):

            parquet_out.unlink()


        if not parquet_out.exists():

            print(
                "\nWriting Parquet..."
            )


            t0 = time.time()


            con.execute(
                f"""

                COPY (

                    {select_sql}

                )

                TO
                '{sql_path(parquet_out)}'

                (

                    FORMAT PARQUET,

                    COMPRESSION ZSTD,

                    ROW_GROUP_SIZE 250000

                )

                """
            )


            elapsed = (
                time.time()
                -
                t0
            )


            print(
                f"Parquet finished:"
                f" {elapsed / 60:.1f} min"
            )


        else:

            print(
                "\nParquet already exists."
            )


    # ========================================================
    # 10. Parquet QC
    # ========================================================

    if parquet_out.exists():


        output_rows = con.execute(
            f"""

            SELECT COUNT(*)

            FROM read_parquet(
                '{sql_path(parquet_out)}'
            )

            """
        ).fetchone()[0]


        output_cols = [

            row[0]

            for row in con.execute(
                f"""

                DESCRIBE

                SELECT *

                FROM read_parquet(
                    '{sql_path(parquet_out)}'
                )

                """
            ).fetchall()
        ]


        print(
            f"\nOutput rows:"
            f" {output_rows:,}"
        )


        print(
            f"Output columns:"
            f" {len(output_cols)}"
        )


        print(
            f"Parquet size:"
            f" {size_gb(parquet_out):.3f} GB"
        )


        # row count 必須完全一致
        if output_rows != input_rows:

            raise RuntimeError(

                f"Row count changed: "
                f"{input_rows:,}"
                f" -> "
                f"{output_rows:,}"

            )


        # 欄位必須完全一致
        if output_cols != FINAL_COLUMNS:

            print(
                "\nExpected:"
            )

            print(
                FINAL_COLUMNS
            )

            print(
                "\nActual:"
            )

            print(
                output_cols
            )


            raise RuntimeError(
                "Output schema mismatch."
            )


        print(
            "Row count QC: PASS"
        )

        print(
            "Column QC   : PASS"
        )


    # ========================================================
    # 11. CSV
    # ========================================================

    if WRITE_CSV:


        if (
            csv_out.exists()
            and FORCE_REBUILD
        ):

            csv_out.unlink()


        if not csv_out.exists():


            print(
                "\nWriting CSV..."
            )


            print(
                "CSV 有數千萬 rows，"
                "檔案會比 Parquet 大很多，"
                "這是正常的。"
            )


            t0 = time.time()


            # 已經有精簡 Parquet，
            # CSV 直接從精簡 Parquet 轉，
            # 不再掃一次 78 欄的大檔。
            con.execute(
                f"""

                COPY (

                    SELECT *

                    FROM read_parquet(
                        '{sql_path(parquet_out)}'
                    )

                )

                TO
                '{sql_path(csv_out)}'

                (

                    FORMAT CSV,

                    HEADER,

                    DELIMITER ','

                )

                """
            )


            elapsed = (
                time.time()
                -
                t0
            )


            print(
                f"CSV finished:"
                f" {elapsed / 60:.1f} min"
            )


            print(
                f"CSV size:"
                f" {size_gb(csv_out):.3f} GB"
            )


        else:

            print(
                "\nCSV already exists."
            )


    # ========================================================
    # 12. DayType QC
    # ========================================================

    if parquet_out.exists():


        day_qc = con.execute(
            f"""

            SELECT

                DayType,

                COUNT(*) AS Rows,

                ROUND(

                    100.0
                    *
                    COUNT(*)
                    /
                    SUM(
                        COUNT(*)
                    )
                    OVER (),

                    4

                )
                AS Pct

            FROM read_parquet(
                '{sql_path(parquet_out)}'
            )

            GROUP BY
                DayType

            ORDER BY
                Rows DESC

            """
        ).fetchdf()


        print(
            "\nDayType distribution:"
        )


        print(
            day_qc.to_string(
                index=False
            )
        )


        # ----------------------------------------------------
        # DayType 只能有四種
        # ----------------------------------------------------

        actual_types = set(
            day_qc[
                "DayType"
            ].tolist()
        )


        allowed_types = {

            "WEEKDAY",
            "WEEKEND",
            "LONGHOLIDAY",
            "STANDALONEHOLIDAY",

        }


        unknown_types = (
            actual_types
            -
            allowed_types
        )


        if unknown_types:

            raise RuntimeError(

                "Unexpected DayType: "
                + str(
                    unknown_types
                )

            )


        print(
            "DayType QC: PASS"
        )


    con.close()


    print(
        "\nParquet:"
    )

    print(
        parquet_out
    )


    print(
        "\nCSV:"
    )

    print(
        csv_out
    )


# ============================================================
# 13. Main
# ============================================================

def main():


    ensure_dirs()


    print("=" * 100)

    print(
        "BUILD N3 MASTER V1 LITE"
    )

    print("=" * 100)


    print(
        "Input root:"
    )

    print(
        INPUT_ROOT
    )


    print(
        "\nOutput root:"
    )

    print(
        OUTPUT_ROOT
    )


    print(
        "\nFinal columns:"
    )


    for i, col in enumerate(
        FINAL_COLUMNS,
        start=1,
    ):

        print(
            f"{i:02d}. {col}"
        )


    # --------------------------------------------------------
    # N3 北向 + 南向
    # --------------------------------------------------------

    for route_key, cfg in (
        TARGETS.items()
    ):

        process_one(
            route_key,
            cfg,
        )


    print("\n")
    print("=" * 100)

    print(
        "COMPLETE"
    )

    print("=" * 100)


    print(
        "\n輸出資料夾："
    )

    print(
        OUTPUT_ROOT
    )


if __name__ == "__main__":

    main()