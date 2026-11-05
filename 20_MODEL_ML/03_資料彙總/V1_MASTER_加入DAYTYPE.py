from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


# ============================================================
# 1. 路徑
# ============================================================

DATA_ROOT = Path(r"D:\統計實務\Data")

MASTER_ROOT = DATA_ROOT / "MASTER_2024_2025"

OUT_ROOT = (
    DATA_ROOT
    / "MASTER_V1_2024_2025"
)

CALENDAR_FILE = (
    DATA_ROOT
    / "calendar_flags_2024_2025.csv"
)

HOLIDAY_FILE = (
    DATA_ROOT
    / "holiday_periods_2024_2025.csv"
)

START_DATE = pd.Timestamp("2024-01-01")
END_DATE = pd.Timestamp("2025-12-31")

BATCH_SIZE = 250000


# ============================================================
# 2. 建立 HolidayPhase
# ============================================================

def build_holiday_phase():

    hp = pd.read_csv(
        HOLIDAY_FILE,
        encoding="utf-8-sig"
    )

    hp["StartDate"] = pd.to_datetime(
        hp["StartDate"]
    )

    hp["EndDate"] = pd.to_datetime(
        hp["EndDate"]
    )

    rows = []

    for _, r in hp.iterrows():

        name = r["HolidayName"]

        start = r["StartDate"].normalize()
        end = r["EndDate"].normalize()

        total_days = (
            end - start
        ).days + 1

        dates = pd.date_range(
            start - pd.Timedelta(days=3),
            end + pd.Timedelta(days=2),
            freq="D"
        )

        for date in dates:

            if (
                date < START_DATE
                or date > END_DATE
            ):
                continue

            holiday_day_index = np.nan

            # -----------------------------
            # Holiday 前
            # -----------------------------

            if date < start:

                k = (
                    start - date
                ).days

                phase = f"D-{k}"
                phase_group = "PRE"

                priority = (
                    1,
                    k
                )

            # -----------------------------
            # Holiday 後
            # -----------------------------

            elif date > end:

                k = (
                    date - end
                ).days

                phase = f"D+{k}"
                phase_group = "POST"

                priority = (
                    2,
                    k
                )

            # -----------------------------
            # Holiday 內
            # -----------------------------

            else:

                idx = (
                    date - start
                ).days + 1

                holiday_day_index = idx

                phase_group = "HOLIDAY"

                if idx == 1:

                    phase = "Holiday_D1"

                elif date == end:

                    phase = "Holiday_Last"

                else:

                    phase = (
                        f"Holiday_D{idx}"
                    )

                priority = (
                    0,
                    idx
                )

            rows.append({

                "CalendarDate":
                    date,

                "HolidayEventName":
                    name,

                "HolidayEventStart":
                    start,

                "HolidayEventEnd":
                    end,

                "HolidayDaysTotal":
                    total_days,

                "HolidayDayIndex":
                    holiday_day_index,

                "HolidayPhase":
                    phase,

                "HolidayPhaseGroup":
                    phase_group,

                "IsHolidayWindow":
                    1,

                "_priority1":
                    priority[0],

                "_priority2":
                    priority[1],
            })

    phase = pd.DataFrame(
        rows
    )

    # 如果兩個假期 event window 重疊，
    # 優先使用真正 Holiday，其次 PRE，再 POST
    phase = (
        phase
        .sort_values(
            [
                "CalendarDate",
                "_priority1",
                "_priority2"
            ]
        )
        .drop_duplicates(
            subset="CalendarDate",
            keep="first"
        )
        .drop(
            columns=[
                "_priority1",
                "_priority2"
            ]
        )
    )

    return phase


# ============================================================
# 3. 整理 Calendar
# ============================================================

def load_calendar():

    calendar = pd.read_csv(
        CALENDAR_FILE,
        encoding="utf-8-sig"
    )

    calendar["CalendarDate"] = (
        pd.to_datetime(
            calendar["Date"]
        )
        .dt.normalize()
    )

    calendar = calendar.drop(
        columns="Date"
    )

    # Holiday lifecycle
    phase = build_holiday_phase()

    calendar = calendar.merge(
        phase,
        on="CalendarDate",
        how="left",
        validate="one_to_one"
    )

    # ========================================================
    # 一般日
    # ========================================================

    calendar[
        "HolidayPhase"
    ] = calendar[
        "HolidayPhase"
    ].fillna(
        "NORMAL"
    )

    calendar[
        "HolidayPhaseGroup"
    ] = calendar[
        "HolidayPhaseGroup"
    ].fillna(
        "NORMAL"
    )

    calendar[
        "IsHolidayWindow"
    ] = calendar[
        "IsHolidayWindow"
    ].fillna(
        0
    ).astype(int)

    # ========================================================
    # 基本日曆
    # ========================================================

    calendar["Year"] = (
        calendar["CalendarDate"]
        .dt.year
    )

    calendar["Month"] = (
        calendar["CalendarDate"]
        .dt.month
    )

    calendar["Day"] = (
        calendar["CalendarDate"]
        .dt.day
    )

    # ========================================================
    # Summer
    #
    # operational definition:
    # 07/01 ~ 08/31
    # ========================================================

    calendar["IsSummer"] = (
        calendar["Month"]
        .isin([7, 8])
        .astype(int)
    )

    print(
        "Calendar rows:",
        len(calendar)
    )

    print(
        calendar[
            "DayType"
        ].value_counts()
    )

    return calendar


# ============================================================
# 4. 一個 Master 加 Calendar
# ============================================================

def process_master(
    input_file,
    output_file,
    calendar
):

    print()
    print("=" * 80)
    print(input_file.name)
    print("=" * 80)

    parquet_file = (
        pq.ParquetFile(
            input_file
        )
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temp_file = (
        output_file
        .with_suffix(
            ".tmp.parquet"
        )
    )

    if temp_file.exists():

        temp_file.unlink()

    writer = None
    schema = None

    total_rows = 0
    matched_rows = 0

    try:

        batches = (
            parquet_file.iter_batches(
                batch_size=BATCH_SIZE
            )
        )

        for batch in tqdm(
            batches,
            desc=input_file.stem
        ):

            df = (
                batch.to_pandas()
            )

            # =================================================
            # 日期
            # =================================================

            df[
                "BottleneckTime"
            ] = pd.to_datetime(
                df[
                    "BottleneckTime"
                ],
                errors="coerce"
            )

            df[
                "CalendarDate"
            ] = (
                df[
                    "BottleneckTime"
                ]
                .dt.normalize()
            )

            # =================================================
            # Merge calendar
            # =================================================

            df = df.merge(
                calendar,
                on="CalendarDate",
                how="left",
                validate="many_to_one",
                indicator="_CalendarMerge"
            )

            df[
                "Calendar_Matched"
            ] = (
                df[
                    "_CalendarMerge"
                ]
                == "both"
            )

            df = df.drop(
                columns="_CalendarMerge"
            )

            # =================================================
            # Hour / Minute
            # =================================================

            df["Hour"] = (
                df[
                    "BottleneckTime"
                ]
                .dt.hour
            )

            df["Minute"] = (
                df[
                    "BottleneckTime"
                ]
                .dt.minute
            )

            # =================================================
            # QA
            # =================================================

            total_rows += len(
                df
            )

            matched_rows += int(
                df[
                    "Calendar_Matched"
                ].sum()
            )

            # =================================================
            # Write parquet
            # =================================================

            if writer is None:

                table = (
                    pa.Table
                    .from_pandas(
                        df,
                        preserve_index=False
                    )
                )

                schema = (
                    table.schema
                )

                writer = (
                    pq.ParquetWriter(
                        temp_file,
                        schema,
                        compression="zstd"
                    )
                )

            else:

                table = (
                    pa.Table
                    .from_pandas(
                        df,
                        schema=schema,
                        preserve_index=False,
                        safe=False
                    )
                )

            writer.write_table(
                table
            )

    finally:

        if writer is not None:

            writer.close()

    # ========================================================
    # Rename
    # ========================================================

    if output_file.exists():

        output_file.unlink()

    temp_file.replace(
        output_file
    )

    match_rate = (
        matched_rows
        / total_rows
    )

    print(
        "Rows:",
        f"{total_rows:,}"
    )

    print(
        "Calendar match rate:",
        f"{match_rate:.4%}"
    )

    print(
        "Saved:",
        output_file
    )


# ============================================================
# 5. MAIN
# ============================================================

def main():

    calendar = (
        load_calendar()
    )

    files = sorted(
        MASTER_ROOT.glob(
            "*_2024_2025_MASTER.parquet"
        )
    )

    print(
        "Master files:",
        len(files)
    )

    for file in files:

        output_file = (
            OUT_ROOT
            / (
                file.stem
                + "_V1.parquet"
            )
        )

        process_master(
            file,
            output_file,
            calendar
        )


if __name__ == "__main__":

    main()