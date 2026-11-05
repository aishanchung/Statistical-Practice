import pyarrow.parquet as pq
from pathlib import Path

root = Path(
    r"D:\統計實務\Data\MASTER_V1_M03A_2024_2025"
)

files = sorted(root.glob("*.parquet"))

for file in files:

    pf = pq.ParquetFile(file)

    print("=" * 100)
    print(file.name)
    print("Rows   :", f"{pf.metadata.num_rows:,}")
    print("Columns:", pf.metadata.num_columns)
    print()

    for i, field in enumerate(
        pf.schema_arrow,
        start=1
    ):
        print(
            f"{i:02d}. "
            f"{field.name:<35} "
            f"{field.type}"
        )