from pathlib import Path
import csv
from collections import Counter

ROOT = Path(r"D:\M06B\20250228")
EXPECTED_COLS = 9

for hour in range(24):
    hh = f"{hour:02d}"
    file_path = ROOT / hh / f"TDCS_M06B_20250228_{hh}0000.csv"

    if not file_path.exists():
        print(f"{hh}: FILE NOT FOUND")
        continue

    if file_path.stat().st_size == 0:
        print(f"{hh}: EMPTY")
        continue

    counts = Counter()
    bad_rows = []

    with file_path.open(
        "r",
        encoding="utf-8-sig",
        errors="replace",
        newline=""
    ) as f:

        reader = csv.reader(f)

        for line_no, row in enumerate(reader, start=1):
            n = len(row)
            counts[n] += 1

            if n != EXPECTED_COLS and len(bad_rows) < 5:
                bad_rows.append((line_no, n, row))

    print("\n" + "=" * 80)
    print(f"HOUR {hh}")
    print(f"欄位數分布: {dict(counts)}")

    if not bad_rows:
        print("全部都是 9 欄")
    else:
        print("異常列範例：")

        for line_no, n, row in bad_rows:
            print(f"\nLine {line_no}: {n} fields")

            for i, value in enumerate(row, start=1):
                preview = value[:150]
                print(f"  [{i}] {preview}")