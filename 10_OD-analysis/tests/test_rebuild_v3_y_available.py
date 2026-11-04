"""Ensure reusing existing V3 outputs never rescans or overwrites OD/Y."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import duckdb
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
spec = importlib.util.spec_from_file_location("rebuild_v3_y_available", BASE / "rebuild_v3_y_available.py")
util = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(util)


def write_parquet(file: Path, data: pd.DataFrame) -> None:
    con = duckdb.connect()
    con.register("data", data)
    con.execute("COPY data TO ? (FORMAT PARQUET)", [str(file)])
    con.close()


class RebuildTests(unittest.TestCase):
    def test_rebuild_keeps_original_y_and_handles_single_station(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            src = root / "old"
            src.mkdir()
            times = pd.date_range("2024-01-06 14:00", periods=3, freq="5min")
            from build_v3_lite_holiday import SITES
            rows = []
            for idx, time in enumerate(times):
                for n, (vdid, (km, weight)) in enumerate(SITES.items()):
                    observed = idx == 0 or (idx == 1 and n == 0)
                    rows.append({"Time5": time, "Date": time.normalize(), "DayType": "WEEKEND",
                                 "Year": 2024, "VDID": vdid, "SiteKM": km, "weight": weight,
                                 "MissingStatus": "VALID" if observed else "INVALID_SPEED",
                                 "Speed": (35.0 if idx == 0 else 80.0) if observed else -1.0,
                                 "Occupancy": 1.0, "VDType": "OBSERVED"})
            write_parquet(src / "V3_LITE_VD_4SITES.parquet", pd.DataFrame(rows))
            old = pd.DataFrame({"Time5": times, "od_trips_5min": [5, 3, 0],
                                "Year": 2024, "DayType": "WEEKEND", "Y": [2, pd.NA, pd.NA]})
            write_parquet(src / "V3_Y_5MIN.parquet", old)
            pd.DataFrame({"Date": [pd.Timestamp("2024-01-06")], "calendar_source": ["OD_MODAL"],
                          "daytype_conflict": [False], "od_zero_records": [False]}).to_csv(
                              src / "V3_calendar_all_dates.csv", index=False)
            output = root / "new"
            util.main(["--existing-v3", str(src), "--output", str(output), "--c", "0.5"])

            con = duckdb.connect()
            new_y = con.execute("SELECT * FROM read_parquet(?) ORDER BY Time5",
                                [str(output / "V3_Y_5MIN.parquet")]).df()
            original_y = con.execute("SELECT * FROM read_parquet(?) ORDER BY Time5",
                                     [str(src / "V3_Y_5MIN.parquet")]).df()
            con.close()
            self.assertEqual(original_y.Y.tolist()[0], 2)
            self.assertTrue(pd.isna(original_y.Y.iloc[1]))
            self.assertEqual(new_y.Y.iloc[0], 2)
            self.assertEqual(new_y.Y.iloc[1], 0)  # single valid VD, free-flow
            self.assertTrue(pd.isna(new_y.Y.iloc[2]))  # four missing
            self.assertEqual(new_y.od_trips_5min.tolist(), [5, 3, 0])
            self.assertAlmostEqual(new_y.weight_coverage.iloc[1], 0.0417)
            daily = pd.read_csv(output / "V3_focus_date_coverage.csv")
            self.assertEqual(int(daily.Y_valid_bins.iloc[0]), 2)
            self.assertEqual(int(daily.Y_missing_bins.iloc[0]), 1)


if __name__ == "__main__":
    unittest.main()
