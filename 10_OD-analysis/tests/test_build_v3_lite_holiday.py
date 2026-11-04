"""Synthetic integration tests; no downloaded OD/VD source data is required."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import duckdb
import pandas as pd


MODULE = Path(__file__).resolve().parents[1] / "build_v3_lite_holiday.py"
spec = importlib.util.spec_from_file_location("v3_holiday", MODULE)
mod = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(mod)


def parquet(path: Path, frame: pd.DataFrame) -> None:
    con = duckdb.connect()
    con.register("sample", frame)
    con.execute(f"COPY sample TO '{path.as_posix()}' (FORMAT PARQUET)")
    con.close()


class HolidayV3Tests(unittest.TestCase):
    def test_weekend_without_od_and_missing_site_classes(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            od_dir = root / "od"
            od_dir.mkdir()
            od = pd.DataFrame({
                "BottleneckTime": pd.to_datetime(["2024-01-05 14:01", "2024-01-07 14:03",
                                                    "2024-01-07 14:07", "2024-01-07 23:17",
                                                    "2024-01-08 12:01", "2024-01-09 18:01"]),
                "DayType": ["LONGHOLIDAY", "WEEKEND", "WEEKDAY", "WEEKEND",
                            "STANDALONEHOLIDAY", "WEEKDAY"],
                "GantryO": ["a"] * 6, "GantryD": ["b"] * 6,
                "VehicleType": ["31"] * 6,
                "VD_61p000_Speed": [90.0] * 6,
                "VD_61p670_Speed": [90.0] * 6,
            })
            parquet(od_dir / "part_0000.parquet", od)

            data = []
            for date in ["2024-01-05", "2024-01-06", "2024-01-07"]:
                for station in mod.SITES:
                    if date == "2024-01-07" and "63.990" in station:
                        continue  # absent row
                    speed = -1.0 if date == "2024-01-05" and "61.670" in station else 35.0
                    data.append({"TargetTime": pd.Timestamp(date + " 14:00"), "VDID": station,
                                 "Speed": speed, "Occupancy": 0.5, "VDType": "OBSERVED"})
                data.append({"TargetTime": pd.Timestamp(date + " 14:00"),
                             "VDID": "VD-N3-N-61.000-M-LOOP", "Speed": 90.0,
                             "Occupancy": 0.5, "VDType": "OBSERVED"})
            vd_path = root / "vd.parquet"
            parquet(vd_path, pd.DataFrame(data))
            out = root / "results"
            mod.main(["--od-input", str(od_dir), "--vd-input", str(vd_path),
                      "--output", str(out), "--start-date", "2024-01-05",
                      "--end-date", "2024-01-09"])

            cal = pd.read_csv(out / "V3_calendar_all_dates.csv")
            focus = cal.loc[cal.is_focus_date]
            self.assertEqual(len(focus), 3)
            self.assertEqual(focus.loc[focus.Date.eq("2024-01-06"), "calendar_source"].item(),
                             "WEEKDAY_INFERRED_NO_OD")
            self.assertTrue(cal.loc[cal.Date.eq("2024-01-07"), "daytype_conflict"].item())
            self.assertFalse(cal.loc[cal.Date.eq("2024-01-08"), "is_focus_date"].item())

            con = duckdb.connect()
            vd = con.execute("SELECT * FROM read_parquet(?)", [str(out / "V3_LITE_VD_4SITES.parquet")]).df()
            yy = con.execute("SELECT * FROM read_parquet(?)", [str(out / "V3_Y_5MIN.parquet")]).df()
            od_v3 = con.execute("SELECT * FROM read_parquet(?)", [str(out / "V3_LITE_OD_TRIPS.parquet")]).df()
            con.close()
            self.assertEqual(len(vd), 3 * 288 * 4)
            self.assertEqual(len(yy), 3 * 288)
            self.assertEqual(len(od_v3), 4)  # One Friday + three Sunday; Sat has no OD
            self.assertNotIn("VD_61p000_Speed", od_v3.columns)
            self.assertNotIn("VD_61p000_Occupancy", od_v3.columns if "VD_61p000_Occupancy" in od_v3 else [])
            for table in (od_v3, vd, yy):
                self.assertNotIn("is_focus_window", table.columns)
            self.assertTrue(od_v3.BottleneckTime.eq(pd.Timestamp("2024-01-07 23:17")).any())
            self.assertFalse(vd.VDID.str.contains("61.000").any())
            self.assertEqual(vd.loc[(vd.Time5.eq(pd.Timestamp("2024-01-05 14:00"))) &
                                    vd.SiteKM.eq("61.670"), "MissingStatus"].item(), "INVALID_SPEED")
            self.assertEqual(vd.loc[(vd.Time5.eq(pd.Timestamp("2024-01-07 14:00"))) &
                                    vd.SiteKM.eq("63.990"), "MissingStatus"].item(), "ROW_ABSENT")
            self.assertTrue(pd.isna(yy.loc[yy.Time5.eq(pd.Timestamp("2024-01-05 14:00")), "Y"].item()))
            self.assertEqual(yy.loc[yy.Time5.eq(pd.Timestamp("2024-01-06 14:00")), "Y"].item(), 2)
            day = pd.read_csv(out / "VD_missing_by_date_station.csv")
            jan7 = day[(day.Date.eq("2024-01-07")) & day.SiteKM.eq(63.990) &
                       day.scope.eq("full_day")]
            self.assertEqual(int(jan7.expected_bins.item()), 288)
            self.assertEqual(int(jan7.row_absent_bins.item()), 288)
            self.assertEqual(set(day.scope), {"full_day"})
            summary = pd.read_csv(out / "VD_missing_summary.csv")
            self.assertEqual(set(summary.scope), {"full_day"})
            daily = pd.read_csv(out / "V3_focus_date_coverage.csv")
            self.assertEqual(len(daily), 3)
            self.assertEqual(set(daily.scope), {"full_day"})
            self.assertTrue((out / "VD_missing_runs_joint.csv").exists())

    def test_calendar_override_discovers_unseen_weekday_holiday(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            od = pd.DataFrame({"BottleneckTime": pd.to_datetime(["2024-01-05 12:00"]),
                               "DayType": ["WEEKDAY"], "GantryO": ["a"],
                               "GantryD": ["b"], "VehicleType": ["31"]})
            vd = pd.DataFrame({"TargetTime": [pd.Timestamp("2024-01-06 00:00")],
                               "VDID": [list(mod.SITES)[0]], "Speed": [80.0],
                               "Occupancy": [1.0], "VDType": ["OBSERVED"]})
            parquet(root / "od.parquet", od)
            parquet(root / "vd.parquet", vd)
            pd.DataFrame([{"Date": "2024-01-05", "DayType": "LONGHOLIDAY"}]).to_csv(root / "calendar.csv", index=False)
            out = root / "out"
            mod.main(["--od-input", str(root / "od.parquet"),
                      "--vd-input", str(root / "vd.parquet"), "--output", str(out),
                      "--start-date", "2024-01-05", "--end-date", "2024-01-06",
                      "--calendar-csv", str(root / "calendar.csv"), "--audit-only"])
            cal = pd.read_csv(out / "V3_calendar_focus_dates.csv")
            self.assertEqual(len(cal), 2)
            self.assertEqual(cal.loc[cal.Date.eq("2024-01-05"), "calendar_source"].item(), "CALENDAR_CSV")
            self.assertFalse((out / "V3_LITE_OD_TRIPS.parquet").exists())


if __name__ == "__main__":
    unittest.main()
