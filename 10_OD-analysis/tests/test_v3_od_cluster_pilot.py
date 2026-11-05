"""Focused tests for the V3 weekend/holiday OD feature engineering."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


MODULE = Path(__file__).resolve().parents[1] / "v3_od_cluster_pilot.py"
SPEC = importlib.util.spec_from_file_location("v3_od_cluster_pilot", MODULE)
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


class V3PilotFeatures(unittest.TestCase):
    def test_feature_engineering_2024_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pd.DataFrame([
                {"Year": 2024, "DayType": "WEEKEND", "days": 98},
                {"Year": 2024, "DayType": "LONGHOLIDAY", "days": 15},
                {"Year": 2025, "DayType": "WEEKEND", "days": 86},
                {"Year": 2025, "DayType": "LONGHOLIDAY", "days": 31},
            ]).to_csv(root / "00_calendar_days.csv", index=False)
            screen = pd.DataFrame([
                {"GantryO": "O1", "GantryD": "D1", "n_2024": 300,
                 "n_2024_weekend": 270, "n_2024_longholiday": 30,
                 "n_2025": 999999, "triplen_median_raw": 80,
                 "holiday_weekend_daily_ratio_2024": 1.0,
                 "vehicle_share_pct_31": 80, "vehicle_share_pct_32": 18,
                 "vehicle_share_pct_41": 1, "vehicle_share_pct_42": 1},
                {"GantryO": "O2", "GantryD": "D2", "n_2024": 200,
                 "n_2024_weekend": 200, "n_2024_longholiday": 0,
                 "n_2025": 500000, "triplen_median_raw": 120,
                 "holiday_weekend_daily_ratio_2024": 0,
                 "vehicle_share_pct_31": 70, "vehicle_share_pct_32": 28,
                 "vehicle_share_pct_41": 1, "vehicle_share_pct_42": 1},
            ])
            screen.to_csv(root / "08_od_feature_screen_2024.csv", index=False)
            hourly = pd.DataFrame([
                {"GantryO": "O1", "GantryD": "D1", "Year": year,
                 "DayType": dt, "hour": hour, "trips": trips}
                for year, dt, hours in [(2024, "WEEKEND", [270, 0]),
                                        (2024, "LONGHOLIDAY", [0, 30]),
                                        (2025, "WEEKEND", [999, 999])]
                for hour, trips in zip([0, 16], hours)
            ])
            hourly.to_csv(root / "04_od_hourly_profiles.csv", index=False)
            data, blocks, stats = mod.prepare(root, min_trips=300, alpha=10)
            self.assertEqual(stats["n_OD"], 1)
            self.assertEqual(stats["n_train_trips"], 300)
            self.assertTrue(all("2025" not in col for col in data))
            self.assertAlmostEqual(sum(data[c].iloc[0] for c in blocks["weekend"]), 1.0)
            self.assertGreater(data["holiday_minus_weekend_b4_share"].iloc[0], 0)
            features = [c for group in blocks.values() for c in group]
            self.assertTrue(np.isfinite(data[features].to_numpy()).all())


if __name__ == "__main__":
    unittest.main()
