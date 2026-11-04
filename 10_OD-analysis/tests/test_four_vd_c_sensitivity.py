"""Regression tests for the project-wide four-station Y target."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE = Path(__file__).resolve().parents[1] / "four_vd_c_sensitivity.py"
SPEC = importlib.util.spec_from_file_location("four_vd_c_sensitivity", SOURCE)
vd = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(vd)


class FourVdCTest(unittest.TestCase):
    def test_parquet_and_csv_ingestion(self):
        raw = pd.DataFrame({
            "Time5": ["2024-07-01 08:00:00"] * 4,
            "VDID": ["K69.510", "K65.990", "K63.990", "K61.670"],
            "Speed": [30, 80, 80, 80]
        })

        class Args:
            threads = 1
            memory_limit = "1GB"
            time_col = None
            site_col = None
            speed_col = None
            wide_speeds = None
            flow_col = None
            station_map = None
            missing_speed_values = []

        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            csv_path = folder / "vd.csv"
            parquet_path = folder / "vd.parquet"
            raw.to_csv(csv_path, index=False)
            import duckdb
            con = duckdb.connect()
            con.register("fixture", raw)
            con.execute("COPY fixture TO ? (FORMAT PARQUET)", [str(parquet_path)])
            con.close()
            for path in (csv_path, parquet_path):
                df = vd.read_vd(vd.files_from_input(str(path)), Args())
                wide = vd.prepare_wide(df, Args())
                self.assertEqual(len(wide), 1)
                self.assertEqual(wide.loc[pd.Timestamp("2024-07-01 08:00"), "69.510"], 30)

    def test_weights_and_labels_missing_not_free_flow(self):
        times = pd.date_range("2024-07-01 00:00:00", periods=4, freq="5min")
        speeds = pd.DataFrame({
            "69.510": [30.0, 80.0, 80.0, 80.0],
            "65.990": [80.0, 30.0, 30.0, 80.0],
            "63.990": [80.0, 80.0, 30.0, 80.0],
            "61.670": [80.0, 80.0, 80.0, np.nan],
        }, index=times)
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            vd.analyze(speeds, [0.25, 0.5, 0.75], 2024, out)
            table = pd.read_csv(out / "four_vd_y_timeseries.csv")
            yearly = pd.read_csv(out / "c_yearly_comparison.csv")
            self.assertAlmostEqual(table.C_weighted.iloc[0], 0.0417)
            self.assertAlmostEqual(table.C_weighted.iloc[1], 0.4600)
            self.assertAlmostEqual(table.C_weighted.iloc[2], 0.8200)
            self.assertTrue(np.isnan(table.C_weighted.iloc[3]))
            self.assertEqual(table.Y_c025.tolist()[:3], [1.0, 2.0, 2.0])
            self.assertEqual(table.Y_c050.tolist()[:3], [1.0, 1.0, 2.0])
            self.assertEqual(table.Y_c075.tolist()[:3], [1.0, 1.0, 2.0])
            self.assertTrue(np.isnan(table.Y_c050.iloc[3]))
            self.assertEqual(int(yearly.missing_bins.iloc[0]), 1)

    def test_long_and_missing_time_slot(self):
        source = pd.DataFrame({
            "time": ["2024-01-01 00:00:00"] * 4 + ["2024-01-01 00:10:00"] * 4,
            "site": ["K69.510", "K65.990", "K63.990", "K61.670"] * 2,
            "speed": [80, 80, 80, 80, 30, 30, 80, 80]
        })
        class Args:
            time_col = "time"
            site_col = "site"
            speed_col = "speed"
            wide_speeds = None
            flow_col = None
            station_map = None
            missing_speed_values = []
        wide = vd.prepare_wide(source, Args())
        self.assertEqual(len(wide), 3)
        self.assertTrue(wide.iloc[1].isna().all())
        self.assertEqual(wide.iloc[2]["65.990"], 30.0)


if __name__ == "__main__":
    unittest.main()
