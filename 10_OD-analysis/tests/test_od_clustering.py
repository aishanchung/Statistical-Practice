"""Small, generated-data checks. No Google Drive data is stored in Git."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "od_clustering.py"
spec = importlib.util.spec_from_file_location("od_clustering", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Tests(unittest.TestCase):
    def test_sparse_weights_l1_l2(self):
        w = module.sparse_weights(np.array([10.0, 5.0, 2.0, 1.0]), 1.4)
        self.assertTrue(np.all(w >= 0))
        self.assertAlmostEqual(np.linalg.norm(w), 1.0, places=5)
        self.assertLessEqual(w.sum(), 1.40001)
        self.assertEqual(w[-1], 0.0)

    def test_missing_interior_part(self):
        with tempfile.TemporaryDirectory() as td:
            Path(td, "part_0000.parquet").touch()
            Path(td, "part_0002.parquet").touch()
            with self.assertRaisesRegex(ValueError, "Missing Parquet part"):
                module.discover_files(td)

    def test_end_to_end_counts_and_labels(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rows = []
            for year in [2024, 2025]:
                for od in range(16):
                    for n in range(22 if year == 2024 else 6):
                        t = datetime(year, 1, 1) + timedelta(hours=4 * ((n + od) % 6), days=n % 14)
                        rows.append({"GantryO": f"O{od}", "GantryD": f"D{od}",
                                     "BottleneckTime": t, "DetectionTimeD": t + timedelta(hours=1),
                                     "DayType": "WEEKEND" if (n + od) % 5 == 0 else "WEEKDAY",
                                     "VehicleType": "31" if od % 2 == 0 else "42",
                                     "TripLength": float(40 + od * 5)})
            df = pd.DataFrame(rows)
            con = duckdb.connect()
            con.register("synthetic", df)
            filename = root / "part_0000.parquet"
            con.execute(f"COPY synthetic TO '{filename.as_posix()}' (FORMAT PARQUET)")
            con.close()
            out = root / "results"
            command = [sys.executable, str(SCRIPT), "--input", str(root), "--output", str(out),
                       "--min-trips", "10", "--k-min", "2", "--k-max", "2",
                       "--methods", "kmeans", "sparse", "--feature-sets", "time",
                       "--restarts", "1", "--five-minute", "both", "--threads", "1", "--memory-limit", "1GB"]
            proc = subprocess.run(command, capture_output=True, text=True, timeout=90)
            self.assertEqual(proc.returncode, 0, proc.stdout + "\n" + proc.stderr)
            count = pd.read_csv(out / "od_yearly_counts.csv")
            labels = pd.read_csv(out / "od_cluster_labels.csv")
            hist = pd.read_csv(out / "five_min_bottleneck.csv")
            future = pd.read_csv(out / "five_min_completed.csv")
            self.assertEqual(int(count.n_train.sum()), 16 * 22)
            self.assertEqual(int(count.n_test.sum()), 16 * 6)
            self.assertEqual(len(labels), 16)
            self.assertEqual(int(hist.trips_total.sum()), 16 * 28)
            self.assertEqual(int(future.trips_total.sum()), 16 * 28)
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["candidate"]["k"], 2)
            self.assertTrue((out / "sparse_feature_weights_all.csv").exists())


if __name__ == "__main__":
    unittest.main()
