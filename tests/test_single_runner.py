import csv
import tempfile
import unittest
from pathlib import Path

from test_single import RESULT_FIELDS, write_metrics_csv


class SingleRunnerTest(unittest.TestCase):
    def test_metrics_csv_includes_evaluation_mode(self):
        result = {
            "model": "cnn",
            "variant": "baseline",
            "mode": "baseline",
            "occlusion_ratio_percent": 0.0,
            "mAP_percent": 75.0,
            "rank1_percent": 80.0,
            "rank5_percent": 90.0,
            "checkpoint": "outputs/cnn_baseline_final.pth",
        }

        with tempfile.TemporaryDirectory() as directory:
            metrics_path = Path(directory) / "metrics.csv"
            write_metrics_csv(metrics_path, [result])

            with metrics_path.open(
                "r", newline="", encoding="utf-8"
            ) as metrics_file:
                reader = csv.DictReader(metrics_file)
                self.assertEqual(reader.fieldnames, list(RESULT_FIELDS))
                rows = list(reader)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["mode"], "baseline")
        self.assertEqual(rows[0]["occlusion_ratio_percent"], "0.0")


if __name__ == "__main__":
    unittest.main()
