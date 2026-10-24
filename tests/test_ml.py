import csv
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scoring.ml_model import ARTIFACT_PATH, LogisticModel
from scoring.models import get_model
from scoring.service import score_application, validate_request
from training.data_hash import dataset_sha256
from training.generate_synthetic import generate_file


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "synthetic_v1"
SHOP = {
    "application_id": "shop-1",
    "monthly_revenue": 2_400_000,
    "monthly_expenses": 1_500_000,
    "existing_debt": 500_000,
    "monthly_debt_payments": 90_000,
    "overdue_count": 0,
    "requested_amount": 1_500_000,
    "term_months": 12,
}


class SyntheticMLTests(unittest.TestCase):
    def test_generated_rows_match_public_request_and_train_test_are_separate(self):
        train_ids = set()
        for split, expected_count in (("train", 20_000), ("test", 5_000)):
            with (DATA_DIR / f"{split}.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), expected_count)
            self.assertEqual({int(row["default_12m"]) for row in rows}, {0, 1})
            for row in rows:
                request = {key: int(value) if key != "application_id" else value
                           for key, value in row.items() if key != "default_12m"}
                validate_request(request)
            ids = {row["application_id"] for row in rows}
            self.assertEqual(len(ids), expected_count)
            if split == "train":
                train_ids = ids
            else:
                self.assertFalse(train_ids & ids)

    def test_generation_is_reproducible(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            first = Path(directory) / "first.csv"
            second = Path(directory) / "second.csv"
            self.assertEqual(generate_file(first, 25, 19, "test"), generate_file(second, 25, 19, "test"))
            self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_artifact_tracks_exact_training_files(self):
        artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
        for split in ("train", "test"):
            digest = dataset_sha256(DATA_DIR / f"{split}.csv")
            self.assertEqual(artifact["training"][f"{split}_sha256"], digest)

    def test_dataset_hash_ignores_checkout_line_endings(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            lf = Path(directory) / "lf.csv"
            crlf = Path(directory) / "crlf.csv"
            lf.write_bytes(b"a,b\n1,2\n")
            crlf.write_bytes(b"a,b\r\n1,2\r\n")
            self.assertEqual(dataset_sha256(lf), dataset_sha256(crlf))

    def test_ml_score_responds_to_financial_risk(self):
        model = LogisticModel()
        baseline = score_application(SHOP, model)
        overdue = score_application({**SHOP, "overdue_count": 2}, model)
        negative_cash_flow = score_application({**SHOP, "monthly_expenses": 2_500_000}, model)
        self.assertEqual(baseline["score"], 97)
        self.assertEqual(overdue["score"], 64)
        self.assertEqual(negative_cash_flow["score"], 1)
        self.assertEqual((baseline["decision"], overdue["decision"], negative_cash_flow["decision"]),
                         ("approved", "counter_offer", "rejected"))
        self.assertLess(negative_cash_flow["score"], overdue["score"])
        self.assertLess(overdue["score"], baseline["score"])
        self.assertEqual(baseline, score_application(SHOP, model))

    def test_configured_model_preserves_public_response_shape(self):
        with patch.dict(os.environ, {"SCORING_MODEL": "logreg_v1"}):
            get_model.cache_clear()
            try:
                result = score_application(SHOP)
            finally:
                get_model.cache_clear()
        self.assertEqual(result["score"], 97)
        self.assertEqual(set(result), {"decision", "score", "approved_amount", "reasons", "factors"})


if __name__ == "__main__":
    unittest.main()
