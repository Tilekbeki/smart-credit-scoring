import json
from pathlib import Path
import unittest


class ContractTests(unittest.TestCase):
    def test_schema_is_valid_json_with_expected_fields(self):
        schema = json.loads(Path("contracts/scoring.schema.json").read_text())
        self.assertEqual(schema["additionalProperties"], False)
        self.assertIn("requested_amount", schema["required"])
        self.assertIn("term_months", schema["required"])
        response = json.loads(Path("contracts/scoring-response.schema.json").read_text())
        self.assertEqual(response["properties"]["decision"]["enum"], ["approved", "counter_offer", "rejected"])
        self.assertIn("factors", response["required"])


if __name__ == "__main__":
    unittest.main()
