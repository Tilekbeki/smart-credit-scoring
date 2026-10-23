import json
from pathlib import Path
import unittest

from scoring.service import ScoringFeatures


ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = ROOT / "contracts"


class ContractTests(unittest.TestCase):
    def test_files_match_public_shape(self):
        request = json.loads((CONTRACTS / "scoring.schema.json").read_text(encoding="utf-8-sig"))
        response = json.loads((CONTRACTS / "scoring-response.schema.json").read_text(encoding="utf-8-sig"))
        openapi = json.loads((CONTRACTS / "openapi.json").read_text(encoding="utf-8-sig"))

        self.assertEqual(set(request["required"]), {"application_id", *ScoringFeatures.__dataclass_fields__})
        self.assertFalse(request["additionalProperties"])
        self.assertEqual(set(response["required"]), {"decision", "score", "approved_amount", "reasons", "factors"})
        self.assertEqual(set(response["properties"]), set(response["required"]))
        self.assertFalse(response["additionalProperties"])
        factors = response["properties"]["factors"]
        self.assertFalse(factors["additionalProperties"])
        self.assertEqual(set(factors["required"]), set(factors["properties"]))
        self.assertEqual(response["properties"]["decision"]["enum"], ["approved", "counter_offer", "rejected"])
        self.assertEqual(response["properties"]["score"]["minimum"], 0)
        self.assertEqual(response["properties"]["score"]["maximum"], 100)
        self.assertEqual(openapi["openapi"], "3.1.0")
        self.assertEqual(
            openapi["paths"]["/v1/score"]["post"]["responses"]["200"]["content"]["application/json"]["schema"]["$ref"],
            "./scoring-response.schema.json",
        )


if __name__ == "__main__":
    unittest.main()
