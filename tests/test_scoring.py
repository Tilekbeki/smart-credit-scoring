from email.message import Message
from io import BytesIO
import json
import unittest
from unittest.mock import patch

from scoring.app import Handler
from scoring.metrics import annuity_payment
from scoring.service import ValidationError, score_application, validate_request


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


class ScoringTests(unittest.TestCase):
    def test_score_and_recommendation_are_deterministic(self):
        result = score_application(SHOP)
        self.assertEqual(result["score"], 100)
        self.assertEqual(result["decision"], "approved")
        self.assertEqual(result["approved_amount"], SHOP["requested_amount"])
        self.assertTrue(result["reasons"])
        self.assertEqual(result["factors"]["operating_cash_flow"], 900_000)
        self.assertEqual(result["factors"]["overdue_count"], 0)
        self.assertEqual(result, score_application(SHOP))

    def test_risk_factors_lower_score(self):
        baseline = score_application(SHOP)["score"]
        self.assertLess(score_application({**SHOP, "overdue_count": 2})["score"], baseline)
        self.assertLess(score_application({**SHOP, "monthly_expenses": 2_300_000})["score"], baseline)
        self.assertLess(score_application({**SHOP, "requested_amount": 8_000_000})["score"], baseline)
        self.assertLess(score_application({**SHOP, "existing_debt": 25_000_000})["score"], baseline)

    def test_counter_offer_is_smaller_and_affordable(self):
        result = score_application({**SHOP, "overdue_count": 1, "requested_amount": 8_000_000})
        self.assertEqual(result["decision"], "counter_offer")
        self.assertGreater(result["approved_amount"], 0)
        self.assertLess(result["approved_amount"], 8_000_000)
        self.assertLessEqual(result["approved_amount"], 8_000_000 * 0.75)
        offered_payment = annuity_payment(result["approved_amount"], SHOP["term_months"])
        self.assertLessEqual(SHOP["monthly_debt_payments"] + offered_payment, 900_000 * 0.55 + 0.01)
        self.assertEqual(result["factors"]["overdue_count"], 1)
        self.assertGreater(result["factors"]["estimated_new_monthly_payment"], 700_000)
        self.assertTrue(any("Просрочки" in reason for reason in result["reasons"]))

    def test_hard_rejection_has_zero_amount_and_reasons(self):
        result = score_application({**SHOP, "monthly_expenses": 2_500_000, "overdue_count": 3})
        self.assertEqual(result["decision"], "rejected")
        self.assertEqual(result["approved_amount"], 0)
        self.assertEqual(len(result["reasons"]), 2)
        self.assertIsNone(result["factors"]["debt_service_share"])
        self.assertLess(result["score"], score_application(SHOP)["score"])

    def test_zero_revenue_ratios_are_null(self):
        result = score_application({**SHOP, "monthly_revenue": 0})
        self.assertIsNone(result["factors"]["debt_service_share"])
        self.assertIsNone(result["factors"]["existing_debt_to_annual_revenue"])
        self.assertEqual(result["decision"], "rejected")

    def test_application_id_is_not_a_model_feature(self):
        self.assertEqual(
            score_application(SHOP),
            score_application({**SHOP, "application_id": "another-application"}),
        )

    def test_model_can_be_replaced_without_changing_contract(self):
        class ReplacementModel:
            def predict_score(self, features):
                self.features = features
                return 61

        model = ReplacementModel()
        replaced = score_application(SHOP, model)
        self.assertEqual(replaced["score"], 61)
        self.assertEqual(replaced["decision"], "approved")
        self.assertEqual(replaced["factors"], score_application(SHOP)["factors"])
        self.assertEqual(model.features.monthly_revenue, SHOP["monthly_revenue"])
        self.assertFalse(hasattr(model.features, "application_id"))

        for bad_score in (True, 101, -1, 65.5):
            with self.subTest(score=bad_score):
                class InvalidModel:
                    def predict_score(self, _features):
                        return bad_score

                with self.assertRaises(ValueError):
                    score_application(SHOP, InvalidModel())

    def test_contract_rejects_missing_extra_or_invalid_fields(self):
        bad_inputs = (
            {key: value for key, value in SHOP.items() if key != "term_months"},
            {**SHOP, "unknown": 1},
            {**SHOP, "application_id": "  "},
            {**SHOP, "monthly_revenue": 2.5},
            {**SHOP, "overdue_count": True},
            {**SHOP, "requested_amount": 0},
            {**SHOP, "term_months": 121},
            {**SHOP, "existing_debt": -1},
            [],
        )
        for payload in bad_inputs:
            with self.subTest(payload=payload), self.assertRaises(ValidationError):
                validate_request(payload)


class HTTPTests(unittest.TestCase):
    def request(self, method, path, payload=None, content_type="application/json", raw=None):
        handler = Handler.__new__(Handler)
        handler.path = path
        handler.headers = Message()
        handler.headers["Content-Type"] = content_type
        body = raw if raw is not None else json.dumps(payload).encode() if payload is not None else b""
        handler.headers["Content-Length"] = str(len(body))
        handler.rfile = BytesIO(body)
        handler.wfile = BytesIO()
        status = []
        headers = {}
        handler.send_response = lambda code: status.append(code)
        handler.send_header = lambda name, value: headers.__setitem__(name, value)
        handler.end_headers = lambda: None
        handler.log_error = lambda *_args: None
        getattr(handler, "do_" + method)()
        return status[0], json.loads(handler.wfile.getvalue()), headers

    def test_health_and_complete_response(self):
        self.assertEqual(self.request("GET", "/health")[0], 200)
        status, response, headers = self.request("POST", "/v1/score", SHOP)
        self.assertEqual(status, 200)
        self.assertEqual(set(response), {"decision", "score", "approved_amount", "reasons", "factors"})
        self.assertEqual(response["score"], 100)
        self.assertIn("application/json", headers["Content-Type"])

    def test_input_errors(self):
        self.assertEqual(self.request("POST", "/v1/score", {**SHOP, "term_months": 0})[0], 422)
        self.assertEqual(self.request("POST", "/v1/score", raw=b"{bad")[0], 400)
        self.assertEqual(self.request("POST", "/v1/score", SHOP, content_type="text/plain")[0], 415)
        self.assertEqual(self.request("POST", "/unknown", SHOP)[0], 404)
        self.assertEqual(self.request("GET", "/unknown")[0], 404)

    def test_model_failure_is_a_server_error_not_a_credit_result(self):
        with patch("scoring.app.score_application", side_effect=RuntimeError("model unavailable")):
            status, response, _ = self.request("POST", "/v1/score", SHOP)
        self.assertEqual(status, 500)
        self.assertEqual(response["error"], "scoring_error")
        self.assertNotIn("score", response)


if __name__ == "__main__":
    unittest.main()
