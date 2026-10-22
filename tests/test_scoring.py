from email.message import Message
from io import BytesIO
import json
import unittest

from scoring.app import Handler, ValidationError, score_application


SHOP = {
    "application_id": "test-shop-1",
    "monthly_revenue": 2_400_000,
    "monthly_expenses": 1_500_000,
    "existing_debt": 500_000,
    "monthly_debt_payments": 90_000,
    "overdue_count": 0,
    "requested_amount": 1_500_000,
    "term_months": 12,
}


class ScoringTests(unittest.TestCase):
    def test_healthy_shop_gets_full_automated_approval(self):
        result = score_application(SHOP)
        self.assertEqual(result["decision"], "approved")
        self.assertEqual(result["approved_amount"], 1_500_000)
        self.assertGreaterEqual(result["score"], 80)
        self.assertEqual(result["factors"]["operating_cash_flow"], 900_000)
        self.assertTrue(result["reasons"])

    def test_large_request_gets_lower_counter_offer(self):
        result = score_application({**SHOP, "requested_amount": 8_000_000})
        self.assertEqual(result["decision"], "counter_offer")
        self.assertGreaterEqual(result["approved_amount"], 100_000)
        self.assertLess(result["approved_amount"], 8_000_000)
        self.assertLessEqual(
            90_000 + result["approved_amount"] * (
                result["factors"]["estimated_new_monthly_payment"] / 8_000_000
            ),
            900_000 * 0.55 + 1,
        )

    def test_critical_risks_are_rejected(self):
        overdue = score_application({**SHOP, "overdue_count": 3})
        self.assertEqual(overdue["decision"], "rejected")
        self.assertEqual(overdue["approved_amount"], 0)
        no_cash = score_application({**SHOP, "monthly_expenses": 2_500_000})
        self.assertEqual(no_cash["decision"], "rejected")
        self.assertIsNone(no_cash["factors"]["debt_service_share"])

    def test_input_validation_rejects_bad_types_and_ranges(self):
        for payload in (
            {**SHOP, "requested_amount": True},
            {**SHOP, "monthly_revenue": float("nan")},
            {**SHOP, "term_months": 0},
            {**SHOP, "extra": 1},
            {key: value for key, value in SHOP.items() if key != "requested_amount"},
        ):
            with self.subTest(payload=payload), self.assertRaises(ValidationError):
                score_application(payload)


class HTTPTests(unittest.TestCase):
    def request(self, method, path, payload=None, content_type="application/json", raw=None):
        # Exercise the HTTP handler directly so the test also runs in
        # environments that prohibit binding local sockets.
        handler = Handler.__new__(Handler)
        handler.path = path
        handler.headers = Message()
        handler.headers["Content-Type"] = content_type
        body = raw if raw is not None else json.dumps(payload).encode() if payload is not None else b""
        handler.headers["Content-Length"] = str(len(body))
        handler.rfile = BytesIO(body)
        handler.wfile = BytesIO()
        handler.send_response = lambda status: setattr(handler, "response_status", status)
        handler.send_header = lambda *_: None
        handler.end_headers = lambda: None
        if method == "GET":
            handler.do_GET()
        else:
            handler.do_POST()
        return handler.response_status, json.loads(handler.wfile.getvalue())

    def test_health_and_approved_response(self):
        status, health = self.request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(health["service"], "scoring")
        status, result = self.request("POST", "/v1/score", SHOP)
        self.assertEqual(status, 200)
        self.assertEqual(result["decision"], "approved")

    def test_http_validation_and_invalid_json(self):
        status, result = self.request("POST", "/v1/score", {**SHOP, "term_months": -1})
        self.assertEqual(status, 422)
        self.assertEqual(result["error"], "validation_error")
        status, result = self.request("POST", "/v1/score", raw=b"{")
        self.assertEqual(status, 400)
        self.assertEqual(result["error"], "invalid_json")
        status, result = self.request("POST", "/v1/score", SHOP, content_type="text/plain")
        self.assertEqual(status, 415)
        self.assertEqual(result["error"], "unsupported_media_type")
