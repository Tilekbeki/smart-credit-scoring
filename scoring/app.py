"""Deterministic scoring for synthetic applications in a product prototype.

These invented thresholds are not a bank's underwriting policy and must never
be used for a real credit decision.
"""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os


MAX_BODY_BYTES = 16_384
MAX_INPUT_AMOUNT = 1_000_000_000_000
MIN_COUNTER_AMOUNT = 100_000
STRESS_ANNUAL_RATE = 0.20


class ValidationError(ValueError):
    """Invalid request with a client-facing explanation."""


def validate_request(data):
    if not isinstance(data, dict):
        raise ValidationError("Ожидается JSON-объект.")

    fields = {
        "application_id", "monthly_revenue", "monthly_expenses",
        "existing_debt", "monthly_debt_payments", "overdue_count",
        "requested_amount", "term_months",
    }
    missing = fields - data.keys()
    extra = data.keys() - fields
    if missing:
        raise ValidationError("Не хватает полей: " + ", ".join(sorted(missing)) + ".")
    if extra:
        raise ValidationError("Неизвестные поля: " + ", ".join(sorted(extra)) + ".")

    application_id = data["application_id"]
    if not isinstance(application_id, str) or not application_id.strip() or len(application_id) > 100:
        raise ValidationError("application_id должен быть непустой строкой до 100 символов.")

    for name in ("monthly_revenue", "monthly_expenses", "existing_debt", "monthly_debt_payments"):
        value = data[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValidationError(f"{name} должен быть конечным числом.")
        if value < 0 or value > MAX_INPUT_AMOUNT or not math.isfinite(value):
            raise ValidationError(f"{name} должен быть от 0 до {MAX_INPUT_AMOUNT}.")

    for name, minimum, maximum in (
        ("overdue_count", 0, 100),
        ("requested_amount", 1, MAX_INPUT_AMOUNT),
        ("term_months", 1, 120),
    ):
        value = data[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum or value > maximum:
            raise ValidationError(f"{name} должен быть целым числом от {minimum} до {maximum}.")
    return data


def annuity_payment(principal, months):
    """Stress-test payment, not the price or terms of a credit product."""
    rate = STRESS_ANNUAL_RATE / 12
    return principal * rate / (1 - (1 + rate) ** (-months))


def affordable_principal(monthly_payment, months):
    if monthly_payment <= 0:
        return 0
    return monthly_payment / annuity_payment(1, months)


def score_application(request):
    """Return a self-service prototype decision with readable rationale."""
    data = validate_request(request)
    revenue = float(data["monthly_revenue"])
    expenses = float(data["monthly_expenses"])
    debt = float(data["existing_debt"])
    debt_payment = float(data["monthly_debt_payments"])
    overdue_count = data["overdue_count"]
    requested = data["requested_amount"]
    months = data["term_months"]

    operating_cash_flow = revenue - expenses
    margin = operating_cash_flow / revenue if revenue else 0.0
    proposed_payment = annuity_payment(requested, months)
    debt_service_share = (
        (debt_payment + proposed_payment) / operating_cash_flow
        if operating_cash_flow > 0 else None
    )
    leverage = debt / (12 * revenue) if revenue else None

    score = 100
    score -= min(overdue_count, 3) * 20
    score -= 30 if margin < 0.1 else 15 if margin < 0.2 else 5 if margin < 0.35 else 0
    if debt_service_share is None or debt_service_share >= 0.8:
        score -= 40
    elif debt_service_share >= 0.55:
        score -= 25
    elif debt_service_share >= 0.45:
        score -= 15
    elif debt_service_share >= 0.3:
        score -= 5
    score -= 25 if leverage is None or leverage >= 1 else 10 if leverage >= 0.5 else 0
    score = max(0, min(100, score))

    factors = {
        "operating_cash_flow": round(operating_cash_flow, 2),
        "operating_margin": round(margin, 4),
        "estimated_new_monthly_payment": round(proposed_payment, 2),
        "debt_service_share": round(debt_service_share, 4) if debt_service_share is not None else None,
        "existing_debt_to_annual_revenue": round(leverage, 4) if leverage is not None else None,
        "overdue_count": overdue_count,
    }

    hard_reasons = []
    if overdue_count >= 3:
        hard_reasons.append("Три или более просрочки в синтетической истории.")
    if operating_cash_flow <= 0:
        hard_reasons.append("Расходы не покрываются месячной выручкой.")
    elif margin < 0.1:
        hard_reasons.append("Операционная маржа ниже 10%.")
    if leverage is not None and leverage >= 1.5:
        hard_reasons.append("Текущий долг слишком велик относительно годовой выручки.")
    if hard_reasons:
        return {"decision": "rejected", "score": score, "approved_amount": 0, "reasons": hard_reasons, "factors": factors}

    if overdue_count == 0 and margin >= 0.15 and leverage < 1 and debt_service_share <= 0.45:
        return {
            "decision": "approved", "score": score, "approved_amount": requested,
            "reasons": [
                "Операционный поток покрывает оценочный платёж по запрошенной сумме.",
                "В синтетической истории нет просрочек.",
            ],
            "factors": factors,
        }

    # All debt payments must fit within 55% of cash flow for a lower offer.
    available_payment = operating_cash_flow * 0.55 - debt_payment
    risk_cap = 0.75 if overdue_count else 0.85
    suggested = min(affordable_principal(available_payment, months), requested * risk_cap)
    suggested = int(suggested // 1000 * 1000)
    if suggested >= MIN_COUNTER_AMOUNT and suggested >= requested * 0.2 and suggested < requested:
        reasons = ["По оценке денежного потока доступна меньшая сумма."]
        if overdue_count:
            reasons.append("Просрочки в синтетической истории снижают доступный лимит.")
        elif margin < 0.15:
            reasons.append("Невысокая операционная маржа снижает доступный лимит.")
        else:
            reasons.append("Нагрузка по текущим и новому платежам превышает порог одобрения исходной суммы.")
        return {
            "decision": "counter_offer", "score": score,
            "approved_amount": suggested, "reasons": reasons, "factors": factors,
        }

    return {
        "decision": "rejected", "score": score, "approved_amount": 0,
        "reasons": ["Свободного денежного потока недостаточно даже для уменьшенной суммы."],
        "factors": factors,
    }


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/health":
            self._json(404, {"error": "not_found"})
            return
        self._json(200, {"status": "ok", "service": "scoring"})

    def do_POST(self):
        if self.path != "/v1/score":
            self._json(404, {"error": "not_found"})
            return
        if self.headers.get_content_type() != "application/json":
            self._json(415, {"error": "unsupported_media_type", "message": "Требуется application/json."})
            return
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self._json(400, {"error": "invalid_content_length"})
            return
        if length < 1 or length > MAX_BODY_BYTES:
            self._json(413 if length > MAX_BODY_BYTES else 400, {"error": "invalid_body_size"})
            return
        try:
            data = json.loads(self.rfile.read(length))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            self._json(400, {"error": "invalid_json", "message": "Некорректный JSON."})
            return
        try:
            result = score_application(data)
        except ValidationError as error:
            self._json(422, {"error": "validation_error", "message": str(error)})
            return
        self._json(200, result)

    def _json(self, status, data):
        payload = json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8001"))
    ThreadingHTTPServer((host, port), Handler).serve_forever()
