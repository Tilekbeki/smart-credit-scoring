"""Stable request validation and inference boundary for scoring models."""

from dataclasses import dataclass

from scoring.models import ScoringModel, get_model
from scoring.metrics import calculate_metrics
from scoring.recommendation import recommend


MAX_MONEY_RUB = 1_000_000_000_000


class ValidationError(ValueError):
    """Input does not satisfy the public scoring contract."""


@dataclass(frozen=True)
class ScoringFeatures:
    monthly_revenue: int
    monthly_expenses: int
    existing_debt: int
    monthly_debt_payments: int
    overdue_count: int
    requested_amount: int
    term_months: int


def validate_request(data: object) -> ScoringFeatures:
    if not isinstance(data, dict):
        raise ValidationError("Ожидается JSON-объект.")

    fields = {"application_id", *ScoringFeatures.__dataclass_fields__}
    missing = fields - data.keys()
    extra = data.keys() - fields
    if missing:
        raise ValidationError("Не хватает полей: " + ", ".join(sorted(missing)) + ".")
    if extra:
        raise ValidationError("Неизвестные поля: " + ", ".join(sorted(extra)) + ".")

    application_id = data["application_id"]
    if not isinstance(application_id, str) or not application_id.strip() or len(application_id) > 100:
        raise ValidationError("application_id должен быть непустой строкой до 100 символов.")

    bounds = {
        "monthly_revenue": (0, MAX_MONEY_RUB),
        "monthly_expenses": (0, MAX_MONEY_RUB),
        "existing_debt": (0, MAX_MONEY_RUB),
        "monthly_debt_payments": (0, MAX_MONEY_RUB),
        "overdue_count": (0, 100),
        "requested_amount": (1, MAX_MONEY_RUB),
        "term_months": (1, 120),
    }
    for name, (minimum, maximum) in bounds.items():
        value = data[name]
        if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
            raise ValidationError(f"{name} должен быть целым числом от {minimum} до {maximum}.")

    return ScoringFeatures(**{name: data[name] for name in ScoringFeatures.__dataclass_fields__})


def score_application(request: object, model: ScoringModel | None = None) -> dict:
    """Return a score, factors and a preliminary recommendation for the backend."""
    features = validate_request(request)
    score = (model or get_model()).predict_score(features)
    if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 100:
        raise ValueError("The scoring model must return an integer from 0 to 100.")
    metrics = calculate_metrics(features)
    decision, approved_amount, reasons = recommend(features, metrics)
    return {
        "decision": decision,
        "score": score,
        "approved_amount": approved_amount,
        "reasons": reasons,
        "factors": metrics.response_factors(),
    }
