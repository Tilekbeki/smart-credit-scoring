"""Deterministic baseline for synthetic data, not a calibrated risk model."""

from scoring.service import ScoringFeatures


STRESS_ANNUAL_RATE = 0.20


def annuity_payment(principal: int, months: int) -> float:
    rate = STRESS_ANNUAL_RATE / 12
    return principal * rate / (1 - (1 + rate) ** -months)


class RuleBasedModel:
    def predict_score(self, features: ScoringFeatures) -> int:
        revenue = features.monthly_revenue
        cash_flow = revenue - features.monthly_expenses
        margin = cash_flow / revenue if revenue else 0.0
        debt_service_share = (
            (features.monthly_debt_payments + annuity_payment(features.requested_amount, features.term_months))
            / cash_flow if cash_flow > 0 else None
        )
        leverage = features.existing_debt / (12 * revenue) if revenue else None

        score = 100 - min(features.overdue_count, 3) * 20
        score -= 30 if margin < 0.10 else 15 if margin < 0.20 else 5 if margin < 0.35 else 0
        if debt_service_share is None or debt_service_share >= 0.80:
            score -= 40
        elif debt_service_share >= 0.55:
            score -= 25
        elif debt_service_share >= 0.45:
            score -= 15
        elif debt_service_share >= 0.30:
            score -= 5
        score -= 25 if leverage is None or leverage >= 1 else 10 if leverage >= 0.5 else 0
        return max(0, min(100, score))
