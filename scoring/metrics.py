"""Shared financial calculations for the score and the recommendation policy."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scoring.service import ScoringFeatures


STRESS_ANNUAL_RATE = 0.20


def annuity_payment(principal: int | float, months: int) -> float:
    """Estimated payment at a synthetic stress rate, not a loan offer."""
    rate = STRESS_ANNUAL_RATE / 12
    return principal * rate / (1 - (1 + rate) ** -months)


@dataclass(frozen=True)
class FinancialMetrics:
    operating_cash_flow: int
    operating_margin: float
    estimated_new_monthly_payment: float
    debt_service_share: float | None
    existing_debt_to_annual_revenue: float | None
    overdue_count: int

    def response_factors(self) -> dict[str, int | float | None]:
        return {
            "operating_cash_flow": self.operating_cash_flow,
            "operating_margin": round(self.operating_margin, 4),
            "estimated_new_monthly_payment": round(self.estimated_new_monthly_payment, 2),
            "debt_service_share": round(self.debt_service_share, 4) if self.debt_service_share is not None else None,
            "existing_debt_to_annual_revenue": round(self.existing_debt_to_annual_revenue, 4)
            if self.existing_debt_to_annual_revenue is not None else None,
            "overdue_count": self.overdue_count,
        }


def calculate_metrics(features: "ScoringFeatures") -> FinancialMetrics:
    cash_flow = features.monthly_revenue - features.monthly_expenses
    payment = annuity_payment(features.requested_amount, features.term_months)
    return FinancialMetrics(
        operating_cash_flow=cash_flow,
        operating_margin=cash_flow / features.monthly_revenue if features.monthly_revenue else 0.0,
        estimated_new_monthly_payment=payment,
        debt_service_share=(features.monthly_debt_payments + payment) / cash_flow if cash_flow > 0 else None,
        existing_debt_to_annual_revenue=features.existing_debt / (12 * features.monthly_revenue)
        if features.monthly_revenue else None,
        overdue_count=features.overdue_count,
    )
