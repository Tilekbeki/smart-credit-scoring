"""Deterministic v1 recommendation policy; final decisions stay in the backend."""

from typing import TYPE_CHECKING

from scoring.metrics import FinancialMetrics, annuity_payment

if TYPE_CHECKING:
    from scoring.service import ScoringFeatures


def recommend(features: "ScoringFeatures", metrics: FinancialMetrics) -> tuple[str, int, list[str]]:
    reasons = []
    if metrics.overdue_count >= 3:
        reasons.append("Три или более просрочки в синтетической кредитной истории.")
    if metrics.operating_cash_flow <= 0:
        reasons.append("Операционные расходы не покрываются месячной выручкой.")
    elif metrics.operating_margin < 0.10:
        reasons.append("Операционная маржа ниже 10%.")
    leverage = metrics.existing_debt_to_annual_revenue
    if leverage is not None and leverage >= 1.5:
        reasons.append("Текущий долг составляет не менее 150% годовой выручки.")
    if reasons:
        return "rejected", 0, reasons

    if (metrics.overdue_count == 0 and metrics.operating_margin >= 0.15
            and leverage is not None and leverage < 1
            and metrics.debt_service_share is not None and metrics.debt_service_share <= 0.45):
        return "approved", features.requested_amount, [
            "Операционный поток покрывает оценочный платёж по запрошенной сумме.",
            "В синтетической кредитной истории нет просрочек.",
        ]

    # With a smaller principal, existing and new payments may use at most 55%
    # of positive operating cash flow. Cap the suggestion for other risk flags.
    available_payment = metrics.operating_cash_flow * 0.55 - features.monthly_debt_payments
    affordable_amount = max(0, int(available_payment / annuity_payment(1, features.term_months)))
    risk_cap = 0.75 if metrics.overdue_count else 0.85
    suggested = min(affordable_amount, int(features.requested_amount * risk_cap))
    if suggested >= (features.requested_amount + 4) // 5:
        reasons = ["По оценке денежного потока доступна меньшая сумма."]
        if metrics.overdue_count:
            reasons.append("Просрочки в синтетической истории снижают рекомендуемый лимит.")
        elif metrics.operating_margin < 0.15:
            reasons.append("Невысокая операционная маржа снижает рекомендуемый лимит.")
        elif leverage is not None and leverage >= 1:
            reasons.append("Высокий текущий долг снижает рекомендуемый лимит.")
        else:
            reasons.append("Нагрузка по текущим и новому платежам превышает порог для исходной суммы.")
        return "counter_offer", suggested, reasons

    return "rejected", 0, ["Свободного денежного потока недостаточно даже для уменьшенной суммы."]
