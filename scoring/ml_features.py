"""Feature transform shared by offline training and online inference."""

import math

from scoring.metrics import calculate_metrics
from scoring.service import ScoringFeatures


FEATURE_NAMES = (
    "log_monthly_revenue_millions",
    "operating_margin",
    "debt_service_share",
    "existing_debt_to_annual_revenue",
    "overdue_count",
    "requested_to_annual_revenue",
    "term_years",
)


def feature_vector(features: ScoringFeatures) -> list[float]:
    metrics = calculate_metrics(features)
    return [
        math.log1p(features.monthly_revenue / 1_000_000),
        max(-1.0, min(1.0, metrics.operating_margin)),
        min(5.0, metrics.debt_service_share) if metrics.debt_service_share is not None else 5.0,
        min(3.0, metrics.existing_debt_to_annual_revenue)
        if metrics.existing_debt_to_annual_revenue is not None else 3.0,
        float(min(5, features.overdue_count)),
        min(5.0, features.requested_amount / (12 * max(1, features.monthly_revenue))),
        features.term_months / 12,
    ]
