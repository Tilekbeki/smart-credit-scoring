"""Deterministic baseline for synthetic data, not a calibrated risk model."""

from scoring.metrics import calculate_metrics
from scoring.service import ScoringFeatures


class RuleBasedModel:
    def predict_score(self, features: ScoringFeatures) -> int:
        metrics = calculate_metrics(features)
        margin = metrics.operating_margin
        debt_service_share = metrics.debt_service_share
        leverage = metrics.existing_debt_to_annual_revenue

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
        score -= 40 if leverage is None or leverage >= 1.5 else 25 if leverage >= 1 else 10 if leverage >= 0.5 else 0
        return max(0, min(100, score))
