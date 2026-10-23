"""Model interface. A future trained model implements predict_score(features)."""

from functools import lru_cache
import os
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from scoring.service import ScoringFeatures


class ScoringModel(Protocol):
    def predict_score(self, features: "ScoringFeatures") -> int:
        """Higher means lower estimated risk; return an integer in [0, 100]."""


@lru_cache(maxsize=1)
def get_model() -> ScoringModel:
    """Load one model per process. Register future models here without changing the API."""
    from scoring.rules import RuleBasedModel

    model_name = os.environ.get("SCORING_MODEL", "rules_v1")
    registry = {"rules_v1": RuleBasedModel}
    if model_name not in registry:
        raise ValueError(f"Unknown SCORING_MODEL: {model_name}")
    return registry[model_name]()
