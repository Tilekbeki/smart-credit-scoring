"""Dependency-free inference from a trusted, versioned logistic-regression JSON artifact."""

import json
import math
from pathlib import Path

from scoring.ml_features import FEATURE_NAMES, feature_vector
from scoring.service import ScoringFeatures


ARTIFACT_PATH = Path(__file__).resolve().parent / "artifacts" / "logreg_v1.json"


class LogisticModel:
    def __init__(self, artifact_path: Path = ARTIFACT_PATH):
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        if artifact["model_version"] != "logreg_v1" or tuple(artifact["feature_names"]) != FEATURE_NAMES:
            raise ValueError("Incompatible logistic model artifact")
        self.mean = tuple(float(value) for value in artifact["mean"])
        self.scale = tuple(float(value) for value in artifact["scale"])
        self.coef = tuple(float(value) for value in artifact["coef"])
        self.intercept = float(artifact["intercept"])
        if not all(len(values) == len(FEATURE_NAMES) for values in (self.mean, self.scale, self.coef)):
            raise ValueError("Invalid logistic model dimensions")
        if any(value <= 0 for value in self.scale):
            raise ValueError("Invalid logistic model scale")
        if not all(math.isfinite(value) for value in (*self.mean, *self.scale, *self.coef, self.intercept)):
            raise ValueError("Non-finite logistic model parameter")

    def predict_default_probability(self, features: ScoringFeatures) -> float:
        values = feature_vector(features)
        logit = self.intercept + sum(
            weight * (value - mean) / scale
            for value, mean, scale, weight in zip(values, self.mean, self.scale, self.coef)
        )
        if logit >= 0:
            return 1 / (1 + math.exp(-logit))
        exp_logit = math.exp(logit)
        return exp_logit / (1 + exp_logit)

    def predict_score(self, features: ScoringFeatures) -> int:
        return round(100 * (1 - self.predict_default_probability(features)))
