"""Train a logistic baseline on generated applications and export safe JSON coefficients."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scoring.ml_features import FEATURE_NAMES, feature_vector
from scoring.service import ScoringFeatures


def read_dataset(path: Path) -> tuple[list[list[float]], list[int]]:
    vectors: list[list[float]] = []
    labels: list[int] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            features = ScoringFeatures(**{
                name: int(row[name]) for name in ScoringFeatures.__dataclass_fields__
            })
            vectors.append(feature_vector(features))
            labels.append(int(row["default_12m"]))
    if len(vectors) < 2 or set(labels) != {0, 1}:
        raise ValueError(f"Dataset {path} must have both target classes and at least two rows")
    return vectors, labels


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(labels: list[int], probabilities) -> dict[str, float]:
    return {
        "n": len(labels),
        "default_rate": round(sum(labels) / len(labels), 6),
        "roc_auc": round(float(roc_auc_score(labels, probabilities)), 6),
        "brier": round(float(brier_score_loss(labels, probabilities)), 6),
        "baseline_brier": round(float(brier_score_loss(labels, [sum(labels) / len(labels)] * len(labels))), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/synthetic_v1"))
    parser.add_argument("--output", type=Path, default=Path("scoring/artifacts/logreg_v1.json"))
    args = parser.parse_args()
    train_path, test_path = args.data_dir / "train.csv", args.data_dir / "test.csv"
    train_x, train_y = read_dataset(train_path)
    test_x, test_y = read_dataset(test_path)

    pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=20261024))
    pipeline.fit(train_x, train_y)
    train_probabilities = pipeline.predict_proba(train_x)[:, 1]
    test_probabilities = pipeline.predict_proba(test_x)[:, 1]
    scaler = pipeline.named_steps["standardscaler"]
    classifier = pipeline.named_steps["logisticregression"]
    artifact = {
        "model_version": "logreg_v1",
        "target": "default_12m",
        "feature_names": FEATURE_NAMES,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": classifier.coef_[0].tolist(),
        "intercept": float(classifier.intercept_[0]),
        "training": {
            "python_version": platform.python_version(),
            "sklearn_version": sklearn.__version__,
            "generator_version": json.loads((args.data_dir / "generation.json").read_text(encoding="utf-8"))["generator_version"],
            "train_sha256": file_sha256(train_path),
            "test_sha256": file_sha256(test_path),
            "train_metrics": metrics(train_y, train_probabilities),
            "test_metrics": metrics(test_y, test_probabilities),
        },
    }
    # The service intentionally does not depend on sklearn. Check that its
    # standardized linear formula reproduces sklearn's test probabilities.
    manual_probabilities = []
    for vector in test_x:
        logit = artifact["intercept"] + sum(
            coefficient * (value - mean) / scale
            for value, mean, scale, coefficient in zip(
                vector, artifact["mean"], artifact["scale"], artifact["coef"]
            )
        )
        manual_probabilities.append(1 / (1 + math.exp(-logit)) if logit >= 0
                                    else math.exp(logit) / (1 + math.exp(logit)))
    max_error = max(abs(float(sklearn_p) - manual_p)
                    for sklearn_p, manual_p in zip(test_probabilities, manual_probabilities))
    if max_error > 1e-12:
        raise ValueError(f"Exported model differs from sklearn: {max_error}")
    artifact["training"]["inference_parity_max_abs_diff"] = max_error
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(artifact["training"], indent=2))


if __name__ == "__main__":
    main()
