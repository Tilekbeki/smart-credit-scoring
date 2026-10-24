"""Reproducible synthetic applications and simulated 12-month payment outcomes.

The target is produced by future cash-flow shocks, not by rules_v1 or its score.
No generated row represents a real customer.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import platform
import random

from scoring.metrics import annuity_payment


FIELDS = (
    "application_id", "monthly_revenue", "monthly_expenses", "existing_debt",
    "monthly_debt_payments", "overdue_count", "requested_amount", "term_months",
    "default_12m",
)
GENERATOR_VERSION = "synthetic_cashflow_v1"


def make_application(rng: random.Random, index: int, split: str) -> dict[str, int | str]:
    # Revenue is right-skewed. Most businesses have a 10-45% operating margin;
    # a small stressed group spends more than it earns.
    revenue = max(50_000, min(50_000_000, round(rng.lognormvariate(math.log(1_800_000), 0.8))))
    expense_ratio = rng.uniform(0.52, 0.85) if rng.random() < 0.87 else rng.uniform(0.85, 1.08)
    expenses = round(revenue * expense_ratio)

    # Outstanding balance and monthly payment are linked by an amortization
    # period, so they cannot vary independently in impossible combinations.
    debt_ratio = 0 if rng.random() < 0.18 else min(1.7, max(0.03, rng.lognormvariate(math.log(0.18), 0.75)))
    debt = round(12 * revenue * debt_ratio)
    debt_months = rng.choice((12, 18, 24, 36, 48, 60))
    debt_payment = round(annuity_payment(debt, debt_months)) if debt else 0

    stress = max(0, expense_ratio - 0.82) + max(0, debt_payment / revenue - 0.18)
    overdue_chance = min(0.65, 0.06 + 1.2 * stress)
    overdue = 0 if rng.random() >= overdue_chance else rng.choices((1, 2, 3, 4), (0.65, 0.23, 0.09, 0.03))[0]

    requested_ratio = rng.uniform(0.2, 1.8) if rng.random() < 0.85 else rng.uniform(1.8, 3.5)
    requested = round(revenue * requested_ratio)
    term = rng.choices((6, 12, 18, 24, 36, 48, 60), (2, 8, 10, 28, 26, 16, 10))[0]
    return {
        "application_id": f"SYN-{split.upper()}-{index:06d}",
        "monthly_revenue": revenue,
        "monthly_expenses": expenses,
        "existing_debt": debt,
        "monthly_debt_payments": debt_payment,
        "overdue_count": overdue,
        "requested_amount": requested,
        "term_months": term,
    }


def simulate_default(application: dict[str, int | str], rng: random.Random) -> int:
    """Label 1 when at least two future months cannot cover all debt payments."""
    revenue = int(application["monthly_revenue"])
    expenses = int(application["monthly_expenses"])
    new_payment = annuity_payment(
        int(application["requested_amount"]), int(application["term_months"])
    )
    overdue = int(application["overdue_count"])
    margin_cash = max(0, revenue - expenses)
    reserve = margin_cash * rng.uniform(0.2, 1.6) / (1 + overdue * 0.4)
    # Prior arrears represent instability that is not fully visible in a single
    # monthly snapshot; in this toy world they worsen the next-year shock.
    annual_shock = rng.gauss(-0.10 * overdue, 0.12)
    monthly_volatility = rng.uniform(0.06, 0.20)
    expense_inflation = rng.uniform(0, 0.08)
    missed_months = 0

    for month in range(1, 13):
        due = int(application["monthly_debt_payments"]) + (new_payment if month <= int(application["term_months"]) else 0)
        future_revenue = revenue * max(0.1, 1 + annual_shock + rng.gauss(0, monthly_volatility))
        future_expenses = expenses * max(0.5, 1 + expense_inflation * month / 12 + rng.gauss(0, 0.035))
        remaining = reserve + future_revenue - future_expenses - due
        if remaining < 0:
            missed_months += 1
            reserve = 0
        else:
            reserve = min(2 * revenue, remaining)
        if missed_months >= 2:
            return 1
    return 0


def generate_file(path: Path, count: int, seed: int, split: str) -> int:
    rng = random.Random(seed)
    defaults = 0
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for index in range(1, count + 1):
            application = make_application(rng, index, split)
            label = simulate_default(application, rng)
            writer.writerow({**application, "default_12m": label})
            defaults += label
    return defaults


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/synthetic_v1"))
    parser.add_argument("--train-size", type=int, default=20_000)
    parser.add_argument("--test-size", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=20261024)
    args = parser.parse_args()
    if args.train_size < 2 or args.test_size < 2:
        parser.error("train-size and test-size must be at least 2")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_defaults = generate_file(args.output_dir / "train.csv", args.train_size, args.seed, "train")
    test_defaults = generate_file(args.output_dir / "test.csv", args.test_size, args.seed + 1, "test")
    metadata = {
        "generator_version": GENERATOR_VERSION,
        "python_version": platform.python_version(),
        "seed": args.seed,
        "train_size": args.train_size,
        "test_size": args.test_size,
        "train_default_rate": round(train_defaults / args.train_size, 6),
        "test_default_rate": round(test_defaults / args.test_size, 6),
        "target": "default_12m: at least two of the next 12 months lack cash for existing and requested-loan payments",
        "note": "Entirely synthetic simulation; not real borrower outcomes.",
    }
    (args.output_dir / "generation.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
