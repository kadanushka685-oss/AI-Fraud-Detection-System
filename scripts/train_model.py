"""Generate the synthetic dataset if needed, train and compare models, and save the best one.

Usage (from the project folder):
    python scripts/train_model.py                # reuse existing data if present
    python scripts/train_model.py --regenerate   # rebuild the dataset, sample file and models
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from config.settings import (  # noqa: E402
    EVALUATION_PATH,
    MODEL_PATH,
    SAMPLE_DATA_PATH,
    TRAINING_DATA_PATH,
)
from src.data_generator import generate_synthetic_transactions  # noqa: E402
from src.model_pipeline import train_and_compare  # noqa: E402
from src.predictor import save_model_bundle  # noqa: E402

SAMPLE_COLUMNS = [
    "transaction_id", "timestamp", "merchant_category", "amount", "hour_of_day",
    "distance_from_home_km", "card_present", "foreign_transaction",
    "num_transactions_last_24h", "customer_avg_amount",
]


def write_sample_file(path):
    """A small unlabelled CSV for the Batch CSV Analysis page. It includes some missing values."""
    sample = generate_synthetic_transactions(n_rows=200, fraud_rate=0.05, seed=2024)
    sample[SAMPLE_COLUMNS].to_csv(path, index=False)


def print_report(summary):
    print(f"\nDataset: {summary['dataset_rows']:,} rows, fraud rate {summary['fraud_rate_pct']:.2f}%")
    print(f"Test split: {summary['test_rows']:,} rows ({summary['test_fraud_cases']} fraud cases)\n")
    header = f"{'Model':<22}{'Precision':>10}{'Recall':>9}{'F1':>7}{'ROC-AUC':>9}{'PR-AUC':>8}"
    print(header)
    print("-" * len(header))
    for name, m in summary["models"].items():
        print(f"{name:<22}{m['precision']:>10.3f}{m['recall']:>9.3f}{m['f1']:>7.3f}"
              f"{m['roc_auc']:>9.3f}{m['pr_auc']:>8.3f}")
    print(f"\nBest model by {summary['selection_metric']}: {summary['best_model']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train and compare fraud detection models.")
    parser.add_argument("--rows", type=int, default=20_000, help="rows to generate (default 20000)")
    parser.add_argument("--regenerate", action="store_true", help="rebuild the synthetic dataset")
    args = parser.parse_args(argv)

    TRAINING_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if args.regenerate or not TRAINING_DATA_PATH.exists():
        print(f"Generating {args.rows:,} synthetic transactions...")
        data = generate_synthetic_transactions(n_rows=args.rows)
        data.to_csv(TRAINING_DATA_PATH, index=False)
        print(f"Saved {TRAINING_DATA_PATH}")
    else:
        data = pd.read_csv(TRAINING_DATA_PATH, parse_dates=["timestamp"])
        print(f"Using existing dataset {TRAINING_DATA_PATH}")

    if args.regenerate or not SAMPLE_DATA_PATH.exists():
        write_sample_file(SAMPLE_DATA_PATH)
        print(f"Saved {SAMPLE_DATA_PATH}")

    print("Training and comparing models...")
    result = train_and_compare(data)
    save_model_bundle(result["best_model"], result["best_model_name"], path=MODEL_PATH)
    EVALUATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVALUATION_PATH.write_text(json.dumps(result["summary"], indent=2), encoding="utf-8")

    print_report(result["summary"])
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved evaluation report to {EVALUATION_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
