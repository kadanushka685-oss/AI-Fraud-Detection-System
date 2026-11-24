"""Saving, loading and scoring with the trained model bundle.

The bundle is a dict: {"pipeline", "model_name", "features", "trained_at"}. Keeping the feature
list and model name next to the pipeline means the app never has to guess either.
"""
from datetime import datetime
from pathlib import Path

import joblib

from config.settings import (
    DECISION_THRESHOLD,
    LOW_RISK_MAX,
    MEDIUM_RISK_MAX,
    MODEL_PATH,
    REQUIRED_INPUT_COLUMNS,
)


class ModelNotFoundError(FileNotFoundError):
    """Raised when no trained model file exists yet."""


def save_model_bundle(pipeline, model_name, path=MODEL_PATH, trained_at=None):
    """Persist the fitted pipeline with its metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "pipeline": pipeline,
        "model_name": model_name,
        "features": list(REQUIRED_INPUT_COLUMNS),
        "trained_at": trained_at or datetime.now().isoformat(timespec="seconds"),
    }
    joblib.dump(bundle, path, compress=3)
    return path


def load_model(path=MODEL_PATH):
    """Load the model bundle, or raise ModelNotFoundError with the command to fix it."""
    path = Path(path)
    if not path.exists():
        raise ModelNotFoundError(
            f"No trained model found at {path}. Run `python scripts/train_model.py` first."
        )
    return joblib.load(path)


def risk_level(probability):
    """Map a fraud probability to Low, Medium or High."""
    if probability < LOW_RISK_MAX:
        return "Low"
    if probability < MEDIUM_RISK_MAX:
        return "Medium"
    return "High"


def predict_transactions(bundle, transactions):
    """Score transactions. Returns a copy with probability, risk level and predicted label."""
    if transactions.empty:
        raise ValueError("There are no transactions to score.")
    missing = [c for c in bundle["features"] if c not in transactions.columns]
    if missing:
        raise ValueError(f"Transactions are missing column(s): {', '.join(missing)}")

    probabilities = bundle["pipeline"].predict_proba(transactions[bundle["features"]])[:, 1]
    result = transactions.copy()
    result["fraud_probability"] = probabilities
    result["risk_level"] = [risk_level(p) for p in probabilities]
    result["predicted_fraud"] = (probabilities >= DECISION_THRESHOLD).astype(int)
    result["model_name"] = bundle["model_name"]
    return result
