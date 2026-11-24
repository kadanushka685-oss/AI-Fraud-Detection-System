"""Tests for SQLite storage. Every test uses a temporary database file."""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src import database as db


def _scored():
    return pd.DataFrame({
        "transaction_id": ["T0", "T1", "T2"],
        "amount": [10.0, 300.0, 55.5],
        "merchant_category": ["grocery", "electronics", "travel"],
        "hour_of_day": [10, 2, 23],
        "distance_from_home_km": [1.0, np.nan, 300.0],
        "card_present": [1.0, 0.0, 0.0],
        "foreign_transaction": [0.0, 0.0, 1.0],
        "num_transactions_last_24h": [1.0, np.nan, 7.0],
        "customer_avg_amount": [40.0, 50.0, np.nan],
        "fraud_probability": [0.05, 0.55, 0.92],
        "risk_level": ["Low", "Medium", "High"],
        "predicted_fraud": [0, 1, 1],
    })


def test_init_creates_empty_table(tmp_path):
    path = tmp_path / "t.db"
    db.init_db(path)
    assert path.exists()
    assert db.fetch_predictions(path).empty


def test_save_and_fetch_round_trip(tmp_path):
    path = tmp_path / "t.db"
    assert db.save_predictions(_scored(), "Random Forest", "batch", db_path=path) == 3
    history = db.fetch_predictions(path)
    assert len(history) == 3
    assert set(history["model_name"]) == {"Random Forest"}
    assert history["distance_from_home_km"].isna().sum() == 1


def test_risk_level_filter(tmp_path):
    path = tmp_path / "t.db"
    db.save_predictions(_scored(), "Random Forest", "batch", db_path=path)
    high = db.fetch_predictions(path, risk_levels=["High"])
    assert len(high) == 1
    assert high.iloc[0]["transaction_id"] == "T2"


def test_empty_risk_filter_returns_nothing(tmp_path):
    path = tmp_path / "t.db"
    db.save_predictions(_scored(), "Random Forest", "batch", db_path=path)
    assert db.fetch_predictions(path, risk_levels=[]).empty


def test_source_filter(tmp_path):
    path = tmp_path / "t.db"
    db.save_predictions(_scored(), "Random Forest", "batch", db_path=path)
    db.save_predictions(_scored().head(1), "Random Forest", "manual", db_path=path)
    assert len(db.fetch_predictions(path, source="manual")) == 1
    assert len(db.fetch_predictions(path, source="batch")) == 3


def test_date_filter(tmp_path):
    path = tmp_path / "t.db"
    db.save_predictions(_scored(), "Random Forest", "batch", db_path=path)
    today = date.today()
    included = db.fetch_predictions(path, start_date=today.isoformat(),
                                    end_date=(today + timedelta(days=1)).isoformat())
    excluded = db.fetch_predictions(path, start_date=(today + timedelta(days=1)).isoformat())
    assert len(included) == 3
    assert excluded.empty


def test_summary_counts(tmp_path):
    path = tmp_path / "t.db"
    db.save_predictions(_scored(), "Random Forest", "batch", db_path=path)
    summary = db.get_summary(path)
    assert summary["total"] == 3
    assert summary["flagged"] == 2
    assert summary["high_risk"] == 1
    assert summary["total_amount"] == pytest.approx(365.5)


def test_summary_of_empty_database(tmp_path):
    summary = db.get_summary(tmp_path / "empty.db")
    assert summary["total"] == 0
    assert summary["avg_probability"] == 0.0


def test_delete_all(tmp_path):
    path = tmp_path / "t.db"
    db.save_predictions(_scored(), "Random Forest", "batch", db_path=path)
    assert db.delete_all_predictions(path) == 3
    assert db.get_summary(path)["total"] == 0


def test_sql_injection_text_is_treated_as_data(tmp_path):
    path = tmp_path / "t.db"
    db.save_predictions(_scored(), "Random Forest", "batch", db_path=path)
    assert db.fetch_predictions(path, source="x' OR '1'='1").empty


def test_missing_columns_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        db.save_predictions(_scored().drop(columns=["risk_level"]), "RF", "batch", db_path=tmp_path / "x.db")


def test_unknown_source_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        db.save_predictions(_scored(), "RF", "api", db_path=tmp_path / "x.db")
