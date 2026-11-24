"""Tests for the synthetic transaction generator."""
import pandas as pd
import pytest

from config.settings import MERCHANT_CATEGORIES, TARGET_COLUMN
from src.data_generator import generate_synthetic_transactions

EXPECTED_COLUMNS = {
    "transaction_id", "timestamp", "customer_id", "merchant_category", "amount",
    "hour_of_day", "distance_from_home_km", "card_present", "foreign_transaction",
    "num_transactions_last_24h", "customer_avg_amount", TARGET_COLUMN,
}


def test_schema_and_row_count(small_df):
    assert EXPECTED_COLUMNS.issubset(small_df.columns)
    assert len(small_df) == 2000


def test_fraud_rate_matches_request(small_df):
    assert abs(float(small_df[TARGET_COLUMN].mean()) - 0.06) < 0.001


def test_values_are_in_expected_ranges(small_df):
    assert (small_df["amount"].dropna() > 0).all()
    assert small_df["hour_of_day"].between(0, 23).all()
    assert small_df["card_present"].isin([0, 1]).all()
    assert small_df["foreign_transaction"].isin([0, 1]).all()
    assert set(small_df["merchant_category"].dropna()) <= set(MERCHANT_CATEGORIES)


def test_missing_values_are_injected(small_df):
    assert small_df["distance_from_home_km"].isna().any()
    assert small_df["merchant_category"].isna().any()


def test_fraud_differs_from_legitimate(small_df):
    fraud = small_df[small_df[TARGET_COLUMN] == 1]
    legit = small_df[small_df[TARGET_COLUMN] == 0]
    assert fraud["foreign_transaction"].mean() > legit["foreign_transaction"].mean()
    assert fraud["card_present"].mean() < legit["card_present"].mean()
    assert fraud["amount"].median() > legit["amount"].median()


def test_generation_is_reproducible():
    first = generate_synthetic_transactions(n_rows=300, seed=1)
    second = generate_synthetic_transactions(n_rows=300, seed=1)
    pd.testing.assert_frame_equal(first, second)


def test_timestamps_are_sorted(small_df):
    assert small_df["timestamp"].is_monotonic_increasing


@pytest.mark.parametrize("kwargs", [
    {"n_rows": 5},
    {"fraud_rate": 1.5},
    {"fraud_rate": 0},
    {"missing_rate": -0.1},
])
def test_invalid_arguments_are_rejected(kwargs):
    with pytest.raises(ValueError):
        generate_synthetic_transactions(**kwargs)
