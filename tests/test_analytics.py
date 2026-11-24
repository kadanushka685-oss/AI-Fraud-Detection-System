"""Tests for dataset statistics used by the dashboard and analytics pages."""
import pandas as pd
import pytest

from src.analytics import category_fraud_rates, summarise_dataset, weekly_fraud_trend


def test_summary_counts(small_df):
    summary = summarise_dataset(small_df)
    assert summary["total"] == len(small_df)
    assert summary["fraud"] + summary["legit"] == summary["total"]
    assert summary["fraud_pct"] == pytest.approx(summary["fraud"] / summary["total"] * 100)
    assert summary["fraud_amount"] <= summary["total_amount"]


def test_weekly_trend_covers_every_transaction(small_df):
    trend = weekly_fraud_trend(small_df)
    assert trend["total"].sum() == len(small_df)
    assert trend["fraud"].sum() == small_df["is_fraud"].sum()
    assert trend["fraud_rate_pct"].between(0, 100).all()


def test_weekly_trend_of_empty_frame_is_empty():
    empty = pd.DataFrame(columns=["timestamp", "is_fraud"])
    assert weekly_fraud_trend(empty).empty


def test_category_rates(small_df):
    rates = category_fraud_rates(small_df)
    assert rates["total"].sum() == len(small_df)
    assert rates["fraud_rate_pct"].is_monotonic_decreasing
    assert "unknown" in set(rates["merchant_category"])
