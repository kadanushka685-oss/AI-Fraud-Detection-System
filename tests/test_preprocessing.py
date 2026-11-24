"""Tests for feature engineering, preprocessing and data splitting."""
import numpy as np
import pandas as pd
import pytest

from config.settings import REQUIRED_INPUT_COLUMNS
from src.features import FeatureEngineer
from src.model_pipeline import build_models, split_features_target, stratified_split


def make_row(**overrides):
    row = {
        "amount": 100.0, "hour_of_day": 3, "distance_from_home_km": 2.0,
        "card_present": 1, "foreign_transaction": 0, "num_transactions_last_24h": 2,
        "customer_avg_amount": 50.0, "merchant_category": "grocery",
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_engineered_features_are_computed():
    out = FeatureEngineer().fit_transform(make_row())
    assert out.loc[0, "amount_log"] == pytest.approx(np.log1p(100.0))
    assert out.loc[0, "amount_to_avg_ratio"] == pytest.approx(2.0)
    assert out.loc[0, "is_night"] == 1.0


def test_zero_customer_average_gives_missing_ratio():
    out = FeatureEngineer().transform(make_row(customer_avg_amount=0))
    assert np.isnan(out.loc[0, "amount_to_avg_ratio"])


def test_missing_hour_stays_missing_for_night_flag():
    out = FeatureEngineer().transform(make_row(hour_of_day=np.nan))
    assert np.isnan(out.loc[0, "is_night"])


def test_missing_category_becomes_unknown():
    out = FeatureEngineer().transform(make_row(merchant_category=None))
    assert out.loc[0, "merchant_category"] == "unknown"


def test_text_numbers_are_converted():
    out = FeatureEngineer().transform(make_row(amount="250.5"))
    assert out.loc[0, "amount"] == pytest.approx(250.5)


def test_feature_engineer_does_not_modify_input():
    original = make_row()
    FeatureEngineer().transform(original)
    assert "amount_log" not in original.columns


def test_identifiers_and_label_are_not_model_inputs(small_df):
    X, y = split_features_target(small_df)
    for leaked in ("is_fraud", "transaction_id", "timestamp", "customer_id"):
        assert leaked not in X.columns
    assert list(X.columns) == REQUIRED_INPUT_COLUMNS
    assert set(y.unique()) <= {0, 1}


def test_split_is_stratified_and_disjoint(small_df):
    X_train, X_test, y_train, y_test = stratified_split(small_df)
    assert len(X_train) + len(X_test) == len(small_df)
    assert set(X_train.index).isdisjoint(X_test.index)
    assert abs(float(y_train.mean()) - float(y_test.mean())) < 0.01


def test_pipeline_handles_missing_values(small_df):
    X_train, X_test, y_train, _ = stratified_split(small_df)
    pipeline = build_models()["Logistic Regression"].fit(X_train, y_train)
    X_bad = X_test.head(5).copy()
    X_bad["distance_from_home_km"] = np.nan
    X_bad["merchant_category"] = None
    proba = pipeline.predict_proba(X_bad)[:, 1]
    assert np.all(np.isfinite(proba))


def test_pipeline_handles_unseen_category(small_df):
    X_train, X_test, y_train, _ = stratified_split(small_df)
    pipeline = build_models()["Random Forest"].fit(X_train, y_train)
    X_new = X_test.head(3).copy()
    X_new["merchant_category"] = "crypto_exchange"
    proba = pipeline.predict_proba(X_new)[:, 1]
    assert np.all((proba >= 0) & (proba <= 1))


def test_missing_training_column_raises(small_df):
    with pytest.raises(ValueError):
        split_features_target(small_df.drop(columns=["amount"]))
