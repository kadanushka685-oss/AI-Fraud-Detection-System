"""Tests for manual-input and CSV validation."""
import io

import numpy as np
import pandas as pd
import pytest

from config.settings import MAX_BATCH_ROWS
from src import validation as v

VALID_MANUAL = {
    "amount": "120.5", "hour_of_day": 14, "distance_from_home_km": 3,
    "customer_avg_amount": 80, "num_transactions_last_24h": 2,
    "card_present": True, "foreign_transaction": False, "merchant_category": "grocery",
}


def _valid_batch():
    return pd.DataFrame([
        {"transaction_id": "A1", "amount": 50.0, "merchant_category": "grocery", "hour_of_day": 10,
         "distance_from_home_km": 2.0, "card_present": "yes", "foreign_transaction": "no",
         "num_transactions_last_24h": 1, "customer_avg_amount": 45.0},
        {"transaction_id": "B2", "amount": 900.0, "merchant_category": "Electronics", "hour_of_day": 2,
         "distance_from_home_km": 40.0, "card_present": "no", "foreign_transaction": "true",
         "num_transactions_last_24h": 6, "customer_avg_amount": 60.0},
    ])


def test_valid_manual_input_is_cleaned():
    cleaned, errors = v.validate_manual_input(VALID_MANUAL)
    assert errors == []
    assert cleaned["amount"] == pytest.approx(120.5)
    assert cleaned["card_present"] == 1.0
    assert cleaned["foreign_transaction"] == 0.0


@pytest.mark.parametrize("field,value", [
    ("amount", -5),
    ("amount", "abc"),
    ("amount", float("nan")),
    ("hour_of_day", 24),
    ("hour_of_day", 2.5),
    ("distance_from_home_km", -1),
    ("num_transactions_last_24h", 5000),
    ("merchant_category", "casino"),
])
def test_invalid_manual_inputs_are_rejected(field, value):
    cleaned, errors = v.validate_manual_input({**VALID_MANUAL, field: value})
    assert cleaned is None
    assert errors


def test_missing_manual_field_is_reported():
    data = {key: value for key, value in VALID_MANUAL.items() if key != "amount"}
    cleaned, errors = v.validate_manual_input(data)
    assert cleaned is None
    assert any("Transaction amount" in message for message in errors)


def test_flag_coercion():
    assert v.coerce_flag("Yes") == 1.0
    assert v.coerce_flag("no") == 0.0
    assert v.coerce_flag(1) == 1.0
    assert np.isnan(v.coerce_flag("maybe"))
    assert np.isnan(v.coerce_flag(None))
    assert np.isnan(v.coerce_flag(7))


def test_category_normalisation():
    assert v.normalise_category(" Online Retail ") == "online_retail"
    assert v.normalise_category("casino") == "unknown"
    assert v.normalise_category(None) == "unknown"


def test_upload_size_limit():
    with pytest.raises(v.ValidationError):
        v.validate_upload_size(10 * 1024 * 1024)
    v.validate_upload_size(1024)


def test_valid_batch_passes_and_is_normalised():
    result = v.validate_batch(_valid_batch())
    assert result.is_valid
    assert result.errors == []
    assert list(result.data["card_present"]) == [1.0, 0.0]
    assert list(result.data["foreign_transaction"]) == [0.0, 1.0]
    assert list(result.data["merchant_category"]) == ["grocery", "electronics"]


def test_batch_missing_column_is_an_error():
    result = v.validate_batch(_valid_batch().drop(columns=["amount"]))
    assert not result.is_valid
    assert any("amount" in message for message in result.errors)


def test_batch_drops_invalid_rows_with_warning():
    bad = _valid_batch().iloc[[0]].copy()
    bad["amount"] = -10.0
    result = v.validate_batch(pd.concat([_valid_batch(), bad], ignore_index=True))
    assert len(result.data) == 2
    assert any("Removed 1 row" in message for message in result.warnings)


def test_batch_with_no_valid_rows_is_an_error():
    bad = _valid_batch()
    bad["hour_of_day"] = 99
    result = v.validate_batch(bad)
    assert not result.is_valid
    assert any("No valid rows" in message for message in result.errors)


def test_batch_missing_values_are_kept_and_warned():
    df = _valid_batch()
    df.loc[0, "distance_from_home_km"] = np.nan
    df.loc[1, "merchant_category"] = None
    result = v.validate_batch(df)
    assert len(result.data) == 2
    assert result.data["distance_from_home_km"].isna().sum() == 1
    assert result.data.loc[1, "merchant_category"] == "unknown"
    assert result.warnings


def test_batch_generates_ids_when_missing():
    result = v.validate_batch(_valid_batch().drop(columns=["transaction_id"]))
    assert list(result.data["transaction_id"]) == ["ROW000001", "ROW000002"]


def test_batch_row_limit_is_enforced():
    big = pd.concat([_valid_batch()] * (MAX_BATCH_ROWS // 2 + 1), ignore_index=True)
    result = v.validate_batch(big)
    assert not result.is_valid
    assert any("limit" in message for message in result.errors)


def test_read_csv_rejects_empty_and_header_only_files():
    with pytest.raises(v.ValidationError):
        v.read_csv_safely(io.BytesIO(b""))
    with pytest.raises(v.ValidationError):
        v.read_csv_safely(io.BytesIO(b"amount,hour_of_day\n"))


def test_read_csv_normalises_headers():
    df = v.read_csv_safely(io.BytesIO(b" Amount ,Hour_Of_Day\n10,5\n"))
    assert list(df.columns) == ["amount", "hour_of_day"]
