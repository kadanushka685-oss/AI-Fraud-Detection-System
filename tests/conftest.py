"""Shared fixtures. The small dataset keeps the test suite fast."""
import pytest

from src.data_generator import generate_synthetic_transactions


@pytest.fixture(scope="session")
def small_df():
    return generate_synthetic_transactions(n_rows=2000, fraud_rate=0.06, seed=7)
