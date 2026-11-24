"""Synthetic transaction generator.

The build environment had no internet access, so a public card-fraud dataset could not be
downloaded. This module creates a labelled dataset with plausible fraud patterns: amounts that
deviate from the customer's usual spending, night-time activity, distant locations,
card-not-present use, foreign transactions and bursts of activity. Noise is added so the task
is not trivially separable. Nothing here reflects real banking data.
"""
import numpy as np
import pandas as pd

from config.settings import MERCHANT_CATEGORIES, RANDOM_STATE

# Category weights, in the same order as MERCHANT_CATEGORIES. Each set sums to 1.
LEGIT_CATEGORY_WEIGHTS = [0.20, 0.12, 0.12, 0.05, 0.12, 0.06, 0.10, 0.07, 0.06, 0.10, 0.00]
FRAUD_CATEGORY_WEIGHTS = [0.03, 0.05, 0.04, 0.08, 0.02, 0.03, 0.10, 0.22, 0.20, 0.23, 0.00]


def generate_synthetic_transactions(
    n_rows=20_000,
    fraud_rate=0.025,
    seed=RANDOM_STATE,
    start_date="2025-01-01",
    days=180,
    missing_rate=0.02,
):
    """Return a DataFrame of labelled synthetic transactions, sorted by timestamp."""
    if n_rows < 10:
        raise ValueError("n_rows must be at least 10.")
    if not 0 < fraud_rate < 1:
        raise ValueError("fraud_rate must be between 0 and 1 (exclusive).")
    if not 0 <= missing_rate < 1:
        raise ValueError("missing_rate must be between 0 and 1 (exclusive).")

    rng = np.random.default_rng(seed)
    n_fraud = max(1, int(round(n_rows * fraud_rate)))
    n_legit = n_rows - n_fraud
    is_fraud = np.zeros(n_rows, dtype=int)
    is_fraud[:n_fraud] = 1
    rng.shuffle(is_fraud)
    fraud = is_fraud == 1

    # Each customer has a personal spending level; fraud amounts deviate from it.
    n_customers = max(50, n_rows // 7)
    customer_base = rng.lognormal(mean=3.4, sigma=0.6, size=n_customers + 1)
    customer_id = rng.integers(1, n_customers + 1, size=n_rows)
    customer_avg_amount = customer_base[customer_id]

    amount = np.empty(n_rows)
    amount[~fraud] = customer_avg_amount[~fraud] * rng.lognormal(0.0, 0.5, n_legit)
    amount[fraud] = customer_avg_amount[fraud] * rng.lognormal(0.75, 0.7, n_fraud)

    hour = np.empty(n_rows, dtype=int)
    hour[~fraud] = np.where(
        rng.random(n_legit) < 0.85, rng.integers(7, 23, n_legit), rng.integers(0, 24, n_legit)
    )
    hour[fraud] = np.where(
        rng.random(n_fraud) < 0.30, rng.integers(0, 6, n_fraud), rng.integers(0, 24, n_fraud)
    )
    day_offset = rng.integers(0, days, size=n_rows)
    minute = rng.integers(0, 60, size=n_rows)
    timestamp = (
        pd.Timestamp(start_date)
        + pd.to_timedelta(day_offset, unit="D")
        + pd.to_timedelta(hour, unit="h")
        + pd.to_timedelta(minute, unit="min")
    )

    distance = np.empty(n_rows)
    distance[~fraud] = rng.exponential(6.0, n_legit)
    distance[fraud] = rng.exponential(22.0, n_fraud)

    card_present = np.empty(n_rows, dtype=int)
    card_present[~fraud] = rng.random(n_legit) < 0.72
    card_present[fraud] = rng.random(n_fraud) < 0.35

    foreign = np.empty(n_rows, dtype=int)
    foreign[~fraud] = rng.random(n_legit) < 0.03
    foreign[fraud] = rng.random(n_fraud) < 0.15

    tx_last_24h = np.empty(n_rows, dtype=int)
    tx_last_24h[~fraud] = rng.poisson(1.8, n_legit)
    tx_last_24h[fraud] = rng.poisson(2.8, n_fraud)

    category = np.empty(n_rows, dtype=object)
    category[~fraud] = rng.choice(MERCHANT_CATEGORIES, size=n_legit, p=LEGIT_CATEGORY_WEIGHTS)
    category[fraud] = rng.choice(MERCHANT_CATEGORIES, size=n_fraud, p=FRAUD_CATEGORY_WEIGHTS)

    df = pd.DataFrame({
        "transaction_id": [f"TXN{i:07d}" for i in range(n_rows)],
        "timestamp": timestamp.to_numpy(),
        "customer_id": customer_id,
        "merchant_category": category,
        "amount": amount.round(2),
        "hour_of_day": hour,
        "distance_from_home_km": distance.round(1),
        "card_present": card_present,
        "foreign_transaction": foreign,
        "num_transactions_last_24h": tx_last_24h,
        "customer_avg_amount": customer_avg_amount.round(2),
        "is_fraud": is_fraud,
    })

    # Real data has gaps; inject some so preprocessing is exercised.
    if missing_rate > 0:
        for column in (
            "distance_from_home_km", "customer_avg_amount",
            "num_transactions_last_24h", "merchant_category",
        ):
            df.loc[rng.random(n_rows) < missing_rate, column] = np.nan

    return df.sort_values("timestamp", kind="stable").reset_index(drop=True)
