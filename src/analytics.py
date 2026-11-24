"""Descriptive statistics for the transaction dataset. Pure pandas, no plotting."""
import pandas as pd

from config.settings import TARGET_COLUMN


def summarise_dataset(df):
    """Headline numbers for the dashboard."""
    total = int(len(df))
    fraud_mask = df[TARGET_COLUMN] == 1
    fraud = int(fraud_mask.sum())
    return {
        "total": total,
        "fraud": fraud,
        "legit": total - fraud,
        "fraud_pct": (fraud / total * 100) if total else 0.0,
        "total_amount": float(df["amount"].sum()),
        "fraud_amount": float(df.loc[fraud_mask, "amount"].sum()),
    }


def weekly_fraud_trend(df):
    """Fraud count, transaction count and fraud rate per calendar week."""
    columns = ["timestamp", "fraud", "total", "fraud_rate_pct"]
    if df.empty:
        return pd.DataFrame(columns=columns)
    
    indexed = df.assign(timestamp=pd.to_datetime(df["timestamp"])).set_index("timestamp").sort_index()
    weekly = indexed[TARGET_COLUMN].resample("W").agg(fraud="sum", total="count").reset_index()
    weekly = weekly[weekly["total"] > 0].copy()
    weekly["fraud_rate_pct"] = weekly["fraud"] / weekly["total"] * 100
    return weekly[columns]


def category_fraud_rates(df):
    """Fraud rate per merchant category, highest first."""
    categories = df["merchant_category"].fillna("unknown")
    grouped = (
        df.assign(merchant_category=categories)
        .groupby("merchant_category")[TARGET_COLUMN]
        .agg(fraud_count="sum", total="count")
        .reset_index()
    )
    grouped["fraud_rate_pct"] = grouped["fraud_count"] / grouped["total"] * 100
    return grouped.sort_values("fraud_rate_pct", ascending=False).reset_index(drop=True)
