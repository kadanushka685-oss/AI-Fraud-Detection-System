"""Feature engineering as a scikit-learn transformer.

The transformer is stateless: fit() learns nothing from the data, so it cannot leak
information from the test split. It only derives columns from each row's own values.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from config.settings import RAW_NUMERIC_FEATURES


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Coerce types, normalise categories and add derived features."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        for column in RAW_NUMERIC_FEATURES:
            X[column] = pd.to_numeric(X[column], errors="coerce").astype(float)

        category = X["merchant_category"]
        X["merchant_category"] = (
            category.where(category.notna(), "unknown").astype(str).str.strip().str.lower()
        )

        amount = X["amount"]
        customer_avg = X["customer_avg_amount"].where(X["customer_avg_amount"] > 0)
        X["amount_log"] = np.log1p(amount.clip(lower=0))
        X["amount_to_avg_ratio"] = amount / customer_avg

        hour = X["hour_of_day"]
        X["is_night"] = (hour < 6).astype(float).where(hour.notna())
        return X
