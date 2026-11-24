"""Validation for manual inputs and uploaded CSV files.

Every value that comes from a user passes through this module before it reaches the model
or the database. Problem rows are removed with a visible warning; files that cannot be used
at all raise errors that the UI shows to the user.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config.settings import (
    MAX_AMOUNT,
    MAX_BATCH_ROWS,
    MAX_DISTANCE_KM,
    MAX_TX_LAST_24H,
    MAX_UPLOAD_MB,
    MERCHANT_CATEGORIES,
    REQUIRED_INPUT_COLUMNS,
)


class ValidationError(ValueError):
    """Raised when an uploaded file cannot be processed at all."""


_FLAG_TEXT = {
    "1": 1.0, "true": 1.0, "yes": 1.0, "y": 1.0, "t": 1.0,
    "0": 0.0, "false": 0.0, "no": 0.0, "n": 0.0, "f": 0.0,
}


@dataclass
class ValidationResult:
    data: pd.DataFrame
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors and not self.data.empty


def coerce_flag(value):
    """Return 1.0, 0.0 or NaN for common yes/no and true/false spellings."""
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value) if value in (0, 1) else np.nan
    if value is None or pd.isna(value):
        return np.nan
    return _FLAG_TEXT.get(str(value).strip().lower(), np.nan)


def _is_recognised_category(value):
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower().replace(" ", "_") in MERCHANT_CATEGORIES


def normalise_category(value):
    """Lower-case the category, or fall back to 'unknown' when it is missing or unrecognised."""
    if not _is_recognised_category(value):
        return "unknown"
    return str(value).strip().lower().replace(" ", "_")


def _non_negative(series):
    values = pd.to_numeric(series, errors="coerce").astype(float)
    return values.where(values >= 0)


def validate_manual_input(raw):
    """Validate one transaction typed into the form.

    Returns (cleaned_dict, []) when valid, or (None, errors) when not.
    """
    errors = []

    def number(key, low, high, label, whole=False):
        try:
            value = float(raw.get(key))
        except (TypeError, ValueError):
            errors.append(f"{label} must be a number.")
            return None
        if not np.isfinite(value) or not low <= value <= high:
            errors.append(f"{label} must be between {low:g} and {high:g}.")
            return None
        if whole and not value.is_integer():
            errors.append(f"{label} must be a whole number.")
            return None
        return value

    category = raw.get("merchant_category")
    if category not in MERCHANT_CATEGORIES:
        errors.append("Merchant category is not recognised.")
        category = None

    cleaned = {
        "amount": number("amount", 0.01, MAX_AMOUNT, "Transaction amount"),
        "hour_of_day": number("hour_of_day", 0, 23, "Hour of day", whole=True),
        "distance_from_home_km": number("distance_from_home_km", 0, MAX_DISTANCE_KM, "Distance from home"),
        "customer_avg_amount": number("customer_avg_amount", 0.01, MAX_AMOUNT, "Customer average amount"),
        "num_transactions_last_24h": number(
            "num_transactions_last_24h", 0, MAX_TX_LAST_24H, "Transactions in the last 24 hours", whole=True
        ),
        "card_present": coerce_flag(raw.get("card_present")),
        "foreign_transaction": coerce_flag(raw.get("foreign_transaction")),
        "merchant_category": category,
    }
    for key, label in (("card_present", "Card present"), ("foreign_transaction", "Foreign transaction")):
        if np.isnan(cleaned[key]):
            errors.append(f"{label} must be yes or no.")

    if errors or any(value is None for value in cleaned.values()):
        return None, errors or ["Please check the highlighted inputs."]
    return cleaned, []


def validate_upload_size(size_bytes):
    """Reject files larger than the configured limit."""
    limit = MAX_UPLOAD_MB * 1024 * 1024
    if size_bytes is not None and size_bytes > limit:
        raise ValidationError(
            f"The file is {size_bytes / (1024 * 1024):.1f} MB. The limit is {MAX_UPLOAD_MB} MB."
        )


def read_csv_safely(source):
    """Read a CSV from a path or file-like object, with friendly errors and normalised headers."""
    try:
        df = pd.read_csv(source)
    except pd.errors.EmptyDataError as exc:
        raise ValidationError("The file is empty.") from exc
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValidationError("The file could not be read as CSV. Save it as UTF-8 CSV and try again.") from exc
    if df.empty:
        raise ValidationError("The CSV contains no data rows.")
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def validate_batch(raw):
    """Validate an uploaded batch. Bad rows are dropped with a warning; unusable files give errors."""
    errors, warnings = [], []
    df = raw.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in df.columns]
    if missing:
        errors.append("Missing required column(s): " + ", ".join(missing))
        return ValidationResult(pd.DataFrame(), errors, warnings)
    if len(df) > MAX_BATCH_ROWS:
        errors.append(f"The file has {len(df):,} rows. The limit is {MAX_BATCH_ROWS:,}.")
        return ValidationResult(pd.DataFrame(), errors, warnings)

    amount = pd.to_numeric(df["amount"], errors="coerce").astype(float)
    hour = pd.to_numeric(df["hour_of_day"], errors="coerce").astype(float)
    invalid = (
        amount.isna() | (amount <= 0)
        | hour.isna() | (hour < 0) | (hour > 23) | (hour % 1 != 0)
    )
    if invalid.any():
        warnings.append(
            f"Removed {int(invalid.sum()):,} row(s) with a missing, non-positive or out-of-range "
            "amount or hour."
        )

    frame = pd.DataFrame(index=df.index)
    if "transaction_id" in df.columns:
        frame["transaction_id"] = df["transaction_id"].astype(str)
    else:
        frame["transaction_id"] = [f"ROW{i + 1:06d}" for i in range(len(df))]
    frame["amount"] = amount
    frame["hour_of_day"] = hour

    for column in ("distance_from_home_km", "customer_avg_amount", "num_transactions_last_24h"):
        frame[column] = _non_negative(df[column])
        gaps = int(frame[column].isna().sum())
        if gaps:
            warnings.append(f"{gaps:,} missing or invalid value(s) in {column} will be filled with typical values.")

    for column in ("card_present", "foreign_transaction"):
        flags = df[column].map(coerce_flag).astype(float)
        gaps = int(flags.isna().sum())
        if gaps:
            warnings.append(f"{gaps:,} unreadable value(s) in {column} will be filled with typical values.")
        frame[column] = flags

    categories = df["merchant_category"].map(normalise_category)
    raw_text = df["merchant_category"].astype(str).str.strip().str.lower()
    fallback = (categories == "unknown") & (raw_text != "unknown")
    if fallback.any():
        warnings.append(
            f"{int(fallback.sum()):,} missing or unrecognised merchant category value(s) were recorded as 'unknown'."
        )
    frame["merchant_category"] = categories

    frame = frame[~invalid].reset_index(drop=True)
    if frame.empty:
        errors.append("No valid rows remain after validation.")
    duplicates = int(frame["transaction_id"].duplicated().sum())
    if duplicates:
        warnings.append(f"{duplicates:,} duplicate transaction_id value(s) found. Every row is still scored.")
    return ValidationResult(frame, errors, warnings)
