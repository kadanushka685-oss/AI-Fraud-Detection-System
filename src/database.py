"""SQLite storage for transaction details and prediction results.

All SQL uses parameter placeholders; user-supplied values are never formatted into SQL text.
"""
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import pandas as pd

from config.settings import DB_PATH, REQUIRED_INPUT_COLUMNS

TABLE = "predictions"
SOURCES = ("manual", "batch")
PREDICTION_COLUMNS = ["fraud_probability", "risk_level", "predicted_fraud"]
HISTORY_COLUMNS = [
    "created_at", "source", "transaction_id", "amount", "merchant_category",
    "hour_of_day", "distance_from_home_km", "card_present", "foreign_transaction",
    "num_transactions_last_24h", "customer_avg_amount", "model_name",
    "fraud_probability", "risk_level", "predicted_fraud",
]
_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    source TEXT NOT NULL,
    transaction_id TEXT NOT NULL,
    amount REAL NOT NULL,
    merchant_category TEXT NOT NULL,
    hour_of_day INTEGER NOT NULL,
    distance_from_home_km REAL,
    card_present INTEGER,
    foreign_transaction INTEGER,
    num_transactions_last_24h REAL,
    customer_avg_amount REAL,
    model_name TEXT NOT NULL,
    fraud_probability REAL NOT NULL,
    risk_level TEXT NOT NULL,
    predicted_fraud INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON {TABLE} (created_at);
"""


@contextmanager
def get_connection(db_path=DB_PATH):
    """Open a connection, commit on success and always close it."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path=DB_PATH):
    """Create the predictions table if it does not exist."""
    with get_connection(db_path) as conn:
        conn.executescript(_SCHEMA)


def _to_sqlite(value):
    """Convert numpy/pandas scalars to plain Python values SQLite understands; NaN becomes NULL."""
    if value is None or isinstance(value, str):
        return value
    if pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value


def save_predictions(predictions, model_name, source, db_path=DB_PATH, created_at=None):
    """Store scored transactions. Returns the number of rows written."""
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}.")
    required = ["transaction_id"] + REQUIRED_INPUT_COLUMNS + PREDICTION_COLUMNS
    missing = [c for c in required if c not in predictions.columns]
    if missing:
        raise ValueError(f"Predictions are missing column(s): {', '.join(missing)}")
    if predictions.empty:
        return 0

    init_db(db_path)
    stamp = created_at or datetime.now().isoformat(timespec="seconds")
    frame = predictions.assign(created_at=stamp, source=source, model_name=model_name)
    rows = [
        tuple(_to_sqlite(value) for value in row)
        for row in frame[HISTORY_COLUMNS].itertuples(index=False, name=None)
    ]
    placeholders = ", ".join("?" for _ in HISTORY_COLUMNS)
    sql = f"INSERT INTO {TABLE} ({', '.join(HISTORY_COLUMNS)}) VALUES ({placeholders})"
    with get_connection(db_path) as conn:
        conn.executemany(sql, rows)
    return len(rows)


def fetch_predictions(db_path=DB_PATH, risk_levels=None, start_date=None, end_date=None,
                      source=None, limit=None):
    """Return saved predictions, newest first, with optional filters.

    start_date is inclusive and end_date is exclusive; both are ISO date strings.
    """
    init_db(db_path)
    if risk_levels is not None and len(risk_levels) == 0:
        return pd.DataFrame(columns=["id"] + HISTORY_COLUMNS)

    clauses, params = [], []
    if risk_levels is not None:
        clauses.append(f"risk_level IN ({', '.join('?' for _ in risk_levels)})")
        params.extend(risk_levels)
    if start_date:
        clauses.append("created_at >= ?")
        params.append(str(start_date))
    if end_date:
        clauses.append("created_at < ?")
        params.append(str(end_date))
    if source:
        clauses.append("source = ?")
        params.append(source)

    sql = f"SELECT id, {', '.join(HISTORY_COLUMNS)} FROM {TABLE}"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY created_at DESC, id DESC"
    if limit:
        sql += " LIMIT ?"
        params.append(int(limit))

    with get_connection(db_path) as conn:
        return pd.read_sql_query(sql, conn, params=params)


def get_summary(db_path=DB_PATH):
    """Aggregate counts used on the dashboard."""
    init_db(db_path)
    sql = (
        "SELECT COUNT(*), "
        "COALESCE(SUM(predicted_fraud), 0), "
        "COALESCE(SUM(CASE WHEN risk_level = 'High' THEN 1 ELSE 0 END), 0), "
        "COALESCE(AVG(fraud_probability), 0.0), "
        "COALESCE(SUM(amount), 0.0) "
        f"FROM {TABLE}"
    )
    with get_connection(db_path) as conn:
        total, flagged, high, avg_probability, total_amount = conn.execute(sql).fetchone()
    return {
        "total": int(total),
        "flagged": int(flagged),
        "high_risk": int(high),
        "avg_probability": float(avg_probability),
        "total_amount": float(total_amount),
    }


def delete_all_predictions(db_path=DB_PATH):
    """Remove every saved prediction. Returns the number of rows deleted."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.execute(f"DELETE FROM {TABLE}")
        return cursor.rowcount
