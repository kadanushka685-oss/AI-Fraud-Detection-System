"""Central configuration: file paths, feature lists, thresholds and input limits."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
SCREENSHOTS_DIR = PROJECT_ROOT / "screenshots"

TRAINING_DATA_PATH = DATA_DIR / "synthetic_transactions.csv"
SAMPLE_DATA_PATH = DATA_DIR / "sample_transactions.csv"
DB_PATH = DATA_DIR / "transactions.db"
MODEL_PATH = MODELS_DIR / "fraud_model.joblib"
EVALUATION_PATH = MODELS_DIR / "evaluation.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2
DECISION_THRESHOLD = 0.5  # probability at or above which a transaction is labelled fraud

LOW_RISK_MAX = 0.30     # probability below this is "Low"
MEDIUM_RISK_MAX = 0.70  # probability below this is "Medium"; otherwise "High"
RISK_LEVELS = ["Low", "Medium", "High"]

MAX_UPLOAD_MB = 5
MAX_BATCH_ROWS = 50_000
MAX_AMOUNT = 1_000_000.0
MAX_DISTANCE_KM = 20_000.0
MAX_TX_LAST_24H = 1_000

TARGET_COLUMN = "is_fraud"
MERCHANT_CATEGORIES = [
    "grocery", "fuel", "restaurants", "clothing", "utilities", "healthcare",
    "entertainment", "electronics", "travel", "online_retail", "unknown",
]

# Raw inputs the user provides (or that a CSV must contain).
RAW_NUMERIC_FEATURES = [
    "amount", "hour_of_day", "distance_from_home_km", "card_present",
    "foreign_transaction", "num_transactions_last_24h", "customer_avg_amount",
]
CATEGORICAL_FEATURES = ["merchant_category"]
REQUIRED_INPUT_COLUMNS = RAW_NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Features derived from the raw inputs by src/features.py.
ENGINEERED_FEATURES = ["amount_log", "amount_to_avg_ratio", "is_night"]
MODEL_NUMERIC_FEATURES = RAW_NUMERIC_FEATURES + ENGINEERED_FEATURES
