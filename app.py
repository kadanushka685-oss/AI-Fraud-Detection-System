"""AI Fraud Detection System: Streamlit dashboard.

Run from the project folder with:  streamlit run app.py
"""
import json
import sys
import uuid
from datetime import date, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from config.settings import (  # noqa: E402
    DB_PATH,
    EVALUATION_PATH,
    MAX_AMOUNT,
    MAX_DISTANCE_KM,
    MAX_TX_LAST_24H,
    MAX_UPLOAD_MB,
    MERCHANT_CATEGORIES,
    REQUIRED_INPUT_COLUMNS,
    RISK_LEVELS,
    SAMPLE_DATA_PATH,
    TRAINING_DATA_PATH,
)
from src import database, validation  # noqa: E402
from src import visualization as viz  # noqa: E402
from src.analytics import category_fraud_rates, summarise_dataset, weekly_fraud_trend  # noqa: E402
from src.data_generator import generate_synthetic_transactions  # noqa: E402
from src.predictor import ModelNotFoundError, load_model, predict_transactions  # noqa: E402

st.set_page_config(page_title="AI Fraud Detection System", page_icon="🛡️", layout="wide")

DISCLAIMER = (
    "These results are model estimates based on patterns in historical data. They are not "
    "confirmed fraud findings and should be reviewed by a person before any action is taken."
)
RISK_TEXT_COLORS = {"Low": "green", "Medium": "orange", "High": "red"}
PROBABILITY_COLUMN = {
    "fraud_probability": st.column_config.ProgressColumn(
        "Fraud probability", min_value=0.0, max_value=1.0, format="%.2f"
    ),
}
HISTORY_VIEW_COLUMNS = [
    "created_at", "source", "transaction_id", "amount", "merchant_category",
    "fraud_probability", "risk_level", "predicted_fraud", "model_name",
]

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; max-width: 1200px; }
    div[data-testid="stMetric"] {
        background: #FFFFFF; border: 1px solid #DDE3E8; border-radius: 8px; padding: 14px 18px;
    }
    h1 { letter-spacing: -0.02em; font-weight: 650; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_model_bundle():
    return load_model()


@st.cache_data(show_spinner="Loading transaction data...")
def get_training_data():
    if TRAINING_DATA_PATH.exists():
        return pd.read_csv(TRAINING_DATA_PATH, parse_dates=["timestamp"])
    return generate_synthetic_transactions()


@st.cache_data(show_spinner=False)
def get_evaluation():
    if EVALUATION_PATH.exists():
        return json.loads(EVALUATION_PATH.read_text(encoding="utf-8"))
    return None


def require_model():
    """Return the model bundle, or show a clear error and return None."""
    try:
        return get_model_bundle()
    except ModelNotFoundError as exc:
        st.error(str(exc))
        return None


def _date_range(picked, today):
    if isinstance(picked, (tuple, list)) and len(picked) == 2:
        return picked[0], picked[1]
    if isinstance(picked, date):
        return picked, picked
    return today - timedelta(days=30), today


def show_prediction(row):
    probability = float(row["fraud_probability"])
    level = str(row["risk_level"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Estimated fraud probability", f"{probability:.1%}")
    c2.metric("Risk level", level)
    c3.metric("Model used", str(row["model_name"]))
    st.progress(probability)
    st.markdown(f"This transaction is :{RISK_TEXT_COLORS[level]}[{level} risk].")
    st.caption(DISCLAIMER)


def page_dashboard():
    st.title("Fraud Dashboard")
    st.caption("The training dataset at a glance, plus activity from predictions saved in this app.")

    data = get_training_data()
    stats = summarise_dataset(data)
    st.subheader("Training dataset")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total transactions", f"{stats['total']:,}")
    c2.metric("Fraudulent transactions", f"{stats['fraud']:,}")
    c3.metric("Legitimate transactions", f"{stats['legit']:,}")
    c4, c5, c6 = st.columns(3)
    c4.metric("Fraud percentage", f"{stats['fraud_pct']:.2f}%")
    c5.metric("Total transaction amount", f"{stats['total_amount']:,.2f}")
    c6.metric("Amount in fraudulent transactions", f"{stats['fraud_amount']:,.2f}")

    left, right = st.columns([1, 2])
    left.plotly_chart(viz.class_balance_chart(stats), use_container_width=True)
    right.plotly_chart(viz.weekly_trend_chart(weekly_fraud_trend(data)), use_container_width=True)

    st.subheader("Prediction activity")
    summary = database.get_summary(DB_PATH)
    c7, c8, c9, c10 = st.columns(4)
    c7.metric("Predictions saved", f"{summary['total']:,}")
    c8.metric("Flagged as fraud", f"{summary['flagged']:,}")
    c9.metric("High risk", f"{summary['high_risk']:,}")
    c10.metric("Average fraud probability", f"{summary['avg_probability']:.1%}")
    if summary["total"] == 0:
        st.info("No predictions saved yet. Try the Fraud Prediction page.")


def page_prediction():
    st.title("Fraud Prediction")
    st.caption("Enter one transaction to estimate how likely it is to be fraudulent.")
    bundle = require_model()

    with st.form("manual_prediction"):
        c1, c2, c3 = st.columns(3)
        amount = c1.number_input(
            "Transaction amount", min_value=0.01, max_value=MAX_AMOUNT, value=120.0, step=1.0, format="%.2f"
        )
        customer_avg = c2.number_input(
            "Customer's usual average amount", min_value=0.01, max_value=MAX_AMOUNT,
            value=85.0, step=1.0, format="%.2f",
        )
        hour = c3.slider("Hour of day", min_value=0, max_value=23, value=14)
        c4, c5, c6 = st.columns(3)
        distance = c4.number_input(
            "Distance from home (km)", min_value=0.0, max_value=MAX_DISTANCE_KM, value=5.0, step=1.0
        )
        tx_count = c5.number_input(
            "Transactions in the last 24 hours", min_value=0, max_value=MAX_TX_LAST_24H, value=2, step=1
        )
        category = c6.selectbox("Merchant category", [c for c in MERCHANT_CATEGORIES if c != "unknown"])
        c7, c8, c9 = st.columns(3)
        card_present = c7.checkbox("Card was present", value=True)
        foreign = c8.checkbox("Foreign transaction", value=False)
        save = c9.checkbox("Save to history", value=True)
        submitted = st.form_submit_button("Assess transaction", type="primary")

    if not submitted or bundle is None:
        return

    raw = {
        "amount": amount,
        "customer_avg_amount": customer_avg,
        "hour_of_day": hour,
        "distance_from_home_km": distance,
        "num_transactions_last_24h": tx_count,
        "merchant_category": category,
        "card_present": card_present,
        "foreign_transaction": foreign,
    }
    cleaned, errors = validation.validate_manual_input(raw)
    if errors:
        for message in errors:
            st.error(message)
        return

    transaction = pd.DataFrame([cleaned])
    transaction.insert(0, "transaction_id", f"MAN-{uuid.uuid4().hex[:10].upper()}")
    result = predict_transactions(bundle, transaction)
    show_prediction(result.iloc[0])
    if save:
        database.save_predictions(result, model_name=bundle["model_name"], source="manual")
        st.success("Saved to Transaction History.")


def page_batch():
    st.title("Batch CSV Analysis")
    st.caption(f"Upload a CSV of transactions to score them all at once. The limit is {MAX_UPLOAD_MB} MB.")
    st.markdown(
        "Required columns: " + ", ".join(f"`{c}`" for c in REQUIRED_INPUT_COLUMNS)
        + ". Optional: `transaction_id`, `timestamp`."
    )
    st.download_button(
        "Download sample CSV",
        data=SAMPLE_DATA_PATH.read_bytes(),
        file_name="sample_transactions.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV to begin, or download the sample file above to try the page.")
        return
    bundle = require_model()
    if bundle is None:
        return

    try:
        validation.validate_upload_size(uploaded.size)
        raw = validation.read_csv_safely(uploaded)
    except validation.ValidationError as exc:
        st.error(str(exc))
        return

    checked = validation.validate_batch(raw)
    for message in checked.warnings:
        st.warning(message)
    if not checked.is_valid:
        for message in checked.errors:
            st.error(message)
        return

    results = predict_transactions(bundle, checked.data)
    c1, c2, c3 = st.columns(3)
    c1.metric("Transactions analysed", f"{len(results):,}")
    c2.metric("Predicted fraud", f"{int(results['predicted_fraud'].sum()):,}")
    c3.metric("High risk", f"{int((results['risk_level'] == 'High').sum()):,}")
    risk_counts = results["risk_level"].value_counts().reindex(RISK_LEVELS, fill_value=0)
    st.plotly_chart(viz.risk_distribution_chart(risk_counts), use_container_width=True)

    view = results.sort_values("fraud_probability", ascending=False)[[
        "transaction_id", "amount", "merchant_category", "hour_of_day", "distance_from_home_km",
        "foreign_transaction", "fraud_probability", "risk_level",
    ]]
    st.dataframe(view, hide_index=True, column_config=PROBABILITY_COLUMN)
    st.download_button(
        "Download results (CSV)",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="fraud_predictions.csv",
        mime="text/csv",
    )

    saved_key = f"batch-saved::{uploaded.name}::{uploaded.size}"
    if st.session_state.get(saved_key):
        st.caption("These results are saved in Transaction History.")
    elif st.button("Save results to history"):
        database.save_predictions(results, model_name=bundle["model_name"], source="batch")
        st.session_state[saved_key] = True
        st.rerun()


def render_model_performance(evaluation):
    models = evaluation["models"]
    st.info(
        f"Best model by PR-AUC: **{evaluation['best_model']}**. The test split has "
        f"{evaluation['test_rows']:,} transactions, of which {evaluation['test_fraud_cases']} are fraud."
    )
    table = pd.DataFrame({
        name: {label: metrics[key] for key, label in viz.METRIC_LABELS.items()}
        for name, metrics in models.items()
    }).T
    st.dataframe(table.astype(float).round(3))
    st.plotly_chart(viz.model_comparison_chart(models), use_container_width=True)

    chosen = st.selectbox("Model for the confusion matrix", list(models))
    matrix = models[chosen]["confusion_matrix"]
    st.plotly_chart(viz.confusion_matrix_chart(matrix, chosen), use_container_width=True)
    st.caption(
        f"At a {models[chosen]['threshold']:.0%} threshold: {matrix['tp']} fraud cases caught, "
        f"{matrix['fn']} missed, and {matrix['fp']} legitimate transactions flagged."
    )


def page_analytics():
    st.title("Analytics")
    st.caption("Where fraud concentrates in the training data, and how each model performed on unseen data.")
    data = get_training_data()
    evaluation = get_evaluation()
    patterns_tab, performance_tab, curves_tab = st.tabs(
        ["Fraud patterns", "Model performance", "Ranking curves"]
    )

    with patterns_tab:
        st.plotly_chart(viz.weekly_trend_chart(weekly_fraud_trend(data)), use_container_width=True)
        left, right = st.columns(2)
        left.plotly_chart(viz.amount_box_chart(data), use_container_width=True)
        right.plotly_chart(viz.category_rate_chart(category_fraud_rates(data)), use_container_width=True)

    with performance_tab:
        if evaluation is None:
            st.warning("No model report found. Run `python scripts/train_model.py` first.")
        else:
            render_model_performance(evaluation)

    with curves_tab:
        if evaluation is None:
            st.warning("No model report found. Run `python scripts/train_model.py` first.")
        else:
            st.plotly_chart(viz.curves_chart(evaluation["models"]), use_container_width=True)


def page_history():
    st.title("Transaction History")
    st.caption("Every prediction saved from this app. Filter by risk level, source or date.")
    today = date.today()

    f1, f2, f3 = st.columns([2, 1, 2])
    risk_filter = f1.multiselect("Risk level", RISK_LEVELS, default=RISK_LEVELS)
    source_choice = f2.selectbox("Source", ["All", "manual", "batch"])
    picked = f3.date_input("Date range", value=(today - timedelta(days=30), today))
    start_day, end_day = _date_range(picked, today)

    with st.expander("Clear all history"):
        confirmed = st.checkbox("I understand this permanently deletes every saved prediction.")
        if st.button("Delete all history", disabled=not confirmed):
            deleted = database.delete_all_predictions(DB_PATH)
            st.success(f"Deleted {deleted:,} saved prediction(s).")

    history = database.fetch_predictions(
        DB_PATH,
        risk_levels=risk_filter,
        start_date=start_day.isoformat(),
        end_date=(end_day + timedelta(days=1)).isoformat(),
        source=None if source_choice == "All" else source_choice,
    )
    st.markdown(f"**{len(history):,}** matching prediction(s)")
    if history.empty:
        st.info("No predictions match these filters.")
        return
    st.dataframe(history[HISTORY_VIEW_COLUMNS], hide_index=True, column_config=PROBABILITY_COLUMN)
    st.download_button(
        "Download filtered history (CSV)",
        data=history.to_csv(index=False).encode("utf-8"),
        file_name="prediction_history.csv",
        mime="text/csv",
    )


def page_about():
    st.title("About Project")
    st.markdown(
        """
        ### Purpose
        This project estimates how likely a card transaction is to be fraudulent, using machine
        learning trained on historical transactions. It is a portfolio project that demonstrates a
        complete workflow: data preparation, model comparison, evaluation, a dashboard, batch
        scoring and persistent history.

        ### How it works
        1. Eight transaction fields are validated, then three features are derived: the log of the
           amount, the ratio of the amount to the customer's usual spending, and a night-time flag.
        2. Missing values are filled and categories are encoded inside a scikit-learn pipeline that is
           fitted only on training data, which prevents data leakage.
        3. Logistic Regression and Random Forest are trained with balanced class weights, because
           fraud is rare. The model with the higher PR-AUC on held-out data is saved.
        4. Each probability maps to a Low, Medium or High risk band.

        ### Data
        The build environment had no internet access, so the training data is a **synthetic
        dataset** generated by `src/data_generator.py`. Its patterns are plausible but invented.
        The metrics describe that dataset and must not be read as real banking performance.

        ### Limitations
        - Synthetic data cannot reproduce every real fraud pattern.
        - The test split contains only about 100 fraud cases, so the metrics carry noticeable uncertainty.
        - The model sees only the eight provided fields. It has no real customer history or device data.
        - The 50% decision threshold is fixed. A real deployment would tune it to business costs.

        ### Privacy
        No card numbers, names or account identifiers are needed or stored. Saved predictions live in a
        local SQLite file (`data/transactions.db`) on the machine running the app.

        ### Technology
        Python, pandas, NumPy, scikit-learn, Streamlit, Plotly, SQLite and joblib.
        """
    )


PAGES = {
    "Dashboard": page_dashboard,
    "Fraud Prediction": page_prediction,
    "Batch CSV Analysis": page_batch,
    "Analytics": page_analytics,
    "Transaction History": page_history,
    "About Project": page_about,
}

with st.sidebar:
    st.markdown("## 🛡️ Fraud Detection")
    selected = st.radio("Navigation", list(PAGES), label_visibility="collapsed")
    st.divider()
    st.caption("Scores are model estimates, not confirmed fraud findings.")

PAGES[selected]()
