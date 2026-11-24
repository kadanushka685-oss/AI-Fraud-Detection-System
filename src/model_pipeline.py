"""Preprocessing, model construction, training and evaluation.

All preprocessing (imputation, scaling, encoding) lives inside scikit-learn Pipelines and is
fitted only on the training split, so no statistics from the test set leak into training.
Identifiers (transaction_id, customer_id) and the timestamp are excluded from the features.
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config.settings import (
    CATEGORICAL_FEATURES,
    DECISION_THRESHOLD,
    MODEL_NUMERIC_FEATURES,
    RANDOM_STATE,
    REQUIRED_INPUT_COLUMNS,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.features import FeatureEngineer


def build_preprocessor():
    """Impute, scale numeric columns and one-hot encode merchant categories."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric, MODEL_NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])


def build_models(random_state=RANDOM_STATE):
    """Return the two candidate pipelines. class_weight='balanced' handles the rare fraud class."""
    def make_pipeline(estimator):
        return Pipeline([
            ("features", FeatureEngineer()),
            ("preprocess", build_preprocessor()),
            ("model", estimator),
        ])

    return {
        "Logistic Regression": make_pipeline(
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)
        ),
        "Random Forest": make_pipeline(
            RandomForestClassifier(
                n_estimators=150,
                max_depth=8,
                min_samples_leaf=10,
                class_weight="balanced_subsample",
                random_state=random_state,
                n_jobs=-1,
            )
        ),
    }


def split_features_target(df):
    """Select model inputs and the label. Identifiers and the timestamp are never inputs."""
    missing = [c for c in REQUIRED_INPUT_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"Training data is missing column(s): {', '.join(missing)}")
    return df[REQUIRED_INPUT_COLUMNS].copy(), df[TARGET_COLUMN].astype(int)


def stratified_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Split into train and test, keeping the fraud rate equal in both."""
    X, y = split_features_target(df)
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def _downsample(values, max_points):
    values = np.asarray(values, dtype=float)
    if len(values) > max_points:
        idx = np.linspace(0, len(values) - 1, max_points).round().astype(int)
        values = values[idx]
    return [round(float(v), 4) for v in values]


def curve_points(y_true, y_proba, max_points=100):
    """ROC and precision-recall points, downsampled so the saved report stays small."""
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return None
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    return {
        "fpr": _downsample(fpr, max_points),
        "tpr": _downsample(tpr, max_points),
        "precision": _downsample(precision, max_points),
        "recall": _downsample(recall, max_points),
    }


def compute_metrics(y_true, y_proba, threshold=DECISION_THRESHOLD):
    """Classification metrics at a threshold, plus threshold-free ranking metrics."""
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    has_both_classes = len(np.unique(y_true)) == 2
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        # ROC-AUC and PR-AUC need both classes present; otherwise they are undefined.
        "roc_auc": float(roc_auc_score(y_true, y_proba)) if has_both_classes else None,
        "pr_auc": float(average_precision_score(y_true, y_proba)) if has_both_classes else None,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def train_and_compare(df, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Train every candidate on the training split, evaluate on the test split, pick the best.

    The best model is chosen by PR-AUC, which is more informative than accuracy when fraud
    is rare: a model that always says 'legitimate' would still score 97%+ accuracy.
    """
    X_train, X_test, y_train, y_test = stratified_split(df, test_size, random_state)

    results, fitted = {}, {}
    for name, pipeline in build_models(random_state).items():
        pipeline.fit(X_train, y_train)
        proba = pipeline.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, proba)
        metrics["curves"] = curve_points(y_test, proba)
        results[name] = metrics
        fitted[name] = pipeline

    best_name = max(results, key=lambda name: results[name]["pr_auc"] or 0.0)
    summary = {
        "dataset_rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "test_fraud_cases": int(y_test.sum()),
        "fraud_rate_pct": round(float(df[TARGET_COLUMN].mean()) * 100, 3),
        "test_size": test_size,
        "random_state": random_state,
        "selection_metric": "pr_auc",
        "best_model": best_name,
        "models": results,
    }
    return {"best_model_name": best_name, "best_model": fitted[best_name], "summary": summary}
