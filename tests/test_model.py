"""Tests for training, evaluation, model persistence and scoring."""
import numpy as np
import pytest

from config.settings import DECISION_THRESHOLD, MODEL_PATH, REQUIRED_INPUT_COLUMNS
from src.model_pipeline import compute_metrics, curve_points, train_and_compare
from src.predictor import (
    ModelNotFoundError,
    load_model,
    predict_transactions,
    risk_level,
    save_model_bundle,
)


@pytest.fixture(scope="module")
def trained(small_df):
    return train_and_compare(small_df)


def _bundle(trained):
    return {
        "pipeline": trained["best_model"],
        "model_name": trained["best_model_name"],
        "features": REQUIRED_INPUT_COLUMNS,
    }


def test_both_models_are_compared(trained):
    assert set(trained["summary"]["models"]) == {"Logistic Regression", "Random Forest"}
    assert trained["best_model_name"] in trained["summary"]["models"]


def test_metrics_are_within_valid_ranges(trained):
    for name, metrics in trained["summary"]["models"].items():
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"):
            assert 0.0 <= metrics[key] <= 1.0, f"{name}.{key}"


def test_confusion_matrix_totals_match_test_size(trained):
    summary = trained["summary"]
    for metrics in summary["models"].values():
        cm = metrics["confusion_matrix"]
        assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == summary["test_rows"]


def test_best_model_has_highest_pr_auc(trained):
    models = trained["summary"]["models"]
    best = trained["best_model_name"]
    assert models[best]["pr_auc"] == max(m["pr_auc"] for m in models.values())


def test_models_learn_real_signal(trained):
    best = trained["summary"]["models"][trained["best_model_name"]]
    assert best["roc_auc"] > 0.8


def test_predictions_are_valid(trained, small_df):
    scored = predict_transactions(_bundle(trained), small_df.head(50))
    assert len(scored) == 50
    assert scored["fraud_probability"].between(0, 1).all()
    assert set(scored["risk_level"]) <= {"Low", "Medium", "High"}
    expected_label = (scored["fraud_probability"] >= DECISION_THRESHOLD).astype(int)
    assert (scored["predicted_fraud"] == expected_label).all()
    assert (scored["model_name"] == trained["best_model_name"]).all()


def test_prediction_rejects_missing_columns(trained, small_df):
    with pytest.raises(ValueError):
        predict_transactions(_bundle(trained), small_df.drop(columns=["amount"]).head(5))


def test_prediction_rejects_empty_input(trained, small_df):
    with pytest.raises(ValueError):
        predict_transactions(_bundle(trained), small_df.head(0))


def test_risk_band_boundaries():
    assert risk_level(0.0) == "Low"
    assert risk_level(0.299) == "Low"
    assert risk_level(0.30) == "Medium"
    assert risk_level(0.699) == "Medium"
    assert risk_level(0.70) == "High"
    assert risk_level(1.0) == "High"


def test_metrics_with_single_class_do_not_crash():
    metrics = compute_metrics([0, 0, 0], [0.1, 0.2, 0.3])
    assert metrics["roc_auc"] is None
    assert metrics["pr_auc"] is None
    assert metrics["confusion_matrix"] == {"tn": 3, "fp": 0, "fn": 0, "tp": 0}


def test_perfect_ranking_scores_one():
    metrics = compute_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)


def test_curve_points_are_downsampled():
    y = np.array([0, 1] * 500)
    scores = np.linspace(0, 1, 1000)
    curves = curve_points(y, scores, max_points=50)
    assert 0 < len(curves["fpr"]) <= 50
    assert curve_points(np.zeros(10), np.linspace(0, 1, 10)) is None


def test_bundle_round_trip(trained, small_df, tmp_path):
    path = save_model_bundle(trained["best_model"], trained["best_model_name"], path=tmp_path / "model.joblib")
    loaded = load_model(path)
    assert loaded["model_name"] == trained["best_model_name"]
    assert loaded["features"] == REQUIRED_INPUT_COLUMNS
    original = predict_transactions(_bundle(trained), small_df.head(20))["fraud_probability"].to_numpy()
    restored = predict_transactions(loaded, small_df.head(20))["fraud_probability"].to_numpy()
    np.testing.assert_allclose(original, restored)


def test_missing_model_file_has_helpful_error(tmp_path):
    with pytest.raises(ModelNotFoundError, match="train_model"):
        load_model(tmp_path / "missing.joblib")


def test_shipped_model_bundle_is_usable(small_df):
    bundle = load_model(MODEL_PATH)
    scored = predict_transactions(bundle, small_df.head(10))
    assert len(scored) == 10
