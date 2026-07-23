import math

import pytest

from statflow.models.metrics import (
    classification_metrics,
    metrics_frame,
    regression_metrics,
)


def test_classification_perfect_prediction():
    y_true = [1, 1, 0, 0]
    y_prob = [0.99, 0.99, 0.01, 0.01]
    m = classification_metrics(y_true, y_prob)
    assert m.accuracy == 1.0
    assert m.brier == pytest.approx(0.0001, abs=1e-5)
    assert m.log_loss < 0.05


def test_classification_terrible_prediction():
    y_true = [1, 1, 0, 0]
    y_prob = [0.01, 0.01, 0.99, 0.99]
    m = classification_metrics(y_true, y_prob)
    assert m.accuracy == 0.0
    assert m.log_loss > 2.0  # very bad


def test_classification_calibrated_constant():
    """A constant 0.5 predictor has log loss ln(2) ≈ 0.693."""
    y_true = [1, 0, 1, 0, 1, 0]
    y_prob = [0.5] * 6
    m = classification_metrics(y_true, y_prob)
    assert m.log_loss == pytest.approx(math.log(2), abs=1e-3)
    assert m.brier == pytest.approx(0.25, abs=1e-3)


def test_regression_perfect_prediction():
    m = regression_metrics([5, 8, 6], [5, 8, 6])
    assert m.mae == 0.0
    assert m.rmse == 0.0
    assert m.within_1 == 1.0
    assert m.within_2 == 1.0


def test_regression_within_thresholds():
    y_true = [5, 5, 5, 5]
    y_pred = [5, 6, 7, 10]  # errors: 0, 1, 2, 5
    m = regression_metrics(y_true, y_pred)
    assert m.mae == 2.0
    assert m.within_1 == 0.5  # 0 and 1 are within 1
    assert m.within_2 == 0.75  # 0, 1, 2 are within 2


def test_metrics_frame_flattens_rows():
    rows = [
        {"name": "baseline", "metrics": {"mae": 3.0, "rmse": 4.0, "n": 100}},
        {"name": "xgb", "metrics": {"mae": 2.5, "rmse": 3.2, "n": 100}},
    ]
    df = metrics_frame(rows)
    assert list(df.columns) == ["model", "mae", "rmse", "n"]
    assert list(df["model"]) == ["baseline", "xgb"]
