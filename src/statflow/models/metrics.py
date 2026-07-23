"""Evaluation metrics for winner classification and total-runs regression.

Deliberately opinionated:
- Winner uses log loss (proper scoring rule) as the primary metric —
  accuracy alone doesn't punish over-confidence, and a well-calibrated
  55% is more useful than a poorly-calibrated 70%.
- Total runs uses MAE as primary — intuitive units ("off by X runs").
- Both dataclasses include `n` so downstream reporting can spot
  sample-size differences between splits.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss


@dataclass
class ClassificationMetrics:
    n: int
    accuracy: float
    log_loss: float
    brier: float

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class RegressionMetrics:
    n: int
    mae: float
    rmse: float
    within_1: float  # fraction of predictions within +/- 1 run
    within_2: float

    def as_dict(self) -> dict:
        return asdict(self)


def classification_metrics(y_true, y_prob) -> ClassificationMetrics:
    """Compute winner-classification metrics from probabilities."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= 0.5).astype(int)

    return ClassificationMetrics(
        n=len(y_true),
        accuracy=float((y_pred == y_true).mean()),
        log_loss=float(log_loss(y_true, y_prob, labels=[0, 1])),
        brier=float(brier_score_loss(y_true, y_prob)),
    )


def regression_metrics(y_true, y_pred) -> RegressionMetrics:
    """Compute regression metrics for total-runs predictions."""
    y_true = np.asarray(y_true).astype(float)
    y_pred = np.asarray(y_pred).astype(float)
    err = np.abs(y_true - y_pred)

    return RegressionMetrics(
        n=len(y_true),
        mae=float(err.mean()),
        rmse=float(np.sqrt(((y_true - y_pred) ** 2).mean())),
        within_1=float((err <= 1).mean()),
        within_2=float((err <= 2).mean()),
    )


def metrics_frame(rows: list[dict]) -> pd.DataFrame:
    """Convert a list of {'name': ..., 'metrics': dict} into a comparison DataFrame."""
    return pd.DataFrame([{"model": r["name"], **r["metrics"]} for r in rows])
