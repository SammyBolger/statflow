"""Naive baselines. Every ML model must beat these — that's the point.

- HomeAlwaysWinsBaseline: predicts the home-team win rate from training.
  Historically ~54% in MLB. This is the "did the model learn anything?" bar.
- MeanRunsBaseline: predicts training mean of total_runs for every game.
  RMSE of this baseline is the standard deviation of total_runs — any
  regressor doing worse hasn't learned anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class HomeAlwaysWinsBaseline:
    """Constant home-win probability learned from training data."""

    def __init__(self) -> None:
        self.p: float | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> HomeAlwaysWinsBaseline:
        self.p = float(np.asarray(y).mean())
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.p is None:
            raise RuntimeError("Call fit() before predict_proba().")
        return np.full(len(X), self.p, dtype=float)


class MeanRunsBaseline:
    """Constant total-runs prediction: the training mean."""

    def __init__(self) -> None:
        self.mean: float | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> MeanRunsBaseline:
        self.mean = float(np.asarray(y).mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.mean is None:
            raise RuntimeError("Call fit() before predict().")
        return np.full(len(X), self.mean, dtype=float)
