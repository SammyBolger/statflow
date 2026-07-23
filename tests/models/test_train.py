"""Tests for the winner-classifier training functions.

We test `fit_and_evaluate_classifier` — the pure core. The MLflow wrappers
(`train_logistic_regression`, `train_xgb_classifier`) are thin composition
layers; if the core works, they work.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge

from statflow.models.data import FEATURE_COLS, Split
from statflow.models.train import fit_and_evaluate_classifier, fit_and_evaluate_regressor


def _synthetic_features(n: int, seed: int = 0) -> pd.DataFrame:
    """Build a Split-shaped DataFrame the training core can consume."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for i in range(n):
        row = {
            "game_pk": i,
            "season": 2024,
            "target_home_win": bool(rng.random() > 0.5),
            "target_total_runs": int(rng.integers(0, 20)),
        }
        for col in FEATURE_COLS:
            row[col] = float(rng.normal(loc=5.0, scale=2.0))
        rows.append(row)
    return pd.DataFrame(rows)


def _split(train_n: int = 60, val_n: int = 20, test_n: int = 20) -> Split:
    return Split(
        train=_synthetic_features(train_n, seed=1),
        val=_synthetic_features(val_n, seed=2),
        test=_synthetic_features(test_n, seed=3),
    )


def test_fit_and_evaluate_returns_metrics_for_each_split():
    split = _split()
    model = LogisticRegression(max_iter=1000)

    model, metrics = fit_and_evaluate_classifier(model, split)

    assert set(metrics.keys()) == {"train", "val", "test"}
    for name, m in metrics.items():
        assert m.n > 0, f"empty metrics for {name}"
        assert 0.0 <= m.accuracy <= 1.0
        assert m.log_loss > 0.0
        assert 0.0 <= m.brier <= 1.0


def test_fit_and_evaluate_skips_empty_splits():
    """A split with no rows shouldn't produce a metrics entry."""
    split = Split(
        train=_synthetic_features(60, seed=1),
        val=_synthetic_features(20, seed=2),
        test=pd.DataFrame(columns=_synthetic_features(1).columns),
    )
    model = LogisticRegression(max_iter=1000)

    _, metrics = fit_and_evaluate_classifier(model, split)

    assert set(metrics.keys()) == {"train", "val"}
    assert "test" not in metrics


def test_probabilities_are_in_unit_interval():
    split = _split()
    model = LogisticRegression(max_iter=1000)
    model, _ = fit_and_evaluate_classifier(model, split)

    probs = model.predict_proba(split.val[FEATURE_COLS])[:, 1]
    assert (probs >= 0.0).all()
    assert (probs <= 1.0).all()


def test_fit_and_evaluate_beats_naive_baseline_on_separable_data():
    """If the target is a deterministic function of one feature, LR should
    achieve near-perfect accuracy on val — a real signal that training
    actually happened."""
    n = 200
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n):
        # Make target perfectly separable by home_win_pct_l10
        pct = rng.random()
        row = {
            "game_pk": i,
            "season": 2024,
            "target_home_win": pct > 0.5,
            "target_total_runs": 8,
        }
        for col in FEATURE_COLS:
            row[col] = float(rng.normal(loc=5.0, scale=2.0))
        row["home_win_pct_l10"] = pct
        rows.append(row)

    all_df = pd.DataFrame(rows)
    split = Split(
        train=all_df.iloc[:120].reset_index(drop=True),
        val=all_df.iloc[120:160].reset_index(drop=True),
        test=all_df.iloc[160:].reset_index(drop=True),
    )
    model = LogisticRegression(max_iter=1000)
    _, metrics = fit_and_evaluate_classifier(model, split)

    # Deterministic separation → LR should crush a naive 0.5 baseline.
    assert metrics["val"].accuracy >= 0.85


# ---------------------------------------------------------------------------
# Regression tests
# ---------------------------------------------------------------------------


def test_fit_and_evaluate_regressor_returns_metrics_for_each_split():
    split = _split()
    model = Ridge(alpha=1.0)

    model, metrics = fit_and_evaluate_regressor(model, split)

    assert set(metrics.keys()) == {"train", "val", "test"}
    for name, m in metrics.items():
        assert m.n > 0, f"empty metrics for {name}"
        assert m.mae >= 0.0
        assert m.rmse >= m.mae  # RMSE is always >= MAE
        assert 0.0 <= m.within_1 <= 1.0
        assert 0.0 <= m.within_2 <= 1.0


def test_regressor_skips_empty_splits():
    split = Split(
        train=_synthetic_features(60, seed=1),
        val=_synthetic_features(20, seed=2),
        test=pd.DataFrame(columns=_synthetic_features(1).columns),
    )
    model = Ridge(alpha=1.0)
    _, metrics = fit_and_evaluate_regressor(model, split)
    assert set(metrics.keys()) == {"train", "val"}


def test_regressor_beats_mean_baseline_on_separable_data():
    """Total runs is a deterministic linear function of one feature — Ridge
    should hit near-zero MAE."""
    n = 200
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n):
        park = rng.random()  # in [0, 1)
        row = {
            "game_pk": i,
            "season": 2024,
            "target_home_win": True,
            "target_total_runs": int(round(10 * park + 3)),
        }
        for col in FEATURE_COLS:
            row[col] = float(rng.normal(loc=5.0, scale=2.0))
        row["venue_park_factor_runs"] = park
        rows.append(row)

    all_df = pd.DataFrame(rows)
    split = Split(
        train=all_df.iloc[:120].reset_index(drop=True),
        val=all_df.iloc[120:160].reset_index(drop=True),
        test=all_df.iloc[160:].reset_index(drop=True),
    )
    _, metrics = fit_and_evaluate_regressor(Ridge(alpha=0.01), split)

    # Mean-baseline MAE is roughly the standard deviation of the target.
    # A model that learned the linear relationship should be dramatically better.
    assert metrics["val"].mae < 1.0
