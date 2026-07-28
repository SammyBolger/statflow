"""Tests for the per-prediction explanation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from statflow.models.explanations import (
    contributions_frame,
    top_contributions,
)


def _fit_toy_model():
    """A tiny XGBoost model on separable data — for verifying contrib shapes."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "signal": rng.uniform(0, 1, 200),
            "noise_a": rng.normal(0, 1, 200),
            "noise_b": rng.normal(0, 1, 200),
        }
    )
    y = (X["signal"] > 0.5).astype(int)
    model = XGBClassifier(n_estimators=20, max_depth=3, random_state=42)
    model.fit(X, y)
    return model, list(X.columns)


def test_top_contributions_returns_top_n_per_row():
    model, feats = _fit_toy_model()
    X = pd.DataFrame({f: [0.5] for f in feats})

    result = top_contributions(model, X, feats, top_n=2)
    assert len(result) == 1  # one row
    assert len(result[0]) == 2  # top 2 features
    # Should be sorted by |contribution| descending
    assert abs(result[0][0][1]) >= abs(result[0][1][1])


def test_top_contributions_ranks_signal_first_on_separable_data():
    """For a model trained where 'signal' fully determines y, 'signal' should
    have the largest absolute contribution on any prediction."""
    model, feats = _fit_toy_model()
    X = pd.DataFrame({"signal": [0.9], "noise_a": [0.0], "noise_b": [0.0]})

    result = top_contributions(model, X, feats, top_n=3)
    top_feature = result[0][0][0]
    assert top_feature == "signal"


def test_contributions_frame_flattens_correctly():
    contribs = [
        [("signal", 0.8), ("noise_a", -0.1)],
        [("signal", -0.5), ("noise_b", 0.2)],
    ]
    df = contributions_frame([100, 200], contribs)
    assert len(df) == 4
    assert set(df.columns) == {
        "game_pk",
        "rank",
        "feature",
        "contribution",
        "direction",
        "abs_contribution",
    }
    positive_row = df[(df["game_pk"] == 100) & (df["feature"] == "signal")].iloc[0]
    assert positive_row["direction"].startswith("▲")
    negative_row = df[(df["game_pk"] == 200) & (df["feature"] == "signal")].iloc[0]
    assert negative_row["direction"].startswith("▼")


def test_contributions_frame_empty_input():
    df = contributions_frame([], [])
    assert df.empty
    assert "game_pk" in df.columns
