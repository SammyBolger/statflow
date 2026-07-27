"""Tests for the prediction pipeline.

We don't spin up a real MLflow tracking store — we monkey-patch the model
loaders to return trivial predictors so the test focuses on the wiring:
which features go in, what schema comes out, and how empty-date handling
works.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from statflow.models.data import FEATURE_COLS
from statflow.models.predict import predict_for_date


class _FakeWinner:
    def predict_proba(self, X):
        # Constant 0.6 home-win prob
        return np.column_stack([1 - np.full(len(X), 0.6), np.full(len(X), 0.6)])


class _FakeRuns:
    def predict(self, X):
        return np.full(len(X), 8.5)


def _write_features(gold_dir: Path, rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    out = gold_dir / "features" / "features.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)


def _feature_row(game_pk: int, game_date: str) -> dict:
    row = {
        "game_pk": game_pk,
        "game_date": pd.Timestamp(game_date),
        "season": int(game_date[:4]),
        "home_team_id": 100,
        "away_team_id": 200,
    }
    for col in FEATURE_COLS:
        row[col] = 5.0
    return row


def test_predict_writes_predictions_parquet(tmp_path, monkeypatch):
    gold_dir = tmp_path / "gold"
    _write_features(
        gold_dir,
        [
            _feature_row(111, "2026-07-27"),
            _feature_row(222, "2026-07-27"),
        ],
    )
    monkeypatch.setattr(
        "statflow.models.predict.load_latest_winner_model",
        lambda: (_FakeWinner(), "winner-run-abc"),
    )
    monkeypatch.setattr(
        "statflow.models.predict.load_latest_runs_model",
        lambda: (_FakeRuns(), "runs-run-def"),
    )

    path = predict_for_date(date(2026, 7, 27), gold_dir=gold_dir)

    assert path == gold_dir / "predictions" / "date=2026-07-27" / "predictions.parquet"
    df = pd.read_parquet(path)
    assert len(df) == 2
    assert list(df.columns) == [
        "game_pk",
        "game_date",
        "predicted_home_win_prob",
        "predicted_total_runs",
        "winner_model_run_id",
        "runs_model_run_id",
        "predicted_at",
    ]
    assert (df["predicted_home_win_prob"] == 0.6).all()
    assert (df["predicted_total_runs"] == 8.5).all()
    assert (df["winner_model_run_id"] == "winner-run-abc").all()


def test_predict_returns_none_for_date_with_no_games(tmp_path, monkeypatch):
    gold_dir = tmp_path / "gold"
    _write_features(gold_dir, [_feature_row(111, "2026-07-27")])
    monkeypatch.setattr(
        "statflow.models.predict.load_latest_winner_model",
        lambda: (_FakeWinner(), "w"),
    )
    monkeypatch.setattr(
        "statflow.models.predict.load_latest_runs_model",
        lambda: (_FakeRuns(), "r"),
    )

    result = predict_for_date(date(2026, 12, 25), gold_dir=gold_dir)
    assert result is None


def test_predict_only_scores_target_date(tmp_path, monkeypatch):
    gold_dir = tmp_path / "gold"
    _write_features(
        gold_dir,
        [
            _feature_row(111, "2026-07-27"),
            _feature_row(222, "2026-07-28"),  # different date
        ],
    )
    monkeypatch.setattr(
        "statflow.models.predict.load_latest_winner_model",
        lambda: (_FakeWinner(), "w"),
    )
    monkeypatch.setattr(
        "statflow.models.predict.load_latest_runs_model",
        lambda: (_FakeRuns(), "r"),
    )

    path = predict_for_date(date(2026, 7, 27), gold_dir=gold_dir)
    df = pd.read_parquet(path)
    assert len(df) == 1
    assert df.iloc[0]["game_pk"] == 111
