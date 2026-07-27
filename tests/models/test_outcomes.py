"""Tests for the prediction_outcomes monitoring join."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from statflow.models.outcomes import build_prediction_outcomes


def _write_predictions(gold_dir: Path, target_date: str, rows: list[dict]) -> Path:
    df = pd.DataFrame(rows)
    out = gold_dir / "predictions" / f"date={target_date}" / "predictions.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out


def _write_games(silver_dir: Path, rows: list[dict]) -> Path:
    df = pd.DataFrame(rows)
    out = silver_dir / "games" / "games.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out


def _pred(game_pk: int, prob: float, runs: float, model: str = "run-abc") -> dict:
    return {
        "game_pk": game_pk,
        "game_date": "2026-07-27",
        "predicted_home_win_prob": prob,
        "predicted_total_runs": runs,
        "winner_model_run_id": model,
        "runs_model_run_id": model,
        "predicted_at": datetime(2026, 7, 27, tzinfo=UTC),
    }


def _game(game_pk: int, status: str, home_win: bool = True, total_runs: int = 8) -> dict:
    return {
        "game_pk": game_pk,
        "status": status,
        "home_win": home_win,
        "total_runs": total_runs,
    }


def test_outcomes_joins_predictions_to_finals(tmp_path):
    silver = tmp_path / "silver"
    gold = tmp_path / "gold"
    _write_predictions(gold, "2026-07-27", [_pred(111, 0.7, 8.5)])
    _write_games(silver, [_game(111, "Final", home_win=True, total_runs=9)])

    path = build_prediction_outcomes(silver_dir=silver, gold_dir=gold)
    df = pd.read_parquet(path)

    assert len(df) == 1
    row = df.iloc[0]
    assert row["actual_home_win"] == 1
    assert row["actual_total_runs"] == 9
    # winner_correct: predicted 0.7 (favoring home) and home won → True
    assert row["winner_correct"]
    # brier = (0.7 - 1.0)^2 = 0.09
    assert row["winner_brier"] == pytest.approx(0.09, abs=1e-6)
    # runs_abs_error = |8.5 - 9| = 0.5
    assert row["runs_abs_error"] == pytest.approx(0.5)


def test_outcomes_returns_none_when_no_predictions(tmp_path):
    silver = tmp_path / "silver"
    gold = tmp_path / "gold"
    _write_games(silver, [_game(111, "Final")])
    assert build_prediction_outcomes(silver_dir=silver, gold_dir=gold) is None


def test_outcomes_returns_none_when_no_games_final_yet(tmp_path):
    """We predicted a game but it hasn't finished yet — no outcome row."""
    silver = tmp_path / "silver"
    gold = tmp_path / "gold"
    _write_predictions(gold, "2026-07-27", [_pred(111, 0.6, 8)])
    _write_games(silver, [_game(111, "Scheduled", home_win=False, total_runs=0)])

    assert build_prediction_outcomes(silver_dir=silver, gold_dir=gold) is None


def test_outcomes_dedups_by_game_and_model(tmp_path):
    """Same game predicted by the same model twice — keep the freshest."""
    silver = tmp_path / "silver"
    gold = tmp_path / "gold"
    early = _pred(111, 0.5, 7.0)
    early["predicted_at"] = datetime(2026, 7, 27, 10, tzinfo=UTC)
    late = _pred(111, 0.7, 8.5)
    late["predicted_at"] = datetime(2026, 7, 27, 15, tzinfo=UTC)
    _write_predictions(gold, "2026-07-27", [early])
    _write_predictions(gold, "2026-07-28", [late])
    _write_games(silver, [_game(111, "Final", home_win=True, total_runs=9)])

    path = build_prediction_outcomes(silver_dir=silver, gold_dir=gold)
    df = pd.read_parquet(path)

    assert len(df) == 1
    assert df.iloc[0]["predicted_home_win_prob"] == pytest.approx(0.7)


def test_outcomes_keeps_predictions_per_model_version(tmp_path):
    """Same game predicted by two model versions — both scored."""
    silver = tmp_path / "silver"
    gold = tmp_path / "gold"
    v1 = _pred(111, 0.4, 7.0, model="run-v1")
    v2 = _pred(111, 0.8, 9.0, model="run-v2")
    _write_predictions(gold, "2026-07-27", [v1, v2])
    _write_games(silver, [_game(111, "Final", home_win=True, total_runs=9)])

    path = build_prediction_outcomes(silver_dir=silver, gold_dir=gold)
    df = pd.read_parquet(path)

    assert len(df) == 2
    assert set(df["winner_model_run_id"]) == {"run-v1", "run-v2"}


def test_outcomes_row_log_loss_matches_hand_computation(tmp_path):
    silver = tmp_path / "silver"
    gold = tmp_path / "gold"
    # Predicted 0.9, actual home_win=True (=1). log_loss = -log(0.9) ≈ 0.1054
    _write_predictions(gold, "2026-07-27", [_pred(111, 0.9, 8)])
    _write_games(silver, [_game(111, "Final", home_win=True, total_runs=8)])

    path = build_prediction_outcomes(silver_dir=silver, gold_dir=gold)
    df = pd.read_parquet(path)

    assert df.iloc[0]["winner_log_loss"] == pytest.approx(-np.log(0.9), abs=1e-4)
