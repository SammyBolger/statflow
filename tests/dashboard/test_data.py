"""Tests for the dashboard's pure data-loading helpers."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from statflow.dashboard.data import (
    baseline_comparison,
    calibration_bins,
    load_prediction_outcomes,
    load_todays_games,
    rolling_performance,
    runs_scatter_data,
)


def _write_games(silver_dir: Path, rows: list[dict]) -> Path:
    out = silver_dir / "games" / "games.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    return out


def _write_predictions(gold_dir: Path, target: str, rows: list[dict]) -> Path:
    out = gold_dir / "predictions" / f"date={target}" / "predictions.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    return out


def _write_outcomes(gold_dir: Path, rows: list[dict]) -> Path:
    out = gold_dir / "prediction_outcomes" / "prediction_outcomes.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    return out


def _game(game_pk: int, game_date: str, home_team_name: str = "Yankees") -> dict:
    return {
        "game_pk": game_pk,
        "game_date": pd.Timestamp(game_date),
        "game_datetime_utc": pd.Timestamp(f"{game_date}T23:00:00Z"),
        "status": "Scheduled",
        "home_team_name": home_team_name,
        "away_team_name": "Pirates",
        "home_score": 0,
        "away_score": 0,
        "home_win": False,
        "home_probable_pitcher_name": "Cole",
        "away_probable_pitcher_name": "Keller",
    }


# ---------------------------------------------------------------------------
# load_todays_games
# ---------------------------------------------------------------------------


def test_todays_games_empty_if_silver_missing(tmp_path):
    silver = tmp_path / "silver"
    gold = tmp_path / "gold"
    result = load_todays_games(date(2026, 7, 27), silver_dir=silver, gold_dir=gold)
    assert result.empty


def test_todays_games_returns_games_for_date(tmp_path):
    silver = tmp_path / "silver"
    gold = tmp_path / "gold"
    _write_games(
        silver,
        [
            _game(1, "2026-07-27"),
            _game(2, "2026-07-27"),
            _game(3, "2026-07-28"),  # different date
        ],
    )

    result = load_todays_games(date(2026, 7, 27), silver_dir=silver, gold_dir=gold)
    assert len(result) == 2
    assert set(result["game_pk"]) == {1, 2}


def test_todays_games_left_joins_predictions_when_present(tmp_path):
    silver = tmp_path / "silver"
    gold = tmp_path / "gold"
    _write_games(silver, [_game(1, "2026-07-27"), _game(2, "2026-07-27")])
    _write_predictions(
        gold,
        "2026-07-27",
        [
            {"game_pk": 1, "predicted_home_win_prob": 0.6, "predicted_total_runs": 8.5},
        ],
    )

    result = load_todays_games(date(2026, 7, 27), silver_dir=silver, gold_dir=gold)
    game_1 = result[result["game_pk"] == 1].iloc[0]
    game_2 = result[result["game_pk"] == 2].iloc[0]
    assert game_1["predicted_home_win_prob"] == pytest.approx(0.6)
    assert pd.isna(game_2["predicted_home_win_prob"])


def test_todays_games_no_predictions_file_leaves_nulls(tmp_path):
    silver = tmp_path / "silver"
    gold = tmp_path / "gold"
    _write_games(silver, [_game(1, "2026-07-27")])

    result = load_todays_games(date(2026, 7, 27), silver_dir=silver, gold_dir=gold)
    assert pd.isna(result.iloc[0]["predicted_home_win_prob"])


# ---------------------------------------------------------------------------
# load_prediction_outcomes / rolling_performance / calibration_bins
# ---------------------------------------------------------------------------


def test_load_outcomes_empty_if_missing(tmp_path):
    assert load_prediction_outcomes(gold_dir=tmp_path).empty


def test_rolling_performance_returns_empty_dict_on_empty_input():
    assert rolling_performance(pd.DataFrame()) == {}


def test_rolling_performance_filters_by_window():
    """Only outcomes within the last `window_days` count."""
    rows = []
    base = pd.Timestamp("2026-07-27")
    for i in range(60):
        rows.append(
            {
                "game_date": base - pd.Timedelta(days=i),
                "winner_correct": i % 2 == 0,
                "winner_log_loss": 0.5,
                "winner_brier": 0.1,
                "runs_abs_error": 2.0,
                "runs_squared_error": 4.0,
            }
        )
    outcomes = pd.DataFrame(rows)

    result = rolling_performance(outcomes, window_days=30)
    assert result["n_games"] == 30  # only 30 days back
    assert result["accuracy"] == pytest.approx(0.5)  # every other row


def test_calibration_bins_perfect_calibration():
    """Mean predicted ≈ mean actual per bin when target = probability."""
    rng = np.random.default_rng(42)
    probs = rng.uniform(0, 1, size=1000)
    actuals = (rng.uniform(0, 1, size=1000) < probs).astype(int)
    outcomes = pd.DataFrame({"predicted_home_win_prob": probs, "actual_home_win": actuals})

    bins = calibration_bins(outcomes, n_bins=10)
    assert len(bins) <= 10
    diffs = (bins["mean_predicted"] - bins["mean_actual"]).abs()
    assert diffs.max() < 0.1


def test_baseline_comparison_empty_input():
    assert baseline_comparison(pd.DataFrame()).empty


def test_baseline_comparison_shape_and_metric_names():
    outcomes = pd.DataFrame(
        {
            "actual_home_win": [1, 1, 0, 0, 1],
            "actual_total_runs": [8, 10, 6, 4, 12],
            "winner_correct": [True, True, False, False, True],
            "winner_log_loss": [0.4, 0.3, 0.9, 0.7, 0.5],
            "winner_brier": [0.1, 0.09, 0.2, 0.15, 0.12],
            "runs_abs_error": [1.0, 2.0, 3.0, 1.5, 2.5],
            "runs_squared_error": [1.0, 4.0, 9.0, 2.25, 6.25],
        }
    )
    compare = baseline_comparison(outcomes)
    assert list(compare["metric"]) == ["Accuracy", "Log loss", "Brier", "MAE (runs)", "RMSE (runs)"]
    assert compare["model"].notna().all()
    assert compare["baseline"].notna().all()


def test_runs_scatter_data_empty_input():
    df = runs_scatter_data(pd.DataFrame())
    assert list(df.columns) == ["game_pk", "predicted_total_runs", "actual_total_runs"]
    assert df.empty


def test_runs_scatter_data_returns_expected_columns():
    outcomes = pd.DataFrame(
        {
            "game_pk": [1, 2, 3],
            "predicted_total_runs": [8.0, 9.5, 7.2],
            "actual_total_runs": [7, 12, 6],
            "some_other_col": [None, None, None],
        }
    )
    df = runs_scatter_data(outcomes)
    assert list(df.columns) == ["game_pk", "predicted_total_runs", "actual_total_runs"]
    assert len(df) == 3
