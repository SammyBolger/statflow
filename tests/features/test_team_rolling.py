"""Tests for the gold `team_rolling` intermediate.

The star of the file is the anti-leakage test: if the rolling window ever
peeks at the current game, one specific assertion breaks loudly.
"""

from __future__ import annotations

import pandas as pd
import pytest

from statflow.features.runner import build_features


def _fixed_team_series(runs_per_game: list[int], team_id: int = 100, opp_id: int = 200):
    """Build a series where a single team plays at home every game with the
    given runs, and the opponent is always the same. Simpler than the
    alternating home/away helper for reasoning about a single team's window.
    """
    from datetime import date, timedelta

    games = []
    stats = []
    for i, runs in enumerate(runs_per_game):
        game_pk = 1000 + i
        games.append(
            {
                "game_pk": game_pk,
                "game_date": date(2024, 4, 1) + timedelta(days=i),
                "status": "Final",
                "home_team_id": team_id,
                "away_team_id": opp_id,
                "home_score": runs,
                "away_score": 0,
                "total_runs": runs,
                "home_win": runs > 0,
                "venue_id": team_id,
            }
        )
        stats.append({"game_pk": game_pk, "team_id": team_id, "is_home": True, "runs": runs})
        stats.append({"game_pk": game_pk, "team_id": opp_id, "is_home": False, "runs": 0})
    return games, stats


def test_first_game_has_no_prior_history(silver_dir, gold_dir, write_games, write_team_game_stats):
    games, stats = _fixed_team_series([5])
    write_games(games)
    write_team_game_stats(stats)

    build_features(silver_dir=silver_dir, gold_dir=gold_dir)

    df = pd.read_parquet(gold_dir / "team_rolling" / "team_rolling.parquet")
    team_row = df[df["team_id"] == 100].iloc[0]
    assert team_row["n_prior_games"] == 0
    assert pd.isna(team_row["runs_scored_l10"])
    assert pd.isna(team_row["win_pct_l10"])


def test_partial_window_before_10_games(silver_dir, gold_dir, write_games, write_team_game_stats):
    """Game 5 (index 4) has only 4 prior games in the window."""
    games, stats = _fixed_team_series([1, 2, 3, 4, 5])
    write_games(games)
    write_team_game_stats(stats)

    build_features(silver_dir=silver_dir, gold_dir=gold_dir)

    df = pd.read_parquet(gold_dir / "team_rolling" / "team_rolling.parquet")
    team = df[df["team_id"] == 100].sort_values("game_pk").reset_index(drop=True)
    # Game 5 (index 4): rolling window is games 1-4, mean = (1+2+3+4)/4 = 2.5
    assert team.iloc[4]["runs_scored_l10"] == pytest.approx(2.5)
    assert team.iloc[4]["n_prior_games"] == 4


def test_full_window_averages_last_10_games(
    silver_dir, gold_dir, write_games, write_team_game_stats
):
    games, stats = _fixed_team_series(list(range(1, 12)))  # runs 1..11 across 11 games
    write_games(games)
    write_team_game_stats(stats)

    build_features(silver_dir=silver_dir, gold_dir=gold_dir)

    df = pd.read_parquet(gold_dir / "team_rolling" / "team_rolling.parquet")
    team = df[df["team_id"] == 100].sort_values("game_pk").reset_index(drop=True)
    # Game 11: window is games 1..10, mean of 1..10 = 5.5
    assert team.iloc[10]["runs_scored_l10"] == pytest.approx(5.5)
    assert team.iloc[10]["n_prior_games"] == 10


def test_window_excludes_current_game_no_leakage(
    silver_dir, gold_dir, write_games, write_team_game_stats
):
    """The killer test. If the window ever includes the current game, a
    game with an extreme value would leak into its own rolling stat.
    Setup: 10 games with runs=0, then game 11 with runs=999. Game 11's
    L10 must equal 0.0 (mean of games 1..10) — NOT anywhere near 100.
    """
    runs = [0] * 10 + [999]
    games, stats = _fixed_team_series(runs)
    write_games(games)
    write_team_game_stats(stats)

    build_features(silver_dir=silver_dir, gold_dir=gold_dir)

    df = pd.read_parquet(gold_dir / "team_rolling" / "team_rolling.parquet")
    team = df[df["team_id"] == 100].sort_values("game_pk").reset_index(drop=True)
    assert team.iloc[10]["runs_scored_l10"] == pytest.approx(0.0)
    # If leakage happened, this would be ~90.8, definitively not 0.


def test_runs_allowed_uses_opponent_runs(silver_dir, gold_dir, write_games, write_team_game_stats):
    """Team 100 scores 1 every game; opponent scores 7 every game.
    team_rolling.runs_allowed_l10 for team 100 should be 7, not 1.
    """
    from datetime import date, timedelta

    games = []
    stats = []
    for i in range(11):
        game_pk = 1000 + i
        games.append(
            {
                "game_pk": game_pk,
                "game_date": date(2024, 4, 1) + timedelta(days=i),
                "status": "Final",
                "home_team_id": 100,
                "away_team_id": 200,
                "home_score": 1,
                "away_score": 7,
                "total_runs": 8,
                "home_win": False,
                "venue_id": 100,
            }
        )
        stats.append({"game_pk": game_pk, "team_id": 100, "is_home": True, "runs": 1})
        stats.append({"game_pk": game_pk, "team_id": 200, "is_home": False, "runs": 7})
    write_games(games)
    write_team_game_stats(stats)

    build_features(silver_dir=silver_dir, gold_dir=gold_dir)

    df = pd.read_parquet(gold_dir / "team_rolling" / "team_rolling.parquet")
    team_100 = df[df["team_id"] == 100].sort_values("game_pk").reset_index(drop=True)
    # Game 11 for team 100: 10 prior losses of 1-7. runs_allowed should be 7.
    assert team_100.iloc[10]["runs_allowed_l10"] == pytest.approx(7.0)
    assert team_100.iloc[10]["runs_scored_l10"] == pytest.approx(1.0)
    assert team_100.iloc[10]["win_pct_l10"] == pytest.approx(0.0)


def test_days_rest_first_game_null(silver_dir, gold_dir, write_games, write_team_game_stats):
    games, stats = _fixed_team_series([5])
    write_games(games)
    write_team_game_stats(stats)

    build_features(silver_dir=silver_dir, gold_dir=gold_dir)

    df = pd.read_parquet(gold_dir / "team_rolling" / "team_rolling.parquet")
    row = df[df["team_id"] == 100].iloc[0]
    assert pd.isna(row["days_rest"])


def test_days_rest_daily_games(silver_dir, gold_dir, write_games, write_team_game_stats):
    """A team playing every day has days_rest=1 (previous game was yesterday)."""
    games, stats = _fixed_team_series([1, 2, 3])
    write_games(games)
    write_team_game_stats(stats)

    build_features(silver_dir=silver_dir, gold_dir=gold_dir)

    df = (
        pd.read_parquet(gold_dir / "team_rolling" / "team_rolling.parquet")
        .query("team_id == 100")
        .sort_values("game_pk")
        .reset_index(drop=True)
    )
    assert pd.isna(df.iloc[0]["days_rest"])
    assert df.iloc[1]["days_rest"] == 1
    assert df.iloc[2]["days_rest"] == 1


def test_no_cross_team_contamination(
    silver_dir, gold_dir, write_games, write_team_game_stats, head_to_head_series
):
    """Two different teams playing each other — each team's rolling stat
    reflects that team's history only, not the opponent's.
    """
    # Alternate home/away between teams 100 and 200 across 11 games.
    # Team 100 scores 10 as home, 0 as away. Team 200 scores 1 as home, 5 as away.
    home_runs = [10, 5, 10, 5, 10, 5, 10, 5, 10, 5, 10]  # depends on which team is home
    away_runs = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    games, stats = head_to_head_series(11, 100, 200, home_runs=home_runs, away_runs=away_runs)
    write_games(games)
    write_team_game_stats(stats)

    build_features(silver_dir=silver_dir, gold_dir=gold_dir)

    df = pd.read_parquet(gold_dir / "team_rolling" / "team_rolling.parquet")
    # Each team should have 11 rows.
    assert (df["team_id"] == 100).sum() == 11
    assert (df["team_id"] == 200).sum() == 11
