"""Tests for the gold `pitcher_form` intermediate.

Same anti-leakage discipline as team_rolling. Additionally verifies the
cumulative-ERA formula and days-rest computation.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from statflow.features.runner import build_features


def _starter_series(
    starts: list[tuple[float, int, int]],
    pitcher_id: int = 999,
    team_id: int = 100,
    days_between_starts: int = 5,
    start_date: date = date(2024, 4, 1),
):
    """Build synthetic games + pitcher_game_stats for a single starter's
    successive starts. `starts` is a list of (innings_pitched, earned_runs,
    strikeouts) tuples.
    """
    games = []
    pitcher_stats = []
    for i, (ip, er, k) in enumerate(starts):
        game_pk = 5000 + i
        game_date = start_date + timedelta(days=i * days_between_starts)
        games.append(
            {
                "game_pk": game_pk,
                "game_date": game_date,
                "status": "Final",
                "home_team_id": team_id,
                "away_team_id": 200,
                "home_score": 3,
                "away_score": er,
                "total_runs": 3 + er,
                "home_win": er < 3,
            }
        )
        pitcher_stats.append(
            {
                "game_pk": game_pk,
                "team_id": team_id,
                "pitcher_id": pitcher_id,
                "is_starter": True,
                "innings_pitched": ip,
                "earned_runs": er,
                "strikeouts": k,
            }
        )
    return games, pitcher_stats


def test_first_start_has_no_prior_history(
    silver_dir, gold_dir, write_games, write_pitcher_game_stats
):
    games, pitchers = _starter_series([(6.0, 2, 8)])
    write_games(games)
    write_pitcher_game_stats(pitchers)

    build_features(silver_dir=silver_dir, gold_dir=gold_dir)

    df = pd.read_parquet(gold_dir / "pitcher_form" / "pitcher_form.parquet")
    row = df.iloc[0]
    assert row["n_prior_starts"] == 0
    assert pd.isna(row["era_l5"])
    assert pd.isna(row["k_per_9_l5"])
    assert pd.isna(row["days_rest"])


def test_days_rest_between_starts(silver_dir, gold_dir, write_games, write_pitcher_game_stats):
    games, pitchers = _starter_series(
        [(6.0, 2, 8), (5.0, 3, 6), (7.0, 1, 9)],
        days_between_starts=5,
    )
    write_games(games)
    write_pitcher_game_stats(pitchers)

    build_features(silver_dir=silver_dir, gold_dir=gold_dir)

    df = (
        pd.read_parquet(gold_dir / "pitcher_form" / "pitcher_form.parquet")
        .sort_values("game_pk")
        .reset_index(drop=True)
    )
    assert pd.isna(df.iloc[0]["days_rest"])
    assert df.iloc[1]["days_rest"] == 5
    assert df.iloc[2]["days_rest"] == 5


def test_era_uses_cumulative_formula(silver_dir, gold_dir, write_games, write_pitcher_game_stats):
    """3 prior starts: (2 ER in 6 IP), (3 ER in 5 IP), (0 ER in 7 IP).
    Sum ER = 5, sum IP = 18. Cumulative ERA = 5 * 9 / 18 = 2.5.
    """
    games, pitchers = _starter_series(
        [
            (6.0, 2, 5),  # start 1
            (5.0, 3, 4),  # start 2
            (7.0, 0, 8),  # start 3
            (6.0, 9, 3),  # start 4 — the row we assert on; window is starts 1-3
        ]
    )
    write_games(games)
    write_pitcher_game_stats(pitchers)

    build_features(silver_dir=silver_dir, gold_dir=gold_dir)

    df = (
        pd.read_parquet(gold_dir / "pitcher_form" / "pitcher_form.parquet")
        .sort_values("game_pk")
        .reset_index(drop=True)
    )
    row = df.iloc[3]
    assert row["era_l5"] == pytest.approx(2.5, abs=0.01)
    assert row["n_prior_starts"] == 3
    # K/9: (5 + 4 + 8) * 9 / 18 = 17 * 0.5 = 8.5
    assert row["k_per_9_l5"] == pytest.approx(8.5, abs=0.01)


def test_window_excludes_current_start_no_leakage(
    silver_dir, gold_dir, write_games, write_pitcher_game_stats
):
    """5 clean starts (0 ER), then a 6th start with 999 ER. The 6th start's
    era_l5 must equal 0.0 — if leakage sneaks in, we'd see a huge ERA.
    """
    starts = [(6.0, 0, 5)] * 5 + [(6.0, 999, 5)]
    games, pitchers = _starter_series(starts)
    write_games(games)
    write_pitcher_game_stats(pitchers)

    build_features(silver_dir=silver_dir, gold_dir=gold_dir)

    df = (
        pd.read_parquet(gold_dir / "pitcher_form" / "pitcher_form.parquet")
        .sort_values("game_pk")
        .reset_index(drop=True)
    )
    assert df.iloc[5]["era_l5"] == pytest.approx(0.0)


def test_relievers_are_excluded(silver_dir, gold_dir, write_games, write_pitcher_game_stats):
    """pitcher_form should only have rows for is_starter=True appearances."""
    # A starter + a reliever in the same game.
    games = [
        {
            "game_pk": 111,
            "game_date": date(2024, 4, 1),
            "status": "Final",
            "home_team_id": 100,
            "away_team_id": 200,
            "home_score": 5,
            "away_score": 3,
            "total_runs": 8,
            "home_win": True,
        }
    ]
    write_games(games)
    write_pitcher_game_stats(
        [
            {
                "game_pk": 111,
                "team_id": 100,
                "pitcher_id": 1001,
                "is_starter": True,
                "innings_pitched": 6.0,
                "earned_runs": 2,
                "strikeouts": 8,
            },
            {
                "game_pk": 111,
                "team_id": 100,
                "pitcher_id": 1002,
                "is_starter": False,  # reliever
                "innings_pitched": 3.0,
                "earned_runs": 1,
                "strikeouts": 4,
            },
        ]
    )

    build_features(silver_dir=silver_dir, gold_dir=gold_dir)

    df = pd.read_parquet(gold_dir / "pitcher_form" / "pitcher_form.parquet")
    assert set(df["pitcher_id"]) == {1001}


def test_multiple_pitchers_isolated(silver_dir, gold_dir, write_games, write_pitcher_game_stats):
    """Two different starters' rolling stats don't cross-contaminate."""
    # Pitcher A: 3 straight great starts (0 ER)
    a_games, a_pitchers = _starter_series([(7.0, 0, 10)] * 3, pitcher_id=1, team_id=100)
    # Pitcher B: 3 straight bad starts (10 ER)
    b_games, b_pitchers = _starter_series(
        [(3.0, 10, 1)] * 3,
        pitcher_id=2,
        team_id=200,
        start_date=date(2024, 4, 2),  # shifted so game_pks don't collide
    )
    # Give unique game_pks
    for g in b_games:
        g["game_pk"] += 100
    for p in b_pitchers:
        p["game_pk"] += 100

    write_games(a_games + b_games)
    write_pitcher_game_stats(a_pitchers + b_pitchers)

    build_features(silver_dir=silver_dir, gold_dir=gold_dir)

    df = pd.read_parquet(gold_dir / "pitcher_form" / "pitcher_form.parquet")
    a = df[df["pitcher_id"] == 1].sort_values("game_pk").reset_index(drop=True)
    b = df[df["pitcher_id"] == 2].sort_values("game_pk").reset_index(drop=True)
    # Pitcher A's 3rd start: 2 prior clean starts → era_l5 = 0
    assert a.iloc[2]["era_l5"] == pytest.approx(0.0)
    # Pitcher B's 3rd start: 2 prior bad starts (10 ER in 3 IP each) → era_l5 = 20*9/6 = 30
    assert b.iloc[2]["era_l5"] == pytest.approx(30.0)
