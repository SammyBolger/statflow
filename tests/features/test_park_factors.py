"""Tests for the gold `park_factors` intermediate."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from statflow.features.runner import build_features


def _games_at_venues(runs_by_game: list[tuple[int, int]]) -> list[dict]:
    """runs_by_game is [(venue_id, total_runs), ...] one entry per game.
    Games are dated consecutively starting 2024-04-01.
    """
    return [
        {
            "game_pk": 1000 + i,
            "game_date": date(2024, 4, 1) + timedelta(days=i),
            "status": "Final",
            "home_team_id": venue_id,
            "away_team_id": 999,
            "home_score": total_runs,
            "away_score": 0,
            "total_runs": total_runs,
            "home_win": True,
            "venue_id": venue_id,
        }
        for i, (venue_id, total_runs) in enumerate(runs_by_game)
    ]


def test_cold_start_no_prior_venue_games(silver_dir, gold_dir, write_games):
    write_games(_games_at_venues([(1, 8)]))

    build_features(silver_dir=silver_dir, gold_dir=gold_dir)

    df = pd.read_parquet(gold_dir / "park_factors" / "park_factors.parquet")
    assert df.iloc[0]["n_prior_venue_games"] == 0
    assert pd.isna(df.iloc[0]["venue_park_factor_runs"])


def test_neutral_park_factor_is_one(silver_dir, gold_dir, write_games):
    """If every game at every venue scores 8 runs, the park factor is 1.0."""
    # 20 games alternating between two venues, all scoring 8 runs.
    write_games(_games_at_venues([(1, 8) if i % 2 == 0 else (2, 8) for i in range(20)]))

    build_features(silver_dir=silver_dir, gold_dir=gold_dir)

    df = pd.read_parquet(gold_dir / "park_factors" / "park_factors.parquet")
    # Later games should have factor ~ 1.0
    later = df.iloc[-1]
    assert later["venue_park_factor_runs"] == pytest.approx(1.0)


def test_hitters_park_factor_above_one(silver_dir, gold_dir, write_games):
    """Venue 1 scores 12 every game; venue 2 scores 4 every game.
    Park factor for venue 1 should be substantially > 1 once enough
    history exists to distinguish it from league average.
    """
    games = _games_at_venues([(1, 12) if i % 2 == 0 else (2, 4) for i in range(30)])
    write_games(games)

    build_features(silver_dir=silver_dir, gold_dir=gold_dir)

    df = pd.read_parquet(gold_dir / "park_factors" / "park_factors.parquet")
    # Find the last game at venue 1
    venue_1_last = df[df["venue_id"] == 1].sort_values("game_pk").iloc[-1]
    venue_2_last = df[df["venue_id"] == 2].sort_values("game_pk").iloc[-1]
    # Venue 1 avg = 12, league avg = 8. Factor = 1.5.
    assert venue_1_last["venue_park_factor_runs"] == pytest.approx(1.5, abs=0.05)
    # Venue 2 avg = 4, league avg = 8. Factor = 0.5.
    assert venue_2_last["venue_park_factor_runs"] == pytest.approx(0.5, abs=0.05)


def test_window_excludes_current_game(silver_dir, gold_dir, write_games):
    """Venue 1 has 10 games all scoring 0. The 11th at venue 1 scores 999.
    Its park factor should reflect the 0-history, not the 999.
    """
    games = _games_at_venues([(1, 0)] * 10 + [(1, 999)])
    write_games(games)

    build_features(silver_dir=silver_dir, gold_dir=gold_dir)

    df = pd.read_parquet(gold_dir / "park_factors" / "park_factors.parquet")
    last = df.sort_values("game_pk").iloc[-1]
    # venue_avg_runs from prior 10 games (all 0) = 0
    assert last["venue_avg_runs"] == pytest.approx(0.0)
    # league_avg over same 10-game window also = 0
    assert last["league_avg_runs"] == pytest.approx(0.0)
    # NULLIF makes 0/0 → NULL, not divide-by-zero crash
    assert pd.isna(last["venue_park_factor_runs"])
