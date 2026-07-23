"""Tests for the final features.sql assembly.

Verifies the ASOF joins for probable pitcher form + park factors, the
direct joins for team_rolling, and the trivial context/target columns.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from statflow.features.runner import build_features


def _game(
    game_pk: int,
    game_date: date,
    home_team: int,
    away_team: int,
    home_score: int,
    away_score: int,
    home_pp: int | None = None,
    away_pp: int | None = None,
    venue_id: int = 100,
    day_night: str = "night",
    status: str = "Final",
) -> dict:
    return {
        "game_pk": game_pk,
        "game_date": game_date,
        "season": game_date.year,
        "game_type": "R",
        "status": status,
        "home_team_id": home_team,
        "away_team_id": away_team,
        "home_score": home_score,
        "away_score": away_score,
        "total_runs": home_score + away_score,
        "home_win": home_score > away_score,
        "venue_id": venue_id,
        "day_night": day_night,
        "home_probable_pitcher_id": home_pp if home_pp is not None else pd.NA,
        "away_probable_pitcher_id": away_pp if away_pp is not None else pd.NA,
    }


def _team_stats(game_pk: int, home_team: int, away_team: int, home_runs: int, away_runs: int):
    return [
        {"game_pk": game_pk, "team_id": home_team, "is_home": True, "runs": home_runs},
        {"game_pk": game_pk, "team_id": away_team, "is_home": False, "runs": away_runs},
    ]


def test_features_has_row_per_game_with_targets(
    silver_dir, gold_dir, write_games, write_team_game_stats
):
    games = [_game(111, date(2024, 4, 1), 100, 200, 5, 3)]
    write_games(games)
    write_team_game_stats(_team_stats(111, 100, 200, 5, 3))

    build_features(silver_dir=silver_dir, gold_dir=gold_dir)

    df = pd.read_parquet(gold_dir / "features" / "features.parquet")
    assert len(df) == 1
    row = df.iloc[0]
    assert row["game_pk"] == 111
    assert row["target_home_win"] is True or row["target_home_win"] == 1
    assert row["target_total_runs"] == 8
    assert row["season_type"] == "R"
    assert row["month"] == 4


def test_features_home_and_away_rolling_not_swapped(
    silver_dir, gold_dir, write_games, write_team_game_stats
):
    """Team 100 (home) has scored 5/game for 11 games. Team 200 (away)
    has scored 1/game. After 11 games:
      - home_runs_scored_l10 should be 5, not 1.
      - away_runs_scored_l10 should be 1, not 5.
    """
    games = []
    stats = []
    for i in range(11):
        games.append(_game(1000 + i, date(2024, 4, 1) + timedelta(days=i), 100, 200, 5, 1))
        stats.extend(_team_stats(1000 + i, 100, 200, 5, 1))
    write_games(games)
    write_team_game_stats(stats)

    build_features(silver_dir=silver_dir, gold_dir=gold_dir)

    df = pd.read_parquet(gold_dir / "features" / "features.parquet").sort_values("game_pk")
    row = df.iloc[10]
    assert row["home_runs_scored_l10"] == pytest.approx(5.0)
    assert row["away_runs_scored_l10"] == pytest.approx(1.0)
    assert row["home_runs_allowed_l10"] == pytest.approx(1.0)
    assert row["away_runs_allowed_l10"] == pytest.approx(5.0)


def test_features_probable_pitcher_asof_join(
    silver_dir,
    gold_dir,
    write_games,
    write_team_game_stats,
    write_pitcher_game_stats,
):
    """Pitcher 999 has 3 great prior starts. Then game G on a later date lists
    pitcher 999 as the home probable. features.sql should ASOF-join and give
    G's home_sp_era_l5 = 0.0 (from those 3 shutouts), not NULL.
    """
    # 3 prior starts by pitcher 999 (all 6 IP, 0 ER, 5 K)
    prior_games = []
    prior_pitchers = []
    prior_stats = []
    for i in range(3):
        pk = 500 + i
        d = date(2024, 4, 1) + timedelta(days=i * 5)
        prior_games.append(_game(pk, d, 100, 200, 3, 1, home_pp=999))
        prior_stats.extend(_team_stats(pk, 100, 200, 3, 1))
        prior_pitchers.append(
            {
                "game_pk": pk,
                "team_id": 100,
                "pitcher_id": 999,
                "is_starter": True,
                "innings_pitched": 6.0,
                "earned_runs": 0,
                "strikeouts": 5,
            }
        )

    # Target game: scheduled game with pitcher 999 as home probable.
    target = _game(
        9999,
        date(2024, 5, 1),
        100,
        200,
        0,
        0,
        home_pp=999,
        status="Scheduled",
    )
    target["home_score"] = 0
    target["away_score"] = 0
    target["total_runs"] = 0
    target["home_win"] = False

    write_games(prior_games + [target])
    write_team_game_stats(prior_stats + _team_stats(9999, 100, 200, 0, 0))
    write_pitcher_game_stats(prior_pitchers)

    build_features(silver_dir=silver_dir, gold_dir=gold_dir)

    df = pd.read_parquet(gold_dir / "features" / "features.parquet")
    scheduled_row = df[df["game_pk"] == 9999].iloc[0]
    assert scheduled_row["home_sp_era_l5"] == pytest.approx(0.0)
    # Away probable pitcher not set → NULL
    assert pd.isna(scheduled_row["away_sp_era_l5"])


def test_features_park_factor_asof_join(silver_dir, gold_dir, write_games, write_team_game_stats):
    """Venue 1 gets high-scoring games; scheduled game at venue 1 should
    inherit the park factor from prior Final games at that venue.
    """
    # 20 prior Final games at venue 1, all high-scoring.
    games = []
    stats = []
    for i in range(20):
        pk = 100 + i
        games.append(
            _game(
                pk,
                date(2024, 4, 1) + timedelta(days=i),
                home_team=100,
                away_team=200,
                home_score=10,
                away_score=6,
                venue_id=1,
            )
        )
        stats.extend(_team_stats(pk, 100, 200, 10, 6))
    # A scheduled future game at venue 1
    scheduled = _game(
        9999,
        date(2024, 5, 1),
        100,
        200,
        0,
        0,
        venue_id=1,
        status="Scheduled",
    )
    scheduled["home_score"] = 0
    scheduled["away_score"] = 0
    scheduled["total_runs"] = 0
    scheduled["home_win"] = False

    write_games(games + [scheduled])
    write_team_game_stats(stats + _team_stats(9999, 100, 200, 0, 0))

    build_features(silver_dir=silver_dir, gold_dir=gold_dir)

    df = pd.read_parquet(gold_dir / "features" / "features.parquet")
    scheduled_row = df[df["game_pk"] == 9999].iloc[0]
    # A park factor should be present (asof-inherited from the venue's history)
    assert not pd.isna(scheduled_row["venue_park_factor_runs"])
    # And it should be roughly 1.0 since only venue 1 had games (venue == league)
    assert scheduled_row["venue_park_factor_runs"] == pytest.approx(1.0)


def test_features_is_day_game_boolean(silver_dir, gold_dir, write_games, write_team_game_stats):
    games = [
        _game(1, date(2024, 4, 1), 100, 200, 5, 3, day_night="day"),
        _game(2, date(2024, 4, 2), 100, 200, 5, 3, day_night="night"),
    ]
    write_games(games)
    write_team_game_stats(_team_stats(1, 100, 200, 5, 3) + _team_stats(2, 100, 200, 5, 3))

    build_features(silver_dir=silver_dir, gold_dir=gold_dir)

    df = pd.read_parquet(gold_dir / "features" / "features.parquet").sort_values("game_pk")
    assert df.iloc[0]["is_day_game"] is True or df.iloc[0]["is_day_game"] == 1
    assert df.iloc[1]["is_day_game"] is False or df.iloc[1]["is_day_game"] == 0
