"""Tests for the silver `team_game_stats` transform."""

from __future__ import annotations

import copy
from datetime import UTC, datetime

import pandas as pd

from statflow.transform.runner import build_silver

SAMPLE_BOXSCORE = {
    "teams": {
        "home": {
            "team": {"id": 147, "name": "New York Yankees"},
            "teamStats": {
                "batting": {
                    "runs": 5,
                    "hits": 10,
                    "atBats": 34,
                    "strikeOuts": 8,
                    "baseOnBalls": 3,
                    "homeRuns": 2,
                    "leftOnBase": 7,
                },
                "fielding": {"errors": 0},
            },
        },
        "away": {
            "team": {"id": 134, "name": "Pittsburgh Pirates"},
            "teamStats": {
                "batting": {
                    "runs": 3,
                    "hits": 7,
                    "atBats": 32,
                    "strikeOuts": 11,
                    "baseOnBalls": 2,
                    "homeRuns": 1,
                    "leftOnBase": 6,
                },
                "fielding": {"errors": 1},
            },
        },
    },
}


def test_team_game_stats_produces_two_rows_per_game(
    bronze_dir, silver_dir, write_boxscores, write_empty_bronze
):
    write_boxscores([{"game_pk": 111, "payload": SAMPLE_BOXSCORE}])
    for name in ("schedule", "plays", "transactions"):
        write_empty_bronze(name)

    build_silver(bronze_dir=bronze_dir, silver_dir=silver_dir)

    df = pd.read_parquet(silver_dir / "team_game_stats" / "team_game_stats.parquet")
    assert len(df) == 2

    home = df[df["is_home"]].iloc[0]
    away = df[~df["is_home"]].iloc[0]

    assert home["team_id"] == 147
    assert home["runs"] == 5
    assert home["hits"] == 10
    assert home["at_bats"] == 34
    assert home["strikeouts"] == 8
    assert home["walks"] == 3
    assert home["home_runs"] == 2
    assert home["left_on_base"] == 7
    assert home["errors"] == 0

    assert away["team_id"] == 134
    assert away["runs"] == 3
    assert away["hits"] == 7
    assert away["errors"] == 1


def test_team_game_stats_handles_multiple_games(
    bronze_dir, silver_dir, write_boxscores, write_empty_bronze
):
    game2 = copy.deepcopy(SAMPLE_BOXSCORE)
    game2["teams"]["home"]["team"]["id"] = 121  # Mets
    game2["teams"]["away"]["team"]["id"] = 143  # Phillies

    write_boxscores(
        [
            {"game_pk": 111, "payload": SAMPLE_BOXSCORE},
            {"game_pk": 222, "payload": game2},
        ]
    )
    for name in ("schedule", "plays", "transactions"):
        write_empty_bronze(name)

    build_silver(bronze_dir=bronze_dir, silver_dir=silver_dir)

    df = pd.read_parquet(silver_dir / "team_game_stats" / "team_game_stats.parquet")
    assert len(df) == 4
    assert set(df["game_pk"].unique()) == {111, 222}
    assert set(df[df["is_home"]]["team_id"]) == {147, 121}


def test_team_game_stats_dedups_across_partitions(
    bronze_dir, silver_dir, write_boxscores, write_empty_bronze
):
    """Same game_pk in two partitions: keep the row from the latest ingested_at."""
    early = copy.deepcopy(SAMPLE_BOXSCORE)
    early["teams"]["home"]["teamStats"]["batting"]["runs"] = 1  # stale value

    write_boxscores(
        [{"game_pk": 111, "payload": early}],
        target_date="2026-07-22",
        ingested_at=datetime(2026, 7, 22, 20, 0, tzinfo=UTC),
    )
    # Second partition, same game_pk, later ingested_at, updated runs = 5
    write_boxscores(
        [{"game_pk": 111, "payload": SAMPLE_BOXSCORE}],
        target_date="2026-07-23",
        ingested_at=datetime(2026, 7, 23, 4, 0, tzinfo=UTC),
    )
    for name in ("schedule", "plays", "transactions"):
        write_empty_bronze(name)

    build_silver(bronze_dir=bronze_dir, silver_dir=silver_dir)

    df = pd.read_parquet(silver_dir / "team_game_stats" / "team_game_stats.parquet")
    assert len(df) == 2  # one home + one away, not four
    home = df[df["is_home"]].iloc[0]
    assert home["runs"] == 5  # latest wins, not the stale 1
