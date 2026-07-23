"""Tests for the silver pitcher_game_stats transform (Python-based)."""

from __future__ import annotations

import copy
from datetime import UTC, datetime

import pandas as pd
import pytest

from statflow.transform.runner import _parse_innings_pitched, build_silver

# One home pitcher (starter), one home reliever, one away starter, and a
# non-pitcher batter (to prove we filter him out).
SAMPLE_BOXSCORE = {
    "teams": {
        "home": {
            "team": {"id": 147, "name": "Yankees"},
            "teamStats": {"batting": {"runs": 5}, "fielding": {"errors": 0}},
            "pitchers": [1001, 1002],
            "players": {
                "ID1001": {
                    "person": {"id": 1001, "fullName": "Gerrit Cole"},
                    "stats": {
                        "pitching": {
                            "inningsPitched": "6.1",
                            "hits": 4,
                            "runs": 2,
                            "earnedRuns": 2,
                            "baseOnBalls": 1,
                            "strikeOuts": 8,
                            "numberOfPitches": 92,
                        }
                    },
                },
                "ID1002": {
                    "person": {"id": 1002, "fullName": "Clay Holmes"},
                    "stats": {
                        "pitching": {
                            "inningsPitched": "1.2",
                            "hits": 1,
                            "runs": 0,
                            "earnedRuns": 0,
                            "baseOnBalls": 0,
                            "strikeOuts": 3,
                            "numberOfPitches": 22,
                        }
                    },
                },
                "ID2001": {
                    # A hitter — no pitching stats, must be filtered out.
                    "person": {"id": 2001, "fullName": "Aaron Judge"},
                    "stats": {"batting": {"hits": 2}},
                },
            },
        },
        "away": {
            "team": {"id": 134, "name": "Pirates"},
            "teamStats": {"batting": {"runs": 3}, "fielding": {"errors": 1}},
            "pitchers": [3001],
            "players": {
                "ID3001": {
                    "person": {"id": 3001, "fullName": "Mitch Keller"},
                    "stats": {
                        "pitching": {
                            "inningsPitched": "5.0",
                            "hits": 8,
                            "runs": 5,
                            "earnedRuns": 5,
                            "baseOnBalls": 2,
                            "strikeOuts": 4,
                            "numberOfPitches": 95,
                        }
                    },
                }
            },
        },
    },
}


@pytest.mark.parametrize(
    ("ip_str", "expected"),
    [
        ("6.0", 6.0),
        ("6.1", pytest.approx(6.333, abs=0.001)),
        ("6.2", pytest.approx(6.667, abs=0.001)),
        ("0.0", 0.0),  # pitcher who faced batters but retired nobody — real
        ("", None),
        (None, None),
    ],
)
def test_parse_innings_pitched(ip_str, expected):
    assert _parse_innings_pitched(ip_str) == expected


def test_pitcher_stats_produces_one_row_per_pitcher_appearance(
    bronze_dir, silver_dir, write_boxscores, write_empty_bronze
):
    write_boxscores([{"game_pk": 111, "payload": SAMPLE_BOXSCORE}])
    for name in ("schedule", "plays", "transactions"):
        write_empty_bronze(name)

    build_silver(bronze_dir=bronze_dir, silver_dir=silver_dir)

    df = pd.read_parquet(silver_dir / "pitcher_game_stats" / "pitcher_game_stats.parquet")
    # Three pitchers, batter Aaron Judge filtered out
    assert len(df) == 3
    assert set(df["pitcher_id"]) == {1001, 1002, 3001}
    assert 2001 not in set(df["pitcher_id"])


def test_pitcher_stats_marks_starter_correctly(
    bronze_dir, silver_dir, write_boxscores, write_empty_bronze
):
    write_boxscores([{"game_pk": 111, "payload": SAMPLE_BOXSCORE}])
    for name in ("schedule", "plays", "transactions"):
        write_empty_bronze(name)

    build_silver(bronze_dir=bronze_dir, silver_dir=silver_dir)

    df = pd.read_parquet(silver_dir / "pitcher_game_stats" / "pitcher_game_stats.parquet")
    starters = df[df["is_starter"]]
    relievers = df[~df["is_starter"]]

    assert set(starters["pitcher_id"]) == {1001, 3001}  # first in each team's pitchers[]
    assert set(relievers["pitcher_id"]) == {1002}


def test_pitcher_stats_parses_innings_and_stats(
    bronze_dir, silver_dir, write_boxscores, write_empty_bronze
):
    write_boxscores([{"game_pk": 111, "payload": SAMPLE_BOXSCORE}])
    for name in ("schedule", "plays", "transactions"):
        write_empty_bronze(name)

    build_silver(bronze_dir=bronze_dir, silver_dir=silver_dir)

    df = pd.read_parquet(silver_dir / "pitcher_game_stats" / "pitcher_game_stats.parquet")
    cole = df[df["pitcher_id"] == 1001].iloc[0]
    assert cole["team_id"] == 147
    assert cole["innings_pitched"] == pytest.approx(6.333, abs=0.001)
    assert cole["hits_allowed"] == 4
    assert cole["earned_runs"] == 2
    assert cole["strikeouts"] == 8
    assert cole["pitches_thrown"] == 92


def test_pitcher_stats_includes_zero_ip_starter(
    bronze_dir, silver_dir, write_boxscores, write_empty_bronze
):
    """A starter who faced batters but retired nobody (IP='0.0') is still a
    real appearance — filtering on innings would drop the row. Regression test
    for a bug caught by the silver quality checks on real data."""
    payload = copy.deepcopy(SAMPLE_BOXSCORE)
    # Home starter Cole exits immediately: 0.0 IP but he still started.
    payload["teams"]["home"]["players"]["ID1001"]["stats"]["pitching"] = {
        "inningsPitched": "0.0",
        "hits": 1,
        "runs": 3,
        "earnedRuns": 3,
        "baseOnBalls": 2,
        "strikeOuts": 0,
        "numberOfPitches": 15,
    }
    write_boxscores([{"game_pk": 111, "payload": payload}])
    for name in ("schedule", "plays", "transactions"):
        write_empty_bronze(name)

    build_silver(bronze_dir=bronze_dir, silver_dir=silver_dir)

    df = pd.read_parquet(silver_dir / "pitcher_game_stats" / "pitcher_game_stats.parquet")
    cole = df[df["pitcher_id"] == 1001]
    assert len(cole) == 1  # not filtered out
    assert cole.iloc[0]["is_starter"]
    assert cole.iloc[0]["innings_pitched"] == 0.0
    assert cole.iloc[0]["pitches_thrown"] == 15


def test_pitcher_stats_dedups_across_partitions(
    bronze_dir, silver_dir, write_boxscores, write_empty_bronze
):
    """Re-ingest of same game: latest ingest wins for the (game_pk, pitcher_id) pair."""
    early = copy.deepcopy(SAMPLE_BOXSCORE)
    # Mid-game snapshot — Cole has only 3 innings
    early["teams"]["home"]["players"]["ID1001"]["stats"]["pitching"]["inningsPitched"] = "3.0"
    early["teams"]["home"]["players"]["ID1001"]["stats"]["pitching"]["strikeOuts"] = 4

    write_boxscores(
        [{"game_pk": 111, "payload": early}],
        target_date="2026-07-22",
        ingested_at=datetime(2026, 7, 22, 20, 0, tzinfo=UTC),
    )
    write_boxscores(
        [{"game_pk": 111, "payload": SAMPLE_BOXSCORE}],
        target_date="2026-07-23",
        ingested_at=datetime(2026, 7, 23, 4, 0, tzinfo=UTC),
    )
    for name in ("schedule", "plays", "transactions"):
        write_empty_bronze(name)

    build_silver(bronze_dir=bronze_dir, silver_dir=silver_dir)

    df = pd.read_parquet(silver_dir / "pitcher_game_stats" / "pitcher_game_stats.parquet")
    cole = df[df["pitcher_id"] == 1001]
    assert len(cole) == 1
    assert cole.iloc[0]["strikeouts"] == 8  # from the late ingest, not the mid-game 4
