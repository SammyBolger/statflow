"""Tests for the silver `games` transform."""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from statflow.transform.runner import build_silver

SAMPLE_GAME = {
    "gamePk": 111,
    "gameDate": "2026-07-22T23:05:00Z",
    "season": 2026,
    "gameType": "R",
    "status": {"detailedState": "Final"},
    "teams": {
        "home": {
            "team": {"id": 147, "name": "New York Yankees"},
            "score": 5,
            "probablePitcher": {"id": 543037, "fullName": "Gerrit Cole"},
        },
        "away": {
            "team": {"id": 134, "name": "Pittsburgh Pirates"},
            "score": 3,
            "probablePitcher": {"id": 656605, "fullName": "Mitch Keller"},
        },
    },
    "venue": {"id": 3313, "name": "Yankee Stadium"},
    "dayNight": "night",
}


def _write_bronze_schedule(
    bronze_dir: Path,
    games: list[dict],
    target_date: str = "2026-07-22",
    ingested_at: datetime | None = None,
) -> Path:
    ingested_at = ingested_at or datetime.now(UTC)
    df = pd.DataFrame(
        {
            "game_pk": [g["gamePk"] for g in games],
            "game_date": target_date,
            "payload": [json.dumps(g) for g in games],
            "ingested_at": ingested_at,
        }
    )
    out = bronze_dir / "schedule" / f"date={target_date}" / "schedule.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out


def _empty_bronze(bronze_dir: Path) -> None:
    """Create empty partitions for the other bronze tables the runner registers."""
    for name in ("boxscores", "plays", "transactions"):
        out = bronze_dir / name / "date=2026-07-22" / f"{name}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        # Use a schema that matches what real ingest writes, so the view is valid.
        if name == "transactions":
            df = pd.DataFrame(
                {
                    "transaction_id": pd.Series([], dtype="int64"),
                    "date": pd.Series([], dtype="object"),
                    "payload": pd.Series([], dtype="object"),
                    "ingested_at": pd.Series([], dtype="datetime64[ns, UTC]"),
                }
            )
        else:
            df = pd.DataFrame(
                {
                    "game_pk": pd.Series([], dtype="int64"),
                    "payload": pd.Series([], dtype="object"),
                    "ingested_at": pd.Series([], dtype="datetime64[ns, UTC]"),
                }
            )
        df.to_parquet(out, index=False)


def test_games_extracts_all_fields(tmp_path):
    bronze = tmp_path / "bronze"
    silver = tmp_path / "silver"
    _write_bronze_schedule(bronze, [SAMPLE_GAME])
    _empty_bronze(bronze)

    build_silver(bronze_dir=bronze, silver_dir=silver)

    df = pd.read_parquet(silver / "games" / "games.parquet")
    assert len(df) == 1

    row = df.iloc[0]
    assert row["game_pk"] == 111
    assert pd.Timestamp(row["game_date"]).date().isoformat() == "2026-07-22"
    assert row["season"] == 2026
    assert row["game_type"] == "R"
    assert row["status"] == "Final"
    assert row["home_team_id"] == 147
    assert row["home_team_name"] == "New York Yankees"
    assert row["away_team_id"] == 134
    assert row["home_score"] == 5
    assert row["away_score"] == 3
    assert row["total_runs"] == 8
    assert row["home_win"] is True or row["home_win"] == 1
    assert row["venue_id"] == 3313
    assert row["home_probable_pitcher_id"] == 543037
    assert row["home_probable_pitcher_name"] == "Gerrit Cole"
    assert row["away_probable_pitcher_id"] == 656605


def test_games_home_win_false_when_home_loses(tmp_path):
    game = copy.deepcopy(SAMPLE_GAME)
    game["teams"]["home"]["score"] = 2
    game["teams"]["away"]["score"] = 7

    bronze = tmp_path / "bronze"
    silver = tmp_path / "silver"
    _write_bronze_schedule(bronze, [game])
    _empty_bronze(bronze)

    build_silver(bronze_dir=bronze, silver_dir=silver)

    df = pd.read_parquet(silver / "games" / "games.parquet")
    assert df.iloc[0]["home_win"] is False or df.iloc[0]["home_win"] == 0
    assert df.iloc[0]["total_runs"] == 9


def test_games_home_win_null_when_not_final(tmp_path):
    """In-progress and scheduled games have no target."""
    game = copy.deepcopy(SAMPLE_GAME)
    game["status"]["detailedState"] = "Scheduled"
    game["teams"]["home"]["score"] = None
    game["teams"]["away"]["score"] = None

    bronze = tmp_path / "bronze"
    silver = tmp_path / "silver"
    _write_bronze_schedule(bronze, [game])
    _empty_bronze(bronze)

    build_silver(bronze_dir=bronze, silver_dir=silver)

    df = pd.read_parquet(silver / "games" / "games.parquet")
    assert pd.isna(df.iloc[0]["home_win"])
    assert pd.isna(df.iloc[0]["total_runs"])


def test_games_deduplicates_across_bronze_partitions(tmp_path):
    """Same game_pk in two partitions: silver keeps the row with the latest ingested_at."""
    early = copy.deepcopy(SAMPLE_GAME)
    early["status"]["detailedState"] = "In Progress"
    early["teams"]["home"]["score"] = 2
    early["teams"]["away"]["score"] = 1

    late = copy.deepcopy(SAMPLE_GAME)  # Final, 5-3

    bronze = tmp_path / "bronze"
    silver = tmp_path / "silver"

    _write_bronze_schedule(
        bronze,
        [early],
        target_date="2026-07-22",
        ingested_at=datetime(2026, 7, 22, 20, 0, tzinfo=UTC),
    )
    # Simulate a second ingest under a different partition dir but for the same game.
    late_dir = bronze / "schedule" / "date=2026-07-23"
    late_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "game_pk": [late["gamePk"]],
            "game_date": "2026-07-22",
            "payload": [json.dumps(late)],
            "ingested_at": datetime(2026, 7, 23, 4, 0, tzinfo=UTC),
        }
    )
    df.to_parquet(late_dir / "schedule.parquet", index=False)

    _empty_bronze(bronze)

    build_silver(bronze_dir=bronze, silver_dir=silver)

    result = pd.read_parquet(silver / "games" / "games.parquet")
    assert len(result) == 1
    assert result.iloc[0]["status"] == "Final"
    assert result.iloc[0]["home_score"] == 5
