"""Fixtures for quality-check tests: write minimal silver parquet files."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def silver_dir(tmp_path: Path) -> Path:
    return tmp_path / "silver"


@pytest.fixture
def write_games(silver_dir: Path) -> Callable[[list[dict]], Path]:
    def _write(rows: list[dict]) -> Path:
        df = pd.DataFrame(rows)
        out = silver_dir / "games" / "games.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        return out

    return _write


@pytest.fixture
def write_team_stats(silver_dir: Path) -> Callable[[list[dict]], Path]:
    def _write(rows: list[dict]) -> Path:
        df = pd.DataFrame(rows)
        out = silver_dir / "team_game_stats" / "team_game_stats.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        return out

    return _write


@pytest.fixture
def write_pitcher_stats(silver_dir: Path) -> Callable[[list[dict]], Path]:
    def _write(rows: list[dict]) -> Path:
        df = pd.DataFrame(rows)
        out = silver_dir / "pitcher_game_stats" / "pitcher_game_stats.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        return out

    return _write


def _final_game(game_pk: int, home_score: int = 5, away_score: int = 3) -> dict:
    return {
        "game_pk": game_pk,
        "game_date": datetime(2026, 7, 22).date(),
        "status": "Final",
        "home_team_id": 147,
        "away_team_id": 134,
        "home_score": home_score,
        "away_score": away_score,
        "total_runs": home_score + away_score,
        "home_win": home_score > away_score,
    }


def _team_stats_row(game_pk: int, team_id: int, is_home: bool) -> dict:
    return {"game_pk": game_pk, "team_id": team_id, "is_home": is_home}


def _pitcher_row(game_pk: int, pitcher_id: int, team_id: int, is_starter: bool) -> dict:
    return {
        "game_pk": game_pk,
        "pitcher_id": pitcher_id,
        "team_id": team_id,
        "is_starter": is_starter,
    }


@pytest.fixture
def valid_silver(write_games, write_team_stats, write_pitcher_stats):
    """A minimal but fully-consistent silver: 2 Final games, all checks should pass."""
    write_games([_final_game(111, 5, 3), _final_game(222, 2, 7)])
    write_team_stats(
        [
            _team_stats_row(111, 147, True),
            _team_stats_row(111, 134, False),
            _team_stats_row(222, 147, True),
            _team_stats_row(222, 134, False),
        ]
    )
    write_pitcher_stats(
        [
            _pitcher_row(111, 1001, 147, True),
            _pitcher_row(111, 3001, 134, True),
            _pitcher_row(111, 1002, 147, False),
            _pitcher_row(222, 1001, 147, True),
            _pitcher_row(222, 3001, 134, True),
        ]
    )
    return None


# Expose the small helpers to tests that want to build failing variants.
@pytest.fixture
def make_final_game():
    return _final_game


@pytest.fixture
def make_team_stats_row():
    return _team_stats_row


@pytest.fixture
def make_pitcher_row():
    return _pitcher_row
