"""Fixtures for gold-feature tests: write minimal silver parquet.

Each test scenario builds a synthetic silver dataset small enough to reason
about (10-20 games), then asserts the resulting gold features are what the
window-function semantics *should* produce.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def silver_dir(tmp_path: Path) -> Path:
    return tmp_path / "silver"


@pytest.fixture
def gold_dir(tmp_path: Path) -> Path:
    return tmp_path / "gold"


def _write_parquet(silver_dir: Path, name: str, df: pd.DataFrame) -> Path:
    out = silver_dir / name / f"{name}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out


@pytest.fixture
def write_games(silver_dir: Path) -> Callable[[list[dict]], Path]:
    def _write(rows: list[dict]) -> Path:
        return _write_parquet(silver_dir, "games", pd.DataFrame(rows))

    return _write


@pytest.fixture
def write_team_game_stats(silver_dir: Path) -> Callable[[list[dict]], Path]:
    def _write(rows: list[dict]) -> Path:
        return _write_parquet(silver_dir, "team_game_stats", pd.DataFrame(rows))

    return _write


@pytest.fixture
def write_pitcher_game_stats(silver_dir: Path) -> Callable[[list[dict]], Path]:
    def _write(rows: list[dict]) -> Path:
        return _write_parquet(silver_dir, "pitcher_game_stats", pd.DataFrame(rows))

    return _write


_EMPTY_SCHEMAS = {
    "games": {
        "game_pk": "int64",
        "game_date": "object",
        "status": "object",
        "home_team_id": "int64",
        "away_team_id": "int64",
        "home_score": "int64",
        "away_score": "int64",
        "total_runs": "int64",
        "home_win": "bool",
    },
    "team_game_stats": {
        "game_pk": "int64",
        "team_id": "int64",
        "is_home": "bool",
        "runs": "int64",
    },
    "pitcher_game_stats": {
        "game_pk": "int64",
        "team_id": "int64",
        "pitcher_id": "int64",
        "is_starter": "bool",
        "innings_pitched": "float64",
        "earned_runs": "int64",
        "strikeouts": "int64",
    },
}


@pytest.fixture
def write_empty_silver(silver_dir: Path) -> Callable[[str], Path]:
    """Write an empty silver parquet with the right schema for tables a test doesn't populate."""

    def _write(name: str) -> Path:
        schema = _EMPTY_SCHEMAS[name]
        df = pd.DataFrame({col: pd.Series([], dtype=dtype) for col, dtype in schema.items()})
        return _write_parquet(silver_dir, name, df)

    return _write


@pytest.fixture(autouse=True)
def _prepopulate_empty_silver(silver_dir: Path):
    """Pre-populate every silver table as an empty placeholder so any feature
    SQL runs (returning zero rows if the test didn't provide data). Real
    fixture writes overwrite these — parquet writes are file-level replace.
    """
    for name, schema in _EMPTY_SCHEMAS.items():
        df = pd.DataFrame({col: pd.Series([], dtype=dtype) for col, dtype in schema.items()})
        _write_parquet(silver_dir, name, df)


# ---------------------------------------------------------------------------
# Helpers for building consistent (games, team_game_stats) fixtures.
# ---------------------------------------------------------------------------


def build_head_to_head_series(
    n_games: int,
    home_team: int = 100,
    away_team: int = 200,
    home_runs: list[int] | None = None,
    away_runs: list[int] | None = None,
    start_date: date = date(2024, 4, 1),
) -> tuple[list[dict], list[dict]]:
    """Build `n_games` all Final between `home_team` and `away_team`.

    Alternates who's home each game so both teams appear on both sides.
    Returns (games_rows, team_stats_rows) suitable for writing to silver.

    `home_runs[i]` and `away_runs[i]` are the *home* and *away* team's
    runs in game i regardless of which real team is home that game.
    """
    home_runs = home_runs or [4] * n_games
    away_runs = away_runs or [3] * n_games

    games: list[dict] = []
    stats: list[dict] = []
    for i in range(n_games):
        game_pk = 1000 + i
        game_date = start_date + timedelta(days=i)
        # Alternate home/away — both teams get to be home about half the time.
        h_team, a_team = (home_team, away_team) if i % 2 == 0 else (away_team, home_team)
        h_runs, a_runs = home_runs[i], away_runs[i]
        games.append(
            {
                "game_pk": game_pk,
                "game_date": game_date,
                "status": "Final",
                "home_team_id": h_team,
                "away_team_id": a_team,
                "home_score": h_runs,
                "away_score": a_runs,
                "total_runs": h_runs + a_runs,
                "home_win": h_runs > a_runs,
            }
        )
        stats.append({"game_pk": game_pk, "team_id": h_team, "is_home": True, "runs": h_runs})
        stats.append({"game_pk": game_pk, "team_id": a_team, "is_home": False, "runs": a_runs})
    return games, stats


@pytest.fixture
def head_to_head_series():
    return build_head_to_head_series
