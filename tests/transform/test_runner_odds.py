"""Tests for the silver odds builder.

Exercises the moneyline de-vig math + the join back to silver.games.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from statflow.transform.runner import _build_odds_silver, _summarize_game_odds, build_silver


def _write_odds_bronze(bronze_dir: Path, rows: list[dict], target_date: str = "2026-07-30") -> Path:
    now = datetime.now(UTC)
    df = pd.DataFrame(
        {
            "odds_event_id": [r["id"] for r in rows],
            "date": target_date,
            "commence_time": [r.get("commence_time") for r in rows],
            "home_team": [r.get("home_team") for r in rows],
            "away_team": [r.get("away_team") for r in rows],
            "payload": [json.dumps(r) for r in rows],
            "fetched_at": now,
        }
    )
    out = bronze_dir / "odds" / f"date={target_date}" / "odds.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out


def _write_games_silver(silver_dir: Path, rows: list[dict]) -> Path:
    df = pd.DataFrame(rows)
    out = silver_dir / "games" / "games.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out


def test_summarize_game_odds_devigs_moneyline_to_sum_one():
    payload = {
        "home_team": "Los Angeles Dodgers",
        "away_team": "San Francisco Giants",
        "bookmakers": [
            {
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Los Angeles Dodgers", "price": -150},
                            {"name": "San Francisco Giants", "price": 130},
                        ],
                    },
                    {
                        "key": "totals",
                        "outcomes": [
                            {"name": "Over", "price": -110, "point": 8.5},
                            {"name": "Under", "price": -110, "point": 8.5},
                        ],
                    },
                ]
            }
        ],
    }
    got = _summarize_game_odds(payload)
    assert got["market_home_win_prob"] + got["market_away_win_prob"] == 1.0
    # Home is favored (-150), so their prob should be above 50%.
    assert got["market_home_win_prob"] > 0.5
    assert got["market_total_line"] == 8.5
    assert got["n_books_moneyline"] == 1


def test_summarize_game_odds_averages_across_books():
    """Two books, different prices — result should be the mean."""
    payload = {
        "home_team": "H",
        "away_team": "A",
        "bookmakers": [
            {
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "H", "price": -110},
                            {"name": "A", "price": -110},
                        ],
                    },
                    {"key": "totals", "outcomes": [{"name": "Over", "point": 9.0}]},
                ]
            },
            {
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "H", "price": -120},
                            {"name": "A", "price": 100},
                        ],
                    },
                    {"key": "totals", "outcomes": [{"name": "Over", "point": 9.5}]},
                ]
            },
        ],
    }
    got = _summarize_game_odds(payload)
    assert got["n_books_moneyline"] == 2
    assert got["market_total_line"] == 9.25


def test_summarize_game_odds_handles_missing_market():
    payload = {"home_team": "H", "away_team": "A", "bookmakers": []}
    got = _summarize_game_odds(payload)
    assert got["market_home_win_prob"] is None
    assert got["market_total_line"] is None


def test_build_odds_silver_no_bronze_returns_none(tmp_path):
    bronze = tmp_path / "bronze"
    silver = tmp_path / "silver"
    bronze.mkdir()
    silver.mkdir()
    assert _build_odds_silver(bronze, silver) is None


def test_build_odds_silver_joins_to_game_pk(tmp_path):
    bronze = tmp_path / "bronze"
    silver = tmp_path / "silver"

    _write_odds_bronze(
        bronze,
        [
            {
                "id": "abc",
                "commence_time": "2026-07-30T23:05:00Z",
                "home_team": "Los Angeles Dodgers",
                "away_team": "San Francisco Giants",
                "bookmakers": [
                    {
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Los Angeles Dodgers", "price": -150},
                                    {"name": "San Francisco Giants", "price": 130},
                                ],
                            },
                            {"key": "totals", "outcomes": [{"name": "Over", "point": 8.5}]},
                        ]
                    }
                ],
            }
        ],
    )
    _write_games_silver(
        silver,
        [
            {
                "game_pk": 999888,
                "game_date": pd.Timestamp("2026-07-30"),
                "home_team_name": "Los Angeles Dodgers",
                "away_team_name": "San Francisco Giants",
            }
        ],
    )

    path = _build_odds_silver(bronze, silver)
    assert path is not None

    df = pd.read_parquet(path)
    assert len(df) == 1
    assert df.iloc[0]["game_pk"] == 999888
    assert df.iloc[0]["market_home_win_prob"] > 0.5
    assert df.iloc[0]["market_total_line"] == 8.5


def test_build_odds_silver_leaves_game_pk_null_when_names_mismatch(tmp_path):
    """A team-name mismatch shouldn't crash — it should surface as NULL game_pk."""
    bronze = tmp_path / "bronze"
    silver = tmp_path / "silver"

    _write_odds_bronze(
        bronze,
        [
            {
                "id": "xyz",
                "commence_time": "2026-07-30T23:05:00Z",
                "home_team": "LA Dodgers",  # note: differs from MLB API's "Los Angeles Dodgers"
                "away_team": "SF Giants",
                "bookmakers": [],
            }
        ],
    )
    _write_games_silver(
        silver,
        [
            {
                "game_pk": 111222,
                "game_date": pd.Timestamp("2026-07-30"),
                "home_team_name": "Los Angeles Dodgers",
                "away_team_name": "San Francisco Giants",
            }
        ],
    )

    path = _build_odds_silver(bronze, silver)
    assert path is not None
    df = pd.read_parquet(path)
    assert pd.isna(df.iloc[0]["game_pk"])


def test_build_silver_includes_odds_when_bronze_present(tmp_path, write_empty_bronze):
    """build_silver()'s pipeline should call the odds builder without errors
    even when only empty bronze tables are present for the SQL sources."""
    bronze = tmp_path / "bronze"
    silver = tmp_path / "silver"

    write_empty_bronze("schedule")
    write_empty_bronze("boxscores")
    write_empty_bronze("plays")
    write_empty_bronze("transactions")
    # No odds bronze — should be a silent no-op.

    build_silver(bronze_dir=bronze, silver_dir=silver)
    # Odds silver should NOT exist (no bronze odds to build from).
    assert not (silver / "odds" / "odds.parquet").exists()
