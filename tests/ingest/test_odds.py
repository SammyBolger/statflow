"""Tests for the odds ingest module.

Covers three shapes: (1) no API key → no-op, (2) API returns games →
bronze parquet with the right columns, (3) API returns no games → no
file written.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import responses

from statflow.ingest.odds import ODDS_API_BASE_URL, ODDS_API_SPORT, fetch_odds, ingest_odds


def _sample_game(home: str = "Los Angeles Dodgers", away: str = "San Francisco Giants") -> dict:
    return {
        "id": "abc123",
        "sport_key": ODDS_API_SPORT,
        "commence_time": "2026-07-30T23:05:00Z",
        "home_team": home,
        "away_team": away,
        "bookmakers": [
            {
                "key": "draftkings",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": home, "price": -150},
                            {"name": away, "price": 130},
                        ],
                    },
                    {
                        "key": "totals",
                        "outcomes": [
                            {"name": "Over", "price": -110, "point": 8.5},
                            {"name": "Under", "price": -110, "point": 8.5},
                        ],
                    },
                ],
            }
        ],
    }


def test_fetch_odds_returns_empty_list_without_api_key(monkeypatch):
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    # No HTTP call should be made — no responses registered means any call would 500.
    assert fetch_odds() == []


def test_ingest_odds_noop_without_api_key(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    result = ingest_odds(date(2026, 7, 30), out_dir=tmp_path)
    assert result is None
    # No file should have been created.
    assert not any(tmp_path.iterdir())


@responses.activate
def test_ingest_odds_writes_bronze_parquet(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", "fake_key")
    payload = [_sample_game(), _sample_game(home="New York Yankees", away="Boston Red Sox")]
    responses.add(
        responses.GET,
        f"{ODDS_API_BASE_URL}/sports/{ODDS_API_SPORT}/odds",
        json=payload,
        status=200,
    )

    path = ingest_odds(date(2026, 7, 30), out_dir=tmp_path)
    assert path is not None
    assert path.exists()

    df = pd.read_parquet(path)
    assert len(df) == 2
    assert set(df.columns) >= {
        "odds_event_id",
        "date",
        "commence_time",
        "home_team",
        "away_team",
        "payload",
        "fetched_at",
    }
    # Payload column keeps the raw JSON so silver can re-extract.
    round_tripped = json.loads(df.iloc[0]["payload"])
    assert round_tripped["home_team"] == "Los Angeles Dodgers"


@responses.activate
def test_ingest_odds_returns_none_on_empty_response(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", "fake_key")
    responses.add(
        responses.GET,
        f"{ODDS_API_BASE_URL}/sports/{ODDS_API_SPORT}/odds",
        json=[],
        status=200,
    )

    result = ingest_odds(date(2026, 7, 30), out_dir=tmp_path)
    assert result is None
    assert not any(tmp_path.iterdir())


def test_ingest_odds_accepts_explicit_key_bypassing_env(tmp_path: Path, monkeypatch):
    """Passing an api_key explicitly should work even if ODDS_API_KEY is missing."""
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    with responses.RequestsMock() as rsps:
        rsps.add(
            responses.GET,
            f"{ODDS_API_BASE_URL}/sports/{ODDS_API_SPORT}/odds",
            json=[_sample_game()],
            status=200,
        )
        path = ingest_odds(date(2026, 7, 30), api_key="explicit_key", out_dir=tmp_path)
    assert path is not None
    assert path.exists()


@pytest.mark.parametrize(
    ("price", "expected_low", "expected_high"),
    [
        (-150, 0.59, 0.61),  # ~60%
        (150, 0.39, 0.41),  # ~40%
        (-110, 0.52, 0.53),  # ~52.4%
        (100, 0.499, 0.501),  # 50%
    ],
)
def test_american_to_implied_prob(price, expected_low, expected_high):
    """Sanity-check the odds converter used inside the silver builder."""
    from statflow.transform.runner import _american_to_implied_prob

    got = _american_to_implied_prob(price)
    assert expected_low <= got <= expected_high, got
