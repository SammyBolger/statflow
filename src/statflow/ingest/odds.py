"""Fetch closing (pre-game) betting lines from The Odds API.

Free tier at https://the-odds-api.com gives 500 requests/month, which
comfortably covers one daily pull. Set `ODDS_API_KEY` in the environment
to enable; when the env var is missing the ingest is a no-op (returns
None) so local dev without a key still works.

Bronze layout:
    data/bronze/odds/date=YYYY-MM-DD/odds.parquet

One row per game per bookmaker. Silver flattens to one row per game
with an average across bookmakers — that's the "market consensus" number
the dashboard compares the model against.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import requests

from statflow.config import BRONZE_DIR

ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"
ODDS_API_SPORT = "baseball_mlb"

# US regions get the major American books (DraftKings, FanDuel, MGM, etc.).
# h2h = moneyline (winner); totals = over/under total runs.
DEFAULT_REGIONS = "us"
DEFAULT_MARKETS = "h2h,totals"
DEFAULT_ODDS_FORMAT = "american"

DEFAULT_TIMEOUT = (5, 30)


def _api_key() -> str | None:
    """Return the API key from env, or None if not configured."""
    return os.environ.get("ODDS_API_KEY")


def fetch_odds(
    api_key: str | None = None,
    regions: str = DEFAULT_REGIONS,
    markets: str = DEFAULT_MARKETS,
) -> list[dict]:
    """GET the current MLB odds snapshot from The Odds API.

    Returns a list of games, each with a `bookmakers` array containing the
    quoted lines. Empty list if the API returns no upcoming games. Raises
    for HTTP errors so the caller sees the failure.
    """
    api_key = api_key or _api_key()
    if api_key is None:
        return []

    url = f"{ODDS_API_BASE_URL}/sports/{ODDS_API_SPORT}/odds"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": DEFAULT_ODDS_FORMAT,
    }
    resp = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def ingest_odds(
    target_date: date,
    api_key: str | None = None,
    out_dir: Path | None = None,
) -> Path | None:
    """Fetch today's odds snapshot and write it to bronze parquet.

    Returns the parquet path, or None if the API key isn't configured or
    the API returned no games. Re-running for the same date overwrites
    the existing file — idempotent.
    """
    api_key = api_key or _api_key()
    if api_key is None:
        # Silent no-op is intentional — local dev without a key should not
        # blow up the daily flow. Callers that care can check the return.
        return None

    games = fetch_odds(api_key=api_key)
    if not games:
        return None

    now = datetime.now(UTC)
    df = pd.DataFrame(
        {
            "odds_event_id": [g.get("id") for g in games],
            "date": target_date.isoformat(),
            "commence_time": [g.get("commence_time") for g in games],
            "home_team": [g.get("home_team") for g in games],
            "away_team": [g.get("away_team") for g in games],
            "payload": [json.dumps(g) for g in games],
            "fetched_at": now,
        }
    )

    out_dir = out_dir or BRONZE_DIR / "odds" / f"date={target_date.isoformat()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "odds.parquet"
    df.to_parquet(path, index=False)
    return path
