"""Fetch MLB game schedules and persist them to bronze parquet.

Bronze layout:
    data/bronze/schedule/date=YYYY-MM-DD/schedule.parquet

Each row is one game. The `payload` column holds the raw JSON blob for
that game so downstream layers can extract whatever fields they need
without another API call.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from statflow.config import BRONZE_DIR, MLB_SPORT_ID
from statflow.ingest.mlb_api import MLBStatsAPI

# `hydrate` tells the schedule endpoint to embed related objects so we
# don't have to make separate calls for each. probablePitcher is the big
# one — it's the strongest injury-adjacent signal we can get for free.
SCHEDULE_HYDRATE = "probablePitcher,team,venue,linescore"


def fetch_schedule(target_date: date, api: MLBStatsAPI | None = None) -> list[dict]:
    """Return the raw game records for a single date."""
    api = api or MLBStatsAPI()
    data = api.get(
        "schedule",
        params={
            "sportId": MLB_SPORT_ID,
            "date": target_date.isoformat(),
            "hydrate": SCHEDULE_HYDRATE,
        },
    )
    games: list[dict] = []
    for date_block in data.get("dates", []):
        games.extend(date_block.get("games", []))
    return games


def ingest_schedule(
    target_date: date,
    api: MLBStatsAPI | None = None,
    out_dir: Path | None = None,
) -> Path | None:
    """Fetch a date's schedule and write it to bronze parquet.

    Returns the path to the written parquet, or None if the date had no
    games (e.g., All-Star break). Re-running for the same date overwrites
    the existing file — the operation is idempotent.
    """
    games = fetch_schedule(target_date, api=api)
    if not games:
        return None

    now = datetime.now(UTC)
    df = pd.DataFrame(
        {
            "game_pk": [g["gamePk"] for g in games],
            "game_date": target_date.isoformat(),
            "payload": [json.dumps(g) for g in games],
            "ingested_at": now,
        }
    )

    out_dir = out_dir or BRONZE_DIR / "schedule" / f"date={target_date.isoformat()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "schedule.parquet"
    df.to_parquet(path, index=False)
    return path
