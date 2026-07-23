"""Fetch MLB play-by-play data and persist it to bronze parquet.

Bronze layout:
    data/bronze/plays/date=YYYY-MM-DD/plays.parquet

One row per game. The `payload` column holds the raw JSON of the entire
play-by-play response for that game, which includes every at-bat and
every pitch.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from statflow.config import BRONZE_DIR
from statflow.ingest.mlb_api import MLBStatsAPI


def fetch_play_by_play(game_pk: int, api: MLBStatsAPI | None = None) -> dict:
    """Return the raw play-by-play JSON for a single game."""
    api = api or MLBStatsAPI()
    return api.get(f"game/{game_pk}/playByPlay")


def ingest_plays(
    target_date: date,
    game_pks: list[int],
    api: MLBStatsAPI | None = None,
    out_dir: Path | None = None,
) -> Path | None:
    """Fetch play-by-play for the given games and write to bronze parquet.

    Returns the path to the written parquet, or None if the game list is
    empty. Re-running overwrites the existing file — idempotent.
    """
    if not game_pks:
        return None

    api = api or MLBStatsAPI()
    now = datetime.now(UTC)

    rows = []
    for pk in game_pks:
        payload = fetch_play_by_play(pk, api=api)
        rows.append(
            {
                "game_pk": pk,
                "payload": json.dumps(payload),
                "ingested_at": now,
            }
        )

    df = pd.DataFrame(rows)

    out_dir = out_dir or BRONZE_DIR / "plays" / f"date={target_date.isoformat()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "plays.parquet"
    df.to_parquet(path, index=False)
    return path
