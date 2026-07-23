"""Fetch MLB game boxscores and persist them to bronze parquet.

Bronze layout:
    data/bronze/boxscores/date=YYYY-MM-DD/boxscores.parquet

One row per game. The `payload` column holds the raw JSON blob.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from statflow.config import BRONZE_DIR
from statflow.ingest.mlb_api import MLBStatsAPI


def fetch_boxscore(game_pk: int, api: MLBStatsAPI | None = None) -> dict:
    """Return the raw boxscore JSON for a single game."""
    api = api or MLBStatsAPI()
    return api.get(f"game/{game_pk}/boxscore")


def ingest_boxscores(
    target_date: date,
    game_pks: list[int],
    api: MLBStatsAPI | None = None,
    out_dir: Path | None = None,
) -> Path | None:
    """Fetch boxscores for the given games and write them to bronze parquet.

    Returns the path to the written parquet, or None if the game list is
    empty. Re-running overwrites the existing file — idempotent.
    """
    if not game_pks:
        return None

    api = api or MLBStatsAPI()
    now = datetime.now(UTC)

    rows = []
    for pk in game_pks:
        payload = fetch_boxscore(pk, api=api)
        rows.append(
            {
                "game_pk": pk,
                "payload": json.dumps(payload),
                "ingested_at": now,
            }
        )

    df = pd.DataFrame(rows)

    out_dir = out_dir or BRONZE_DIR / "boxscores" / f"date={target_date.isoformat()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "boxscores.parquet"
    df.to_parquet(path, index=False)
    return path
