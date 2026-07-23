"""Fetch MLB transactions (IL moves, trades, etc.) and persist to bronze.

Bronze layout:
    data/bronze/transactions/date=YYYY-MM-DD/transactions.parquet

One row per transaction. The transactions feed is what lets us
reconstruct point-in-time roster state historically — critical for
building injury/availability features without data leakage.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from statflow.config import BRONZE_DIR
from statflow.ingest.mlb_api import MLBStatsAPI


def fetch_transactions(target_date: date, api: MLBStatsAPI | None = None) -> list[dict]:
    """Return raw transaction records for a single date."""
    api = api or MLBStatsAPI()
    iso = target_date.isoformat()
    data = api.get("transactions", params={"startDate": iso, "endDate": iso})
    return data.get("transactions", [])


def ingest_transactions(
    target_date: date,
    api: MLBStatsAPI | None = None,
    out_dir: Path | None = None,
) -> Path | None:
    """Fetch a date's transactions and write to bronze parquet.

    Returns the path to the written parquet, or None if no transactions
    occurred that day. Re-running overwrites — idempotent.
    """
    txns = fetch_transactions(target_date, api=api)
    if not txns:
        return None

    now = datetime.now(UTC)
    df = pd.DataFrame(
        {
            "transaction_id": [t["id"] for t in txns],
            "date": target_date.isoformat(),
            "payload": [json.dumps(t) for t in txns],
            "ingested_at": now,
        }
    )

    out_dir = out_dir or BRONZE_DIR / "transactions" / f"date={target_date.isoformat()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "transactions.parquet"
    df.to_parquet(path, index=False)
    return path
