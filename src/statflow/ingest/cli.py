"""Orchestrate a single date's bronze ingest end-to-end.

The CLI runs the four ingest steps in dependency order:
    1. schedule    (need game_pks for the rest)
    2. boxscores   (per game)
    3. plays       (per game)
    4. transactions (independent of games — runs even on off-days)
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from statflow.config import BRONZE_DIR
from statflow.ingest.boxscores import ingest_boxscores
from statflow.ingest.mlb_api import MLBStatsAPI
from statflow.ingest.plays import ingest_plays
from statflow.ingest.schedule import ingest_schedule
from statflow.ingest.transactions import ingest_transactions


def _partition_dir(base_dir: Path, table: str, target_date: date) -> Path:
    return base_dir / table / f"date={target_date.isoformat()}"


def run_daily_ingest(
    target_date: date,
    api: MLBStatsAPI | None = None,
    base_dir: Path = BRONZE_DIR,
) -> None:
    """Ingest schedule, boxscores, plays, and transactions for one date."""
    api = api or MLBStatsAPI()

    print(f"[schedule] {target_date}")
    schedule_path = ingest_schedule(
        target_date,
        api=api,
        out_dir=_partition_dir(base_dir, "schedule", target_date),
    )
    if schedule_path is None:
        print("  no games")
        game_pks: list[int] = []
    else:
        print(f"  -> {schedule_path}")
        game_pks = pd.read_parquet(schedule_path)["game_pk"].tolist()

    if game_pks:
        print(f"[boxscores] {len(game_pks)} games")
        box_path = ingest_boxscores(
            target_date,
            game_pks,
            api=api,
            out_dir=_partition_dir(base_dir, "boxscores", target_date),
        )
        print(f"  -> {box_path}")

        print(f"[plays] {len(game_pks)} games")
        plays_path = ingest_plays(
            target_date,
            game_pks,
            api=api,
            out_dir=_partition_dir(base_dir, "plays", target_date),
        )
        print(f"  -> {plays_path}")

    print(f"[transactions] {target_date}")
    txn_path = ingest_transactions(
        target_date,
        api=api,
        out_dir=_partition_dir(base_dir, "transactions", target_date),
    )
    if txn_path is None:
        print("  none")
    else:
        print(f"  -> {txn_path}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m statflow.ingest",
        description="Ingest MLB Stats API data for a single date to bronze parquet.",
    )
    parser.add_argument(
        "--date",
        type=date.fromisoformat,
        required=True,
        help="Date to ingest, YYYY-MM-DD.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    run_daily_ingest(args.date)
