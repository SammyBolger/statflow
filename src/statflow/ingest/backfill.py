"""Backfill historical bronze data for a range of MLB seasons.

Uses the season-level `/schedule` endpoint to enumerate every game date in
each season, then reuses the daily-ingest orchestration for each date.

Idempotent by default: dates whose schedule partition already exists on
disk are skipped, so the backfill can be interrupted and resumed.
"""

from __future__ import annotations

import argparse
import time
from datetime import date
from pathlib import Path

from statflow.config import BRONZE_DIR, MLB_SPORT_ID
from statflow.ingest.cli import run_daily_ingest
from statflow.ingest.mlb_api import MLBStatsAPI

# Regular + postseason. Skip spring training ("S") and All-Star ("A") — no
# meaningful team data for the model.
GAME_TYPES = "R,P"

DEFAULT_SLEEP_BETWEEN_DATES = 1.0


def dates_with_games(api: MLBStatsAPI, season: int) -> list[date]:
    """Return every date in a season that has at least one MLB game."""
    data = api.get(
        "schedule",
        params={
            "sportId": MLB_SPORT_ID,
            "season": season,
            "gameType": GAME_TYPES,
        },
    )
    unique: set[date] = set()
    for date_block in data.get("dates", []):
        if date_block.get("games"):
            unique.add(date.fromisoformat(date_block["date"]))
    return sorted(unique)


def _partition_exists(bronze_dir: Path, target_date: date) -> bool:
    return (
        bronze_dir / "schedule" / f"date={target_date.isoformat()}" / "schedule.parquet"
    ).exists()


def backfill(
    start_season: int,
    end_season: int,
    api: MLBStatsAPI | None = None,
    bronze_dir: Path = BRONZE_DIR,
    sleep_between_dates: float = DEFAULT_SLEEP_BETWEEN_DATES,
    skip_existing: bool = True,
) -> None:
    """Ingest every game date across a range of seasons (inclusive).

    Args:
        start_season, end_season: inclusive season range (e.g., 2024, 2026).
        sleep_between_dates: pause between date ingests to be polite to the API.
        skip_existing: if True, dates whose schedule partition already exists
            on disk are skipped. Set False to force re-ingest.
    """
    api = api or MLBStatsAPI()

    for season in range(start_season, end_season + 1):
        print(f"[season {season}] fetching schedule...")
        try:
            game_dates = dates_with_games(api, season)
        except Exception as exc:
            print(f"[season {season}] error fetching schedule: {exc}")
            continue
        print(f"[season {season}] {len(game_dates)} game dates")

        for i, target_date in enumerate(game_dates, start=1):
            if skip_existing and _partition_exists(bronze_dir, target_date):
                continue

            print(f"[{target_date}] ({i}/{len(game_dates)}) ingesting...")
            try:
                run_daily_ingest(target_date, api=api, base_dir=bronze_dir)
            except Exception as exc:
                # One bad date shouldn't kill the whole backfill.
                print(f"[{target_date}] error: {exc}")
            time.sleep(sleep_between_dates)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m statflow.ingest.backfill",
        description="Backfill historical MLB bronze data over a season range.",
    )
    parser.add_argument("--start-season", type=int, required=True)
    parser.add_argument("--end-season", type=int, required=True)
    parser.add_argument(
        "--sleep-between-dates",
        type=float,
        default=DEFAULT_SLEEP_BETWEEN_DATES,
        help="Seconds to pause between date ingests (default: %(default)s).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest even dates whose partition already exists.",
    )
    args = parser.parse_args(argv)

    backfill(
        args.start_season,
        args.end_season,
        sleep_between_dates=args.sleep_between_dates,
        skip_existing=not args.force,
    )


if __name__ == "__main__":
    main()
