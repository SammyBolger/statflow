"""CLI: run the daily Prefect flow locally.

Usage:
    uv run python -m statflow.flows                    # today
    uv run python -m statflow.flows --date 2026-07-27  # a specific date
"""

from __future__ import annotations

import argparse
from datetime import date

from statflow.flows.daily import daily_pipeline


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m statflow.flows",
        description=(
            "Run the statflow daily pipeline: ingest -> transform -> "
            "features -> predict -> outcomes."
        ),
    )
    parser.add_argument(
        "--date",
        type=date.fromisoformat,
        default=None,
        help="Date to run for (default: today), YYYY-MM-DD.",
    )
    args = parser.parse_args(argv)
    daily_pipeline(target_date=args.date)
