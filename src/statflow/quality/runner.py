"""Run all data-quality checks against silver and persist a report.

The report is a parquet table that grows over time — one row per (run, check).
Useful downstream for the monitoring dashboard: "did we ever fail this check,
and when?"
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from statflow.config import SILVER_DIR
from statflow.quality.checks import (
    CheckResult,
    check_games_final_has_scores,
    check_games_no_null_ids,
    check_games_pk_unique,
    check_games_total_runs_consistent,
    check_pitcher_stats_pk_unique,
    check_pitcher_stats_two_starters_per_final_game,
    check_team_stats_pk_unique,
    check_team_stats_two_rows_per_final_game,
)

ALL_CHECKS = [
    check_games_pk_unique,
    check_games_no_null_ids,
    check_games_final_has_scores,
    check_games_total_runs_consistent,
    check_team_stats_pk_unique,
    check_team_stats_two_rows_per_final_game,
    check_pitcher_stats_pk_unique,
    check_pitcher_stats_two_starters_per_final_game,
]


def run_checks(silver_dir: Path = SILVER_DIR) -> list[CheckResult]:
    """Run every check and return the results."""
    return [check(silver_dir) for check in ALL_CHECKS]


def write_report(
    results: list[CheckResult],
    silver_dir: Path = SILVER_DIR,
    ran_at: datetime | None = None,
) -> Path:
    """Append this run's results to the quality report parquet."""
    ran_at = ran_at or datetime.now(UTC)
    df = pd.DataFrame([{**asdict(r), "ran_at": ran_at} for r in results])

    out_dir = silver_dir / "_quality"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "checks.parquet"

    if path.exists():
        prior = pd.read_parquet(path)
        df = pd.concat([prior, df], ignore_index=True)
    df.to_parquet(path, index=False)
    return path
