"""Data quality checks over the silver layer.

Each check is a plain function that reads one or two silver parquet files
and returns a `CheckResult`. Registering a new check = write a function
and add it to `ALL_CHECKS` in runner.py.

All checks are non-fatal — they return a result object. The runner
collects them, writes a report, and decides (via `--fail-on-error`)
whether to exit non-zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class CheckResult:
    name: str
    passed: bool
    details: str


def _read(silver_dir: Path, table: str) -> pd.DataFrame:
    return pd.read_parquet(silver_dir / table / f"{table}.parquet")


# ---------------------------------------------------------------------------
# games
# ---------------------------------------------------------------------------


def check_games_pk_unique(silver_dir: Path) -> CheckResult:
    df = _read(silver_dir, "games")
    dupes = int(df["game_pk"].duplicated().sum())
    return CheckResult(
        name="games.pk_unique",
        passed=dupes == 0,
        details=f"{dupes} duplicate game_pk rows out of {len(df)}",
    )


def check_games_no_null_ids(silver_dir: Path) -> CheckResult:
    df = _read(silver_dir, "games")
    cols = ["game_pk", "game_date", "home_team_id", "away_team_id"]
    nulls = {c: int(df[c].isna().sum()) for c in cols}
    total = sum(nulls.values())
    return CheckResult(
        name="games.no_null_ids",
        passed=total == 0,
        details=f"nulls: {nulls}",
    )


def check_games_final_has_scores(silver_dir: Path) -> CheckResult:
    df = _read(silver_dir, "games")
    finals = df[df["status"] == "Final"]
    missing = int(finals["home_score"].isna().sum() + finals["away_score"].isna().sum())
    return CheckResult(
        name="games.final_has_scores",
        passed=missing == 0,
        details=f"{missing} missing scores among {len(finals)} Final games",
    )


def check_games_total_runs_consistent(silver_dir: Path) -> CheckResult:
    df = _read(silver_dir, "games")
    finals = df[df["status"] == "Final"].copy()
    expected = finals["home_score"] + finals["away_score"]
    mismatched = int((finals["total_runs"] != expected).sum())
    return CheckResult(
        name="games.total_runs_consistent",
        passed=mismatched == 0,
        details=f"{mismatched} rows where total_runs != home_score + away_score",
    )


# ---------------------------------------------------------------------------
# team_game_stats
# ---------------------------------------------------------------------------


def check_team_stats_pk_unique(silver_dir: Path) -> CheckResult:
    df = _read(silver_dir, "team_game_stats")
    dupes = int(df.duplicated(subset=["game_pk", "team_id"]).sum())
    return CheckResult(
        name="team_stats.pk_unique",
        passed=dupes == 0,
        details=f"{dupes} duplicate (game_pk, team_id) rows out of {len(df)}",
    )


def check_team_stats_two_rows_per_final_game(silver_dir: Path) -> CheckResult:
    """Every Final game should have exactly two rows (home + away) in team_game_stats."""
    games = _read(silver_dir, "games")
    stats = _read(silver_dir, "team_game_stats")

    finals = games[games["status"] == "Final"]
    counts = stats[stats["game_pk"].isin(finals["game_pk"])].groupby("game_pk").size()
    bad = int((counts != 2).sum())
    missing = int(set(finals["game_pk"]).difference(counts.index).__len__())
    total_issues = bad + missing
    return CheckResult(
        name="team_stats.two_rows_per_final_game",
        passed=total_issues == 0,
        details=(
            f"{bad} Final games with row-count != 2, "
            f"{missing} Final games with no team_game_stats rows at all"
        ),
    )


# ---------------------------------------------------------------------------
# pitcher_game_stats
# ---------------------------------------------------------------------------


def check_pitcher_stats_pk_unique(silver_dir: Path) -> CheckResult:
    df = _read(silver_dir, "pitcher_game_stats")
    dupes = int(df.duplicated(subset=["game_pk", "pitcher_id"]).sum())
    return CheckResult(
        name="pitcher_stats.pk_unique",
        passed=dupes == 0,
        details=f"{dupes} duplicate (game_pk, pitcher_id) rows out of {len(df)}",
    )


def check_pitcher_stats_two_starters_per_final_game(silver_dir: Path) -> CheckResult:
    """Every Final game should have exactly two starters (one per team)."""
    games = _read(silver_dir, "games")
    pitchers = _read(silver_dir, "pitcher_game_stats")

    finals = games[games["status"] == "Final"]
    starters = pitchers[pitchers["is_starter"] & pitchers["game_pk"].isin(finals["game_pk"])]
    counts = starters.groupby("game_pk").size()
    bad = int((counts != 2).sum())
    missing = int(set(finals["game_pk"]).difference(counts.index).__len__())
    total_issues = bad + missing
    return CheckResult(
        name="pitcher_stats.two_starters_per_final_game",
        passed=total_issues == 0,
        details=(
            f"{bad} Final games with != 2 starters, "
            f"{missing} Final games with no starter rows at all"
        ),
    )
