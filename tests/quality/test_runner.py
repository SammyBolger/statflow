from datetime import UTC, datetime

import pandas as pd

from statflow.quality.runner import ALL_CHECKS, run_checks, write_report


def test_run_checks_returns_one_result_per_registered_check(silver_dir, valid_silver):
    results = run_checks(silver_dir)
    assert len(results) == len(ALL_CHECKS)
    assert {r.name for r in results} == {c(silver_dir).name for c in ALL_CHECKS}


def test_write_report_creates_parquet(silver_dir, valid_silver):
    results = run_checks(silver_dir)
    path = write_report(results, silver_dir=silver_dir, ran_at=datetime(2026, 7, 23, tzinfo=UTC))

    assert path == silver_dir / "_quality" / "checks.parquet"
    df = pd.read_parquet(path)
    assert len(df) == len(ALL_CHECKS)
    assert set(df.columns) == {"name", "passed", "details", "ran_at"}


def test_write_report_appends_across_runs(silver_dir, valid_silver):
    results = run_checks(silver_dir)
    write_report(results, silver_dir=silver_dir, ran_at=datetime(2026, 7, 23, tzinfo=UTC))
    write_report(results, silver_dir=silver_dir, ran_at=datetime(2026, 7, 24, tzinfo=UTC))

    df = pd.read_parquet(silver_dir / "_quality" / "checks.parquet")
    assert len(df) == 2 * len(ALL_CHECKS)
    assert set(pd.to_datetime(df["ran_at"]).dt.date) == {
        datetime(2026, 7, 23).date(),
        datetime(2026, 7, 24).date(),
    }
