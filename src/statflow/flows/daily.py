"""The daily statflow pipeline as a Prefect flow.

Wires ingest -> transform -> features -> predict -> outcomes into a single
run. Each step is a Prefect task, so we get:
  * Retries on transient failures (ingest hits the network)
  * Task-level timing + logging in Prefect's runtime
  * Clean dependency graph — Prefect knows transform depends on ingest

Runs locally without needing Prefect Cloud or a server. Scheduling happens
externally via GitHub Actions cron.
"""

from __future__ import annotations

from datetime import date, timedelta

from prefect import flow, get_run_logger, task

from statflow.features.runner import build_features
from statflow.ingest.cli import run_daily_ingest
from statflow.models.outcomes import build_prediction_outcomes
from statflow.models.predict import predict_for_date
from statflow.transform.runner import build_silver

# Re-ingest today + this many prior days each run. Yesterday is the important
# one: its 7am-scaffold boxscore (all zeros, no Final status) needs to be
# overwritten with the finalized version so prediction_outcomes can grade
# yesterday's predictions and rolling stats stop being polluted by scaffolds.
# 2 prior days is enough buffer for late-completing West Coast games and the
# occasional missed daily-flow run.
INGEST_BACKFILL_DAYS = 3


@task(retries=3, retry_delay_seconds=[10, 30, 60], log_prints=True)
def ingest(target_date: date, backfill_days: int = INGEST_BACKFILL_DAYS) -> None:
    """Refresh bronze for `target_date` and the last `backfill_days - 1` prior days."""
    logger = get_run_logger()
    for delta in range(backfill_days - 1, -1, -1):
        d = target_date - timedelta(days=delta)
        logger.info(f"ingesting {d}")
        run_daily_ingest(d)


@task(log_prints=True)
def transform() -> None:
    get_run_logger().info("rebuilding silver")
    build_silver()


@task(log_prints=True)
def features() -> None:
    get_run_logger().info("rebuilding gold features")
    build_features()


@task(log_prints=True)
def predict(target_date: date) -> str | None:
    logger = get_run_logger()
    path = predict_for_date(target_date)
    if path is None:
        logger.info(f"no games on {target_date} — nothing to predict")
        return None
    logger.info(f"wrote predictions to {path}")
    return str(path)


@task(log_prints=True)
def refresh_outcomes() -> str | None:
    logger = get_run_logger()
    path = build_prediction_outcomes()
    if path is None:
        logger.info("no prediction_outcomes to write yet")
        return None
    logger.info(f"wrote prediction_outcomes to {path}")
    return str(path)


@flow(name="statflow-daily")
def daily_pipeline(target_date: date | None = None) -> None:
    """Run the full daily pipeline for a single date (defaults to today)."""
    target_date = target_date or date.today()
    ingest(target_date)
    transform()
    features()
    predict(target_date)
    refresh_outcomes()
