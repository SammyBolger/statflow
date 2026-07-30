"""Tests for the daily Prefect flow.

We test the flow by monkey-patching the underlying pipeline functions with
recorders — we care that (1) the flow calls each step and (2) it passes
target_date to the steps that need it. Actual behavior of ingest/transform/
etc. is covered by their own module tests.
"""

from __future__ import annotations

from datetime import date, timedelta

from statflow.flows.daily import INGEST_BACKFILL_DAYS, daily_pipeline


def test_daily_pipeline_calls_all_steps_in_order(monkeypatch):
    called: list[tuple[str, object]] = []

    def _rec(name):
        def _fn(*args, **kwargs):
            called.append((name, args))

        return _fn

    monkeypatch.setattr("statflow.flows.daily.run_daily_ingest", _rec("ingest"))
    monkeypatch.setattr("statflow.flows.daily.build_silver", _rec("silver"))
    monkeypatch.setattr("statflow.flows.daily.build_features", _rec("features"))
    monkeypatch.setattr(
        "statflow.flows.daily.predict_for_date",
        lambda d: called.append(("predict", (d,))) or None,
    )
    monkeypatch.setattr(
        "statflow.flows.daily.build_prediction_outcomes",
        lambda: called.append(("outcomes", ())) or None,
    )

    daily_pipeline(target_date=date(2026, 7, 27))

    names = [c[0] for c in called]
    # ingest runs once per day in the rolling window, then the rest of the pipeline once.
    assert names == ["ingest"] * INGEST_BACKFILL_DAYS + [
        "silver",
        "features",
        "predict",
        "outcomes",
    ]

    # The rolling window is oldest -> today so yesterday's data is on disk
    # before today gets refreshed.
    ingest_dates = [c[1][0] for c in called if c[0] == "ingest"]
    assert ingest_dates == [
        date(2026, 7, 27) - timedelta(days=delta)
        for delta in range(INGEST_BACKFILL_DAYS - 1, -1, -1)
    ]
    # predict receives the target date (not any of the earlier backfill dates)
    predict_call = next(c for c in called if c[0] == "predict")
    assert predict_call[1] == (date(2026, 7, 27),)


def test_daily_pipeline_defaults_to_today(monkeypatch):
    got_dates: list[date] = []
    monkeypatch.setattr("statflow.flows.daily.run_daily_ingest", lambda d: got_dates.append(d))
    monkeypatch.setattr("statflow.flows.daily.build_silver", lambda: None)
    monkeypatch.setattr("statflow.flows.daily.build_features", lambda: None)
    monkeypatch.setattr("statflow.flows.daily.predict_for_date", lambda d: None)
    monkeypatch.setattr("statflow.flows.daily.build_prediction_outcomes", lambda: None)

    daily_pipeline()

    assert len(got_dates) == INGEST_BACKFILL_DAYS
    # Last ingest is always today; earlier entries are the backfill days.
    assert got_dates[-1] == date.today()
