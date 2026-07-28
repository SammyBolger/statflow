"""FastAPI service exposing StatFlow predictions + monitoring as JSON.

Reads directly from the gold parquet files the daily flow writes — no
model code runs at request time, which means the API is fast (< 100 ms)
and stateless. Same data source as the Streamlit dashboard.

Run locally:
    uv run uvicorn statflow.api.app:app --reload --port 8000

Endpoints:
    GET /health
    GET /predictions/today
    GET /predictions/{date}                (YYYY-MM-DD)
    GET /performance/rolling?days=30
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from statflow import config
from statflow.dashboard.data import (
    load_prediction_outcomes,
    load_todays_games,
    rolling_performance,
)

app = FastAPI(
    title="StatFlow API",
    description=(
        "MLB game prediction service — reads gold parquet directly, "
        "no model inference at request time."
    ),
    version="0.1.0",
)


def _rows_to_json(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame to JSON-safe records (Timestamps → isoformat, NaN → None)."""
    if df.empty:
        return []
    return df.astype(object).where(pd.notna(df), None).to_dict(orient="records")


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe. Returns 200 as long as the process is up."""
    return {"status": "ok"}


@app.get("/predictions/today")
def predictions_today() -> dict[str, Any]:
    """Today's games with predictions joined from gold.

    Response shape:
        {
          "date": "2026-07-28",
          "games": [
            {
              "game_pk": 823518,
              "home_team_name": "New York Yankees",
              "away_team_name": "Pittsburgh Pirates",
              "predicted_home_win_prob": 0.63,
              "predicted_total_runs": 8.9,
              ...
            }
          ]
        }
    """
    today = date.today()
    games = load_todays_games(today, silver_dir=config.SILVER_DIR, gold_dir=config.GOLD_DIR)
    return {"date": today.isoformat(), "games": _rows_to_json(games)}


@app.get("/predictions/{target_date}")
def predictions_for_date(target_date: date) -> dict[str, Any]:
    """Games + predictions for a specific date (YYYY-MM-DD)."""
    games = load_todays_games(target_date, silver_dir=config.SILVER_DIR, gold_dir=config.GOLD_DIR)
    if games.empty:
        raise HTTPException(status_code=404, detail=f"no games found for {target_date}")
    return {"date": target_date.isoformat(), "games": _rows_to_json(games)}


@app.get("/performance/rolling")
def performance_rolling(
    days: int = Query(default=30, ge=1, le=365),
) -> dict[str, Any]:
    """Rolling model performance over the last `days` days of completed games."""
    outcomes = load_prediction_outcomes(gold_dir=config.GOLD_DIR)
    if outcomes.empty:
        return {"window_days": days, "metrics": None}
    metrics = rolling_performance(outcomes, window_days=days)
    return {"window_days": days, "metrics": metrics or None}
