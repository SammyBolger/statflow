"""Tests for the FastAPI service.

Uses FastAPI's TestClient — no real HTTP, no uvicorn — plus dependency-
overrides so we point the endpoints at fixture parquet files in tmp_path.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from statflow.api.app import app


def _write_games(silver_dir: Path, rows: list[dict]) -> Path:
    out = silver_dir / "games" / "games.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    return out


def _write_predictions(gold_dir: Path, target: str, rows: list[dict]) -> Path:
    out = gold_dir / "predictions" / f"date={target}" / "predictions.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    return out


def _write_outcomes(gold_dir: Path, rows: list[dict]) -> Path:
    out = gold_dir / "prediction_outcomes" / "prediction_outcomes.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    return out


@pytest.fixture
def client(tmp_path: Path, monkeypatch):
    """A TestClient with SILVER_DIR / GOLD_DIR redirected to tmp_path."""
    monkeypatch.setattr("statflow.config.SILVER_DIR", tmp_path / "silver")
    monkeypatch.setattr("statflow.config.GOLD_DIR", tmp_path / "gold")
    return TestClient(app), tmp_path


def test_health_returns_ok(client):
    api, _ = client
    resp = api.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_predictions_for_date_returns_games(client):
    api, tmp = client
    _write_games(
        tmp / "silver",
        [
            {
                "game_pk": 111,
                "game_date": pd.Timestamp("2026-07-27"),
                "game_datetime_utc": pd.Timestamp("2026-07-27T23:00:00Z"),
                "status": "Final",
                "home_team_name": "Yankees",
                "away_team_name": "Pirates",
                "home_team_id": 147,
                "away_team_id": 134,
                "home_score": 5,
                "away_score": 3,
                "home_win": True,
                "home_probable_pitcher_name": "Cole",
                "away_probable_pitcher_name": "Keller",
            }
        ],
    )
    _write_predictions(
        tmp / "gold",
        "2026-07-27",
        [{"game_pk": 111, "predicted_home_win_prob": 0.65, "predicted_total_runs": 8.5}],
    )

    resp = api.get("/predictions/2026-07-27")
    assert resp.status_code == 200
    body = resp.json()
    assert body["date"] == "2026-07-27"
    assert len(body["games"]) == 1
    game = body["games"][0]
    assert game["game_pk"] == 111
    assert game["home_team_name"] == "Yankees"
    assert game["predicted_home_win_prob"] == pytest.approx(0.65)


def test_predictions_for_date_returns_404_when_no_games(client):
    api, _ = client
    resp = api.get("/predictions/2099-12-25")
    assert resp.status_code == 404


def test_predictions_today_returns_empty_gracefully(client):
    api, _ = client
    resp = api.get("/predictions/today")
    assert resp.status_code == 200
    body = resp.json()
    assert body["date"] == date.today().isoformat()
    assert body["games"] == []


def test_performance_rolling_returns_null_metrics_when_no_outcomes(client):
    api, _ = client
    resp = api.get("/performance/rolling?days=30")
    assert resp.status_code == 200
    body = resp.json()
    assert body["window_days"] == 30
    assert body["metrics"] is None


def test_performance_rolling_returns_metrics_when_outcomes_exist(client):
    api, tmp = client
    _write_outcomes(
        tmp / "gold",
        [
            {
                "game_date": pd.Timestamp("2026-07-27"),
                "winner_correct": True,
                "winner_log_loss": 0.5,
                "winner_brier": 0.15,
                "runs_abs_error": 2.0,
                "runs_squared_error": 4.0,
            }
        ],
    )
    resp = api.get("/performance/rolling?days=30")
    body = resp.json()
    assert body["metrics"]["n_games"] == 1
    assert body["metrics"]["accuracy"] == 1.0


def test_performance_rolling_validates_days_range(client):
    api, _ = client
    # days=0 is below the ge=1 constraint → 422
    assert api.get("/performance/rolling?days=0").status_code == 422
    # days=1000 is above le=365 → 422
    assert api.get("/performance/rolling?days=1000").status_code == 422
