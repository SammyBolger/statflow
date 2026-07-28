"""Score today's games using the latest trained models.

Reads the latest run per (experiment, model_family) from MLflow, applies each
model to the feature rows for a target date, and writes a single predictions
parquet under `data/gold/predictions/date=YYYY-MM-DD/`.

Prediction schema:
    game_pk                     BIGINT
    game_date                   DATE
    predicted_home_win_prob     DOUBLE
    predicted_total_runs        DOUBLE
    winner_model_run_id         VARCHAR   -- MLflow run id for provenance
    runs_model_run_id           VARCHAR
    predicted_at                TIMESTAMP
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import mlflow
import pandas as pd

from statflow.config import GOLD_DIR
from statflow.models.data import FEATURE_COLS, load_features
from statflow.models.train import RUNS_EXPERIMENT, WINNER_EXPERIMENT, _init_mlflow


def _latest_run_id(experiment_name: str, model_family_tag: str) -> str:
    """Return the run id predict.py should serve for a given experiment.

    Preference order:
      1. Latest run tagged `promoted=true` (the promotion-gate winner).
      2. Latest run for the model_family, if no promoted tag exists yet
         (backfill for pre-promotion-gate runs).
    """
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        raise RuntimeError(f"MLflow experiment not found: {experiment_name}")

    promoted = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=(f"tags.model_family = '{model_family_tag}' AND tags.promoted = 'true'"),
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not promoted.empty:
        return str(promoted.iloc[0]["run_id"])

    any_run = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f"tags.model_family = '{model_family_tag}'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if any_run.empty:
        raise RuntimeError(
            f"No MLflow runs found for experiment={experiment_name} "
            f"model_family={model_family_tag}. Train first with "
            f"`python -m statflow.models train`."
        )
    return str(any_run.iloc[0]["run_id"])


def _load_model_any_flavor(run_id: str) -> object:
    """Load a run's model artifact regardless of whether it was logged via
    mlflow.xgboost (older runs) or mlflow.sklearn (calibrated post-gate)."""
    uri = f"runs:/{run_id}/model"
    try:
        return mlflow.sklearn.load_model(uri)
    except Exception:
        return mlflow.xgboost.load_model(uri)


def load_latest_winner_model() -> tuple[object, str]:
    """Load the model predict.py should serve for the winner target."""
    _init_mlflow(WINNER_EXPERIMENT)
    run_id = _latest_run_id(WINNER_EXPERIMENT, "xgboost")
    return _load_model_any_flavor(run_id), run_id


def load_latest_runs_model() -> tuple[object, str]:
    """Load the model predict.py should serve for the total-runs target."""
    _init_mlflow(RUNS_EXPERIMENT)
    run_id = _latest_run_id(RUNS_EXPERIMENT, "xgboost")
    return _load_model_any_flavor(run_id), run_id


def predict_for_date(
    target_date: date,
    gold_dir: Path = GOLD_DIR,
    now: datetime | None = None,
) -> Path | None:
    """Predict every game on `target_date` and write results to gold.

    Returns the parquet path, or None if there are no games on that date.
    """
    features = load_features(gold_dir)
    todays = features[features["game_date"] == pd.Timestamp(target_date)]
    if todays.empty:
        return None

    winner_model, winner_run_id = load_latest_winner_model()
    runs_model, runs_run_id = load_latest_runs_model()

    X = todays[FEATURE_COLS]
    win_prob = winner_model.predict_proba(X)[:, 1]
    total_runs = runs_model.predict(X)

    out_df = pd.DataFrame(
        {
            "game_pk": todays["game_pk"].to_numpy(),
            "game_date": target_date.isoformat(),
            "predicted_home_win_prob": win_prob,
            "predicted_total_runs": total_runs,
            "winner_model_run_id": winner_run_id,
            "runs_model_run_id": runs_run_id,
            "predicted_at": now or datetime.now(UTC),
        }
    )

    out_dir = gold_dir / "predictions" / f"date={target_date.isoformat()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "predictions.parquet"
    out_df.to_parquet(path, index=False)
    return path
