"""Join stored predictions with actual game results into a monitoring table.

Every daily flow rebuilds this from scratch:
    1. Read all prediction parquets across all date partitions.
    2. Read silver.games for the actual results.
    3. Join on game_pk; keep only games that have completed (status='Final').
    4. Compute per-row error metrics.
    5. Write to data/gold/prediction_outcomes/prediction_outcomes.parquet.

The `winner_model_run_id` from predictions preserves *which model version*
made each prediction — crucial for honest monitoring. When we retrain
tomorrow, we don't want to grade the new model on yesterday's predictions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from statflow.config import GOLD_DIR, SILVER_DIR


def _read_all_predictions(gold_dir: Path) -> pd.DataFrame:
    """Load every predictions parquet under gold/predictions/."""
    paths = sorted(gold_dir.glob("predictions/date=*/predictions.parquet"))
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def _row_log_loss(prob: pd.Series, actual: pd.Series) -> pd.Series:
    """Per-row binary log loss. Clip probabilities to avoid log(0)."""
    p = prob.clip(1e-15, 1 - 1e-15)
    a = actual.astype(int)
    return -(a * np.log(p) + (1 - a) * np.log(1 - p))


def build_prediction_outcomes(
    silver_dir: Path = SILVER_DIR,
    gold_dir: Path = GOLD_DIR,
) -> Path | None:
    """Rebuild the prediction_outcomes monitoring table from scratch.

    Returns the parquet path, or None if no predictions have been made yet.
    """
    predictions = _read_all_predictions(gold_dir)
    if predictions.empty:
        return None

    games = pd.read_parquet(silver_dir / "games" / "games.parquet")
    finals = games.loc[
        games["status"] == "Final",
        ["game_pk", "home_win", "total_runs"],
    ].rename(columns={"home_win": "actual_home_win", "total_runs": "actual_total_runs"})

    df = predictions.merge(finals, on="game_pk", how="inner")
    if df.empty:
        # Predictions exist but none of those games have completed yet.
        return None

    df["actual_home_win"] = df["actual_home_win"].astype(int)
    df["winner_correct"] = (df["predicted_home_win_prob"] >= 0.5).astype(int) == df[
        "actual_home_win"
    ]
    df["winner_log_loss"] = _row_log_loss(df["predicted_home_win_prob"], df["actual_home_win"])
    df["winner_brier"] = (df["predicted_home_win_prob"] - df["actual_home_win"]) ** 2
    df["runs_abs_error"] = (df["predicted_total_runs"] - df["actual_total_runs"]).abs()
    df["runs_squared_error"] = (df["predicted_total_runs"] - df["actual_total_runs"]) ** 2

    # Attach market line + market-implied probability if silver.odds exists.
    # NULL when there's no matching odds row (odds ingest disabled, or the
    # game happened before we started pulling lines).
    odds_path = silver_dir / "odds" / "odds.parquet"
    if odds_path.exists():
        odds = pd.read_parquet(odds_path)[
            ["game_pk", "market_home_win_prob", "market_total_line"]
        ].dropna(subset=["game_pk"])
        odds["game_pk"] = odds["game_pk"].astype(df["game_pk"].dtype, errors="ignore")
        df = df.merge(odds, on="game_pk", how="left")
        df["market_winner_brier"] = np.where(
            df["market_home_win_prob"].notna(),
            (df["market_home_win_prob"] - df["actual_home_win"]) ** 2,
            np.nan,
        )
        df["market_runs_abs_error"] = (df["market_total_line"] - df["actual_total_runs"]).abs()

    # If a game was re-predicted by a newer model run, prefer the freshest.
    df = df.sort_values("predicted_at").drop_duplicates(
        subset=["game_pk", "winner_model_run_id"], keep="last"
    )

    out_dir = gold_dir / "prediction_outcomes"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "prediction_outcomes.parquet"
    df.to_parquet(path, index=False)
    return path
