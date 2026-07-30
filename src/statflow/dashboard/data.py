"""Pure data-loading helpers for the Streamlit dashboard.

No Streamlit imports here — the dashboard app wraps these with
`@st.cache_data` for reruns. Keeping this module pure means tests can
exercise it directly without simulating a Streamlit session.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from statflow.config import GOLD_DIR, SILVER_DIR


def load_todays_games(
    target_date: date,
    silver_dir: Path = SILVER_DIR,
    gold_dir: Path = GOLD_DIR,
) -> pd.DataFrame:
    """Return the games on `target_date`, LEFT-joined with predictions if any.

    Columns include the silver.games identifiers + team/pitcher names +
    (nullable) predicted_home_win_prob and predicted_total_runs.
    """
    games_path = silver_dir / "games" / "games.parquet"
    if not games_path.exists():
        return pd.DataFrame()
    games = pd.read_parquet(games_path)
    todays = games[games["game_date"] == pd.Timestamp(target_date)].copy()
    if todays.empty:
        return todays

    pred_path = gold_dir / "predictions" / f"date={target_date.isoformat()}" / "predictions.parquet"
    if pred_path.exists():
        preds = pd.read_parquet(pred_path)[
            ["game_pk", "predicted_home_win_prob", "predicted_total_runs"]
        ]
        todays = todays.merge(preds, on="game_pk", how="left")
    else:
        todays["predicted_home_win_prob"] = pd.NA
        todays["predicted_total_runs"] = pd.NA

    return todays.sort_values("game_datetime_utc").reset_index(drop=True)


def load_prediction_outcomes(gold_dir: Path = GOLD_DIR) -> pd.DataFrame:
    """Load the prediction_outcomes monitoring table (empty df if not built yet)."""
    path = gold_dir / "prediction_outcomes" / "prediction_outcomes.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def load_features_for_games(
    game_pks: list[int],
    gold_dir: Path = GOLD_DIR,
) -> pd.DataFrame:
    """Return the gold feature rows for a list of game_pks, indexed by game_pk.

    Used by the dashboard's per-game explanation panel — we need the exact
    feature values the model saw to compute SHAP contributions.
    """
    path = gold_dir / "features" / "features.parquet"
    if not path.exists() or not game_pks:
        return pd.DataFrame()
    df = pd.read_parquet(path)
    return df[df["game_pk"].isin(game_pks)].set_index("game_pk", drop=False)


def load_historical_predictions(
    silver_dir: Path = SILVER_DIR,
    gold_dir: Path = GOLD_DIR,
) -> pd.DataFrame:
    """Join prediction_outcomes with silver.games for a browsable history.

    Adds team names + actual scores so the historical explorer tab can
    filter by team and display readable rows without further joins.
    """
    outcomes_path = gold_dir / "prediction_outcomes" / "prediction_outcomes.parquet"
    games_path = silver_dir / "games" / "games.parquet"
    if not outcomes_path.exists() or not games_path.exists():
        return pd.DataFrame()

    outcomes = pd.read_parquet(outcomes_path)
    games = pd.read_parquet(games_path)[
        [
            "game_pk",
            "home_team_name",
            "away_team_name",
            "home_score",
            "away_score",
            "status",
        ]
    ]
    merged = outcomes.merge(games, on="game_pk", how="left")
    merged["game_date"] = pd.to_datetime(merged["game_date"]).dt.date
    return merged.sort_values("game_date", ascending=False).reset_index(drop=True)


def rolling_performance(outcomes: pd.DataFrame, window_days: int = 30) -> dict:
    """Recent per-model summary metrics over the last `window_days`.

    Returns a plain dict — the dashboard renders each field as a metric card.
    """
    if outcomes.empty:
        return {}
    cutoff = pd.Timestamp(outcomes["game_date"].max()) - pd.Timedelta(days=window_days)
    recent = outcomes[pd.to_datetime(outcomes["game_date"]) > cutoff]
    if recent.empty:
        return {}
    return {
        "n_games": int(len(recent)),
        "accuracy": float(recent["winner_correct"].mean()),
        "log_loss": float(recent["winner_log_loss"].mean()),
        "brier": float(recent["winner_brier"].mean()),
        "mae": float(recent["runs_abs_error"].mean()),
        "rmse": float((recent["runs_squared_error"].mean()) ** 0.5),
    }


def baseline_comparison(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Model vs naive-baseline metrics computed on the same completed games.

    Baselines are recomputed on the actuals themselves — "if we had always
    predicted the mean home-win rate / mean total runs, how would we have
    done?" — so this always reflects the current data, not a snapshot from
    training.
    """
    if outcomes.empty:
        return pd.DataFrame()

    import numpy as np

    actuals = outcomes["actual_home_win"].astype(int).to_numpy()
    actual_runs = outcomes["actual_total_runs"].to_numpy(dtype=float)

    # Naive baselines: predict the mean every time.
    p = float(outcomes["actual_home_win"].mean())
    mean_runs = float(outcomes["actual_total_runs"].mean())
    p_clip = min(max(p, 1e-15), 1 - 1e-15)
    baseline_accuracy = float((int(p_clip >= 0.5) == actuals).mean())
    baseline_log_loss = float(
        -(actuals * np.log(p_clip) + (1 - actuals) * np.log(1 - p_clip)).mean()
    )
    baseline_brier = float(((p - actuals) ** 2).mean())
    baseline_mae = float(np.abs(mean_runs - actual_runs).mean())
    baseline_rmse = float(np.sqrt(((mean_runs - actual_runs) ** 2).mean()))

    model_accuracy = float(outcomes["winner_correct"].mean())
    model_log_loss = float(outcomes["winner_log_loss"].mean())
    model_brier = float(outcomes["winner_brier"].mean())
    model_mae = float(outcomes["runs_abs_error"].mean())
    model_rmse = float(np.sqrt(outcomes["runs_squared_error"].mean()))

    return pd.DataFrame(
        [
            {"metric": "Accuracy", "model": model_accuracy, "baseline": baseline_accuracy},
            {"metric": "Log loss", "model": model_log_loss, "baseline": baseline_log_loss},
            {"metric": "Brier", "model": model_brier, "baseline": baseline_brier},
            {"metric": "MAE (runs)", "model": model_mae, "baseline": baseline_mae},
            {"metric": "RMSE (runs)", "model": model_rmse, "baseline": baseline_rmse},
        ]
    )


def market_comparison(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Model vs market metrics on games that have BOTH predictions and market data.

    Empty DataFrame if `outcomes` has no market columns yet (odds ingest not
    enabled) or if the intersection of games with a model prediction and a
    market line is empty.
    """
    if outcomes.empty or "market_home_win_prob" not in outcomes.columns:
        return pd.DataFrame()

    both = outcomes.dropna(subset=["market_home_win_prob", "market_total_line"])
    if both.empty:
        return pd.DataFrame()

    import numpy as np

    actual_wins = both["actual_home_win"].astype(int).to_numpy()
    m_prob = both["market_home_win_prob"].to_numpy(dtype=float).clip(1e-15, 1 - 1e-15)
    p_prob = both["predicted_home_win_prob"].to_numpy(dtype=float).clip(1e-15, 1 - 1e-15)
    market_log_loss = float(
        -(actual_wins * np.log(m_prob) + (1 - actual_wins) * np.log(1 - m_prob)).mean()
    )
    model_log_loss = float(
        -(actual_wins * np.log(p_prob) + (1 - actual_wins) * np.log(1 - p_prob)).mean()
    )

    return pd.DataFrame(
        [
            {
                "metric": "Log loss",
                "model": model_log_loss,
                "market": market_log_loss,
            },
            {
                "metric": "Brier",
                "model": float(both["winner_brier"].mean()),
                "market": float(both["market_winner_brier"].mean()),
            },
            {
                "metric": "MAE (runs)",
                "model": float(both["runs_abs_error"].mean()),
                "market": float(both["market_runs_abs_error"].mean()),
            },
        ]
    )


def runs_scatter_data(outcomes: pd.DataFrame) -> pd.DataFrame:
    """One row per completed game: predicted vs actual total runs."""
    if outcomes.empty:
        return pd.DataFrame(columns=["game_pk", "predicted_total_runs", "actual_total_runs"])
    return outcomes[["game_pk", "predicted_total_runs", "actual_total_runs"]].copy()


def confidence_tier_breakdown(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Accuracy split by how confident the model was on each pick.

    A calibrated + informative model should be MORE accurate on games it was
    more confident about. Flat accuracy across tiers means the "confidence"
    doesn't carry real signal.
    """
    if outcomes.empty:
        return pd.DataFrame(columns=["tier", "n", "accuracy"])

    df = outcomes[["predicted_home_win_prob", "winner_correct"]].copy()
    # Distance from 0.5 — always in [0.5, 1.0], regardless of which team is favored.
    df["confidence"] = df["predicted_home_win_prob"].apply(lambda p: max(p, 1 - p))

    bins = [0.5, 0.55, 0.60, 0.65, 0.70, 1.0001]
    labels = ["50–55%", "55–60%", "60–65%", "65–70%", "70%+"]
    df["tier"] = pd.cut(df["confidence"], bins=bins, labels=labels, include_lowest=True)

    grouped = df.groupby("tier", observed=True).agg(
        n=("winner_correct", "count"),
        accuracy=("winner_correct", "mean"),
    )
    return grouped.reset_index()


def daily_metrics_trend(outcomes: pd.DataFrame, rolling_days: int = 7) -> pd.DataFrame:
    """Per-day aggregate metrics + a rolling average for the trend chart."""
    if outcomes.empty:
        return pd.DataFrame(
            columns=[
                "game_date",
                "n",
                "accuracy",
                "log_loss",
                "mae",
                "accuracy_rolling",
                "log_loss_rolling",
            ]
        )

    df = outcomes.copy()
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    grouped = (
        df.groupby("game_date")
        .agg(
            n=("winner_correct", "count"),
            accuracy=("winner_correct", "mean"),
            log_loss=("winner_log_loss", "mean"),
            mae=("runs_abs_error", "mean"),
        )
        .reset_index()
        .sort_values("game_date")
    )
    grouped["accuracy_rolling"] = grouped["accuracy"].rolling(rolling_days, min_periods=1).mean()
    grouped["log_loss_rolling"] = grouped["log_loss"].rolling(rolling_days, min_periods=1).mean()
    return grouped


def calibration_bins(outcomes: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """Bin predicted probabilities and compute the actual home-win rate per bin.

    A well-calibrated model has mean_actual ≈ mean_predicted in every bin —
    plotted vs the y=x diagonal, points lie on the line.
    """
    if outcomes.empty:
        return pd.DataFrame(columns=["bin", "n", "mean_predicted", "mean_actual"])
    df = outcomes[["predicted_home_win_prob", "actual_home_win"]].copy()
    df["bin"] = pd.cut(df["predicted_home_win_prob"], bins=n_bins, labels=False)
    grouped = df.groupby("bin", observed=True).agg(
        n=("actual_home_win", "count"),
        mean_predicted=("predicted_home_win_prob", "mean"),
        mean_actual=("actual_home_win", "mean"),
    )
    return grouped.reset_index()
