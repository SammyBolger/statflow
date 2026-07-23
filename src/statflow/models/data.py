"""Load gold features and produce time-based splits for modeling.

Time-based splits are non-negotiable — a random shuffle would leak future
outcomes into training. Splits happen by season: earlier seasons train,
later seasons validate/test.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from statflow.config import GOLD_DIR

# Explicit feature list. Identifier columns (game_pk, team ids, venue_id) and
# season/game_date are deliberately excluded — they'd let the model memorize
# rather than generalize.
FEATURE_COLS: list[str] = [
    "home_runs_scored_l10",
    "home_runs_allowed_l10",
    "home_win_pct_l10",
    "home_days_rest",
    "away_runs_scored_l10",
    "away_runs_allowed_l10",
    "away_win_pct_l10",
    "away_days_rest",
    "home_sp_era_l5",
    "home_sp_k_per_9_l5",
    "home_sp_days_rest",
    "away_sp_era_l5",
    "away_sp_k_per_9_l5",
    "away_sp_days_rest",
    "venue_park_factor_runs",
    "is_day_game",
    "month",
]

TARGET_CLASSIFICATION = "target_home_win"
TARGET_REGRESSION = "target_total_runs"


@dataclass
class Split:
    """A train / val / test split of gold features, already filtered to Final games."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def load_features(gold_dir: Path = GOLD_DIR) -> pd.DataFrame:
    """Load the raw features parquet."""
    return pd.read_parquet(gold_dir / "features" / "features.parquet")


def split_by_season(
    df: pd.DataFrame,
    train_seasons: list[int],
    val_seasons: list[int],
    test_seasons: list[int],
) -> Split:
    """Filter to Final games and split by season.

    We filter to Final games (target present) here because that's the only
    rowset the model can train/evaluate on. Scheduled games are still useful
    downstream for prediction — this function isn't for that path.
    """
    finals = df[df[TARGET_CLASSIFICATION].notna()].copy()
    return Split(
        train=finals[finals["season"].isin(train_seasons)].reset_index(drop=True),
        val=finals[finals["season"].isin(val_seasons)].reset_index(drop=True),
        test=finals[finals["season"].isin(test_seasons)].reset_index(drop=True),
    )


def make_classification_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) for the winner-prediction task."""
    return df[FEATURE_COLS], df[TARGET_CLASSIFICATION].astype(int)


def make_regression_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) for the total-runs task."""
    return df[FEATURE_COLS], df[TARGET_REGRESSION].astype(int)
