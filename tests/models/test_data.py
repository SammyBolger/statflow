from pathlib import Path

import pandas as pd
import pytest

from statflow.models.data import (
    FEATURE_COLS,
    load_features,
    make_classification_xy,
    make_regression_xy,
    split_by_season,
)


def _features_row(game_pk: int, season: int, home_win: bool | None, total_runs: int | None) -> dict:
    """Build a full feature row with sane values in every column."""
    row = {
        "game_pk": game_pk,
        "game_date": pd.Timestamp(f"{season}-06-01"),
        "season": season,
        "home_team_id": 100,
        "away_team_id": 200,
        "venue_id": 3313,
        "target_home_win": home_win,
        "target_total_runs": total_runs,
        "season_type": "R",
    }
    for col in FEATURE_COLS:
        row[col] = 0.5 if col.endswith("_pct_l10") else 5.0
    return row


@pytest.fixture
def features_parquet(tmp_path: Path) -> Path:
    rows = [
        _features_row(1, 2024, True, 8),
        _features_row(2, 2024, False, 6),
        _features_row(3, 2025, True, 10),
        _features_row(4, 2026, None, None),  # scheduled — no target
    ]
    df = pd.DataFrame(rows)
    out = tmp_path / "gold" / "features" / "features.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return tmp_path / "gold"


def test_load_features_reads_parquet(features_parquet):
    df = load_features(features_parquet)
    assert len(df) == 4


def test_split_by_season_filters_finals_and_splits(features_parquet):
    df = load_features(features_parquet)
    split = split_by_season(df, [2024], [2025], [2026])

    assert len(split.train) == 2  # 2024 has 2 Final games
    assert len(split.val) == 1
    # 2026 has 1 row but its target is NULL (scheduled), so filtered out
    assert len(split.test) == 0


def test_make_classification_xy_types(features_parquet):
    df = load_features(features_parquet)
    split = split_by_season(df, [2024], [2025], [2026])
    X, y = make_classification_xy(split.train)

    assert list(X.columns) == FEATURE_COLS
    assert len(X) == 2
    assert set(y.unique()).issubset({0, 1})


def test_make_regression_xy_types(features_parquet):
    df = load_features(features_parquet)
    split = split_by_season(df, [2024], [2025], [2026])
    X, y = make_regression_xy(split.train)

    assert list(X.columns) == FEATURE_COLS
    assert (y == pd.Series([8, 6])).all()
