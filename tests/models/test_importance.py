from pathlib import Path

import pandas as pd

from statflow.models.importance import write_importance_csv, xgb_feature_importance


class FakeXGB:
    def __init__(self, importances):
        self.feature_importances_ = importances


def test_xgb_feature_importance_returns_sorted_df():
    features = ["a", "b", "c"]
    model = FakeXGB([0.3, 0.5, 0.2])

    df = xgb_feature_importance(model, features)

    assert list(df.columns) == ["feature", "importance"]
    assert list(df["feature"]) == ["b", "a", "c"]
    assert list(df["importance"]) == [0.5, 0.3, 0.2]


def test_write_importance_csv_roundtrips(tmp_path: Path):
    df = pd.DataFrame({"feature": ["a", "b"], "importance": [0.5, 0.3]})
    out = write_importance_csv(df, tmp_path / "nested" / "imp.csv")

    assert out.exists()
    loaded = pd.read_csv(out)
    assert list(loaded["feature"]) == ["a", "b"]
