"""Feature importance extraction for tree models.

XGBoost has native `.feature_importances_` (gain-based by default). We save
importance to MLflow as both a text log and a CSV artifact — the CSV is the
one to reference from the walkthrough / interview.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def xgb_feature_importance(model, feature_names: list[str]) -> pd.DataFrame:
    """Return a sorted (feature, importance) DataFrame from an XGBoost model.

    Accepts either a plain XGBClassifier/XGBRegressor or a sklearn-wrapped
    variant (e.g., CalibratedClassifierCV) — importance always comes from
    the underlying XGBoost booster.
    """
    from statflow.models.explanations import _extract_xgb_booster

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        booster = _extract_xgb_booster(model)
        # Gain-based importance, aligned to our feature_names order
        score = booster.get_score(importance_type="gain")
        importances = [score.get(f, 0.0) for f in feature_names]

    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def write_importance_csv(df: pd.DataFrame, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path
