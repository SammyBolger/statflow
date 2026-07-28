"""Per-prediction feature contribution helpers.

Uses XGBoost's native `pred_contribs=True` — for tree models this returns
exact Shapley values without needing the (heavy) `shap` library as a
dependency. Cheap, deterministic, works on the same trained artifact
predict.py loads at inference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def top_contributions(
    model,
    X: pd.DataFrame,
    feature_names: list[str],
    top_n: int = 5,
) -> list[list[tuple[str, float]]]:
    """For each row in X, return the top-N (feature, contribution) tuples by absolute magnitude.

    Contributions are in log-odds space (binary classifier) or in the target's
    native units (regressor). Positive = pushed the prediction up; negative =
    pushed it down.
    """
    # XGBoost's native SHAP-equivalent: shape (n_samples, n_features + 1).
    # Last column is the base/bias term — drop it.
    booster = getattr(model, "get_booster", lambda: model)()
    import xgboost as xgb

    dmatrix = xgb.DMatrix(X, feature_names=feature_names)
    contribs = booster.predict(dmatrix, pred_contribs=True)  # (n, n_feat + 1)
    contribs = contribs[:, :-1]  # drop bias

    results: list[list[tuple[str, float]]] = []
    for row in contribs:
        ranked = sorted(
            zip(feature_names, row, strict=True),
            key=lambda pair: abs(pair[1]),
            reverse=True,
        )
        results.append(ranked[:top_n])
    return results


def contributions_frame(
    game_pks: list[int],
    contributions: list[list[tuple[str, float]]],
) -> pd.DataFrame:
    """Flatten a list-of-lists of contributions into a tidy DataFrame."""
    rows = []
    for game_pk, contrib_list in zip(game_pks, contributions, strict=True):
        for rank, (feature, value) in enumerate(contrib_list, start=1):
            rows.append(
                {
                    "game_pk": game_pk,
                    "rank": rank,
                    "feature": feature,
                    "contribution": float(value),
                    "direction": "▲ home" if value > 0 else "▼ away",
                    "abs_contribution": float(abs(value)),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=["game_pk", "rank", "feature", "contribution", "direction", "abs_contribution"]
        )
    return pd.DataFrame(rows)


def format_contribution_line(feature: str, contribution: float) -> str:
    """One-line human-readable summary for a single feature contribution."""
    arrow = "↑" if contribution > 0 else "↓"
    magnitude = abs(contribution)
    return (
        f"{arrow} `{feature}` {magnitude:+.3f}"
        if contribution < 0
        else f"↑ `{feature}` +{magnitude:.3f}"
    )


# Silence unused import (keeps type-checker happy without importing numpy above).
_ = np
