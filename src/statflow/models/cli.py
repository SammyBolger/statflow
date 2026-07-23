"""CLI: train all four models and print a comparison.

Usage:
    uv run python -m statflow.models train
    uv run python -m statflow.models train --seasons 2024,2025 --val-seasons 2026

Everything is logged to MLflow (sqlite:mlflow.db + mlartifacts/). Feature
importance for each XGBoost model is written next to the MLflow artifacts.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import mlflow
import pandas as pd

from statflow.models.baselines import HomeAlwaysWinsBaseline, MeanRunsBaseline
from statflow.models.data import (
    FEATURE_COLS,
    load_features,
    make_classification_xy,
    make_regression_xy,
    split_by_season,
)
from statflow.models.importance import write_importance_csv, xgb_feature_importance
from statflow.models.metrics import classification_metrics, regression_metrics
from statflow.models.train import (
    train_logistic_regression,
    train_ridge_regression,
    train_xgb_classifier,
    train_xgb_regressor,
)


def _parse_seasons(s: str) -> list[int]:
    return [int(x) for x in s.split(",")]


def _baseline_classification(split) -> dict:
    """Score HomeAlwaysWinsBaseline on all splits — separate from MLflow."""
    X_tr, y_tr = make_classification_xy(split.train)
    b = HomeAlwaysWinsBaseline().fit(X_tr, y_tr)
    out = {}
    for name, part in [("train", split.train), ("val", split.val), ("test", split.test)]:
        if len(part) == 0:
            continue
        X, y = make_classification_xy(part)
        out[name] = classification_metrics(y, b.predict_proba(X))
    return out


def _baseline_regression(split) -> dict:
    X_tr, y_tr = make_regression_xy(split.train)
    b = MeanRunsBaseline().fit(X_tr, y_tr)
    out = {}
    for name, part in [("train", split.train), ("val", split.val), ("test", split.test)]:
        if len(part) == 0:
            continue
        X, y = make_regression_xy(part)
        out[name] = regression_metrics(y, b.predict(X))
    return out


def _print_comparison(title: str, results: dict[str, dict]) -> None:
    """Print a per-split comparison table for one target across several models."""
    print(f"\n=== {title} ===")
    for split_name in ("train", "val", "test"):
        rows = []
        for model_name, per_split in results.items():
            m = per_split.get(split_name)
            if m is None:
                continue
            rows.append({"model": model_name, **m.as_dict()})
        if rows:
            print(f"\n[{split_name}]")
            print(pd.DataFrame(rows).to_string(index=False))


def _log_importance(model, tag: str) -> None:
    """Attach XGBoost feature importance to the active MLflow run as a CSV artifact."""
    df = xgb_feature_importance(model, FEATURE_COLS)
    with tempfile.TemporaryDirectory() as tmp:
        path = write_importance_csv(df, Path(tmp) / f"{tag}_importance.csv")
        mlflow.log_artifact(str(path))


def _run_training(train_seasons, val_seasons, test_seasons) -> None:
    print(f"Loading features. train={train_seasons} val={val_seasons} test={test_seasons}")
    df = load_features()
    split = split_by_season(df, train_seasons, val_seasons, test_seasons)
    print(f"n_train={len(split.train)} n_val={len(split.val)} n_test={len(split.test)}")

    # Winner classifier
    winner_results = {"baseline": _baseline_classification(split)}
    _, winner_results["logistic_regression"] = train_logistic_regression(split)
    xgb_winner, winner_results["xgboost"] = train_xgb_classifier(split)
    with mlflow.start_run(run_name="xgboost_winner_importance", nested=False):
        _log_importance(xgb_winner, "winner")

    # Runs regressor
    runs_results = {"baseline": _baseline_regression(split)}
    _, runs_results["ridge"] = train_ridge_regression(split)
    xgb_runs, runs_results["xgboost"] = train_xgb_regressor(split)
    with mlflow.start_run(run_name="xgboost_runs_importance", nested=False):
        _log_importance(xgb_runs, "runs")

    _print_comparison("Winner classification", winner_results)
    _print_comparison("Total runs regression", runs_results)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m statflow.models",
        description="Train baseline + LR/Ridge + XGBoost for both prediction targets.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="Train all four models and print comparison.")
    train.add_argument("--seasons", type=_parse_seasons, default=[2024])
    train.add_argument("--val-seasons", type=_parse_seasons, default=[2025])
    train.add_argument("--test-seasons", type=_parse_seasons, default=[2026])
    args = parser.parse_args(argv)

    if args.command == "train":
        _run_training(args.seasons, args.val_seasons, args.test_seasons)
