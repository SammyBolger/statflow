"""Train the winner classifier: LR baseline and XGBoost.

Both training functions:
  1. Build the model.
  2. Fit on training data.
  3. Evaluate on train / val / test with the same metrics.
  4. Log params, metrics, and model artifact to MLflow.

`fit_and_evaluate_classifier` is the pure-Python core (no MLflow) so it's
straightforward to test — MLflow is a side effect on the outside.
"""

from __future__ import annotations

from typing import Any

import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from statflow.config import PROJECT_ROOT
from statflow.models.data import Split, make_classification_xy
from statflow.models.metrics import ClassificationMetrics, classification_metrics

MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
MLFLOW_ARTIFACTS_DIR = PROJECT_ROOT / "mlartifacts"
WINNER_EXPERIMENT = "statflow_winner"


def _init_mlflow() -> None:
    """Point MLflow at the local SQLite backend + artifacts dir.

    MLflow 3.x deprecated the plain filesystem tracking store; SQLite is
    the recommended local backend. One file (mlflow.db) holds runs and
    metrics; artifacts (models, plots) go under mlartifacts/. Both are
    gitignored.
    """
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
    MLFLOW_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    # Create the experiment on first use so we can pin the artifact location
    # into mlartifacts/. On subsequent runs it already exists — just select it.
    if mlflow.get_experiment_by_name(WINNER_EXPERIMENT) is None:
        mlflow.create_experiment(
            WINNER_EXPERIMENT,
            artifact_location=str(MLFLOW_ARTIFACTS_DIR / WINNER_EXPERIMENT),
        )
    mlflow.set_experiment(WINNER_EXPERIMENT)


def fit_and_evaluate_classifier(
    model,
    split: Split,
) -> tuple[Any, dict[str, ClassificationMetrics]]:
    """Fit `model` on train, evaluate on all three splits. Returns (model, metrics).

    Model must expose `.fit(X, y)` and `.predict_proba(X)` returning shape (n, 2).
    Pure function — no MLflow, no I/O. This is the testable core.
    """
    X_tr, y_tr = make_classification_xy(split.train)
    model.fit(X_tr, y_tr)

    metrics: dict[str, ClassificationMetrics] = {}
    for name, part in [("train", split.train), ("val", split.val), ("test", split.test)]:
        if len(part) == 0:
            continue
        X, y = make_classification_xy(part)
        y_prob = model.predict_proba(X)[:, 1]
        metrics[name] = classification_metrics(y, y_prob)

    return model, metrics


def _build_lr_pipeline(params: dict[str, Any]) -> Pipeline:
    """Impute + scale + logistic regression. Median imputation handles the
    cold-start NULLs (early-season games, rookie pitchers) without imputer-
    picked assumptions. Scaling matters because LR is scale-sensitive."""
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(**params)),
        ]
    )


def _log_metrics(metrics: dict[str, ClassificationMetrics]) -> None:
    for split_name, m in metrics.items():
        for k, v in m.as_dict().items():
            mlflow.log_metric(f"{split_name}_{k}", v)


def train_logistic_regression(
    split: Split,
    params: dict[str, Any] | None = None,
    run_name: str = "logistic_regression",
) -> tuple[Pipeline, dict[str, ClassificationMetrics]]:
    """Fit an LR winner classifier, log to MLflow, return (model, metrics)."""
    params = params or {"C": 1.0, "max_iter": 1000, "random_state": 42}
    _init_mlflow()

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.set_tag("model_family", "logistic_regression")

        pipe = _build_lr_pipeline(params)
        pipe, metrics = fit_and_evaluate_classifier(pipe, split)

        _log_metrics(metrics)
        # MLflow 3's default skops serializer rejects some sklearn/numpy
        # internals — fall back to cloudpickle which handles anything.
        mlflow.sklearn.log_model(pipe, name="model", serialization_format="cloudpickle")

    return pipe, metrics


def train_xgb_classifier(
    split: Split,
    params: dict[str, Any] | None = None,
    run_name: str = "xgboost_winner",
) -> tuple[XGBClassifier, dict[str, ClassificationMetrics]]:
    """Fit an XGBoost winner classifier, log to MLflow, return (model, metrics)."""
    # Conservative defaults — MLB prediction is a hard signal. Shallow trees +
    # aggressive min_child_weight fights overfit better than deeper trees.
    params = params or {
        "n_estimators": 150,
        "max_depth": 3,
        "learning_rate": 0.05,
        "min_child_weight": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "eval_metric": "logloss",
        "random_state": 42,
    }
    _init_mlflow()

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.set_tag("model_family", "xgboost")

        model = XGBClassifier(**params)
        model, metrics = fit_and_evaluate_classifier(model, split)

        _log_metrics(metrics)
        mlflow.xgboost.log_model(model, name="model")

    return model, metrics
