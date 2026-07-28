"""Auto-generate a markdown model card per training run.

Model cards document a specific trained model's hyperparameters + metrics
so the code repo has a permanent, diff-viewable history of what changed
between retrains. Every weekly-retrain workflow appends a new one to
`docs/model_cards/<experiment>/<run_id>.md` and commits it to git.

This is a standard MLOps artifact (Google's Model Cards for Model
Reporting, 2019) — cheap to add, useful for both debugging and
interview conversation.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import mlflow

from statflow.config import PROJECT_ROOT

MODEL_CARDS_DIR = PROJECT_ROOT / "docs" / "model_cards"


def _split_metrics_by_prefix(metrics: dict[str, float]) -> dict[str, dict[str, float]]:
    """Group flat metric names like 'train_accuracy' → {'train': {'accuracy': ...}}."""
    grouped: dict[str, dict[str, float]] = {"train": {}, "val": {}, "test": {}, "other": {}}
    for key, value in metrics.items():
        matched = False
        for prefix in ("train", "val", "test"):
            if key.startswith(f"{prefix}_"):
                grouped[prefix][key[len(prefix) + 1 :]] = value
                matched = True
                break
        if not matched:
            grouped["other"][key] = value
    return {k: v for k, v in grouped.items() if v}


def generate_model_card(run_id: str, model_family: str = "xgboost") -> str:
    """Return a markdown-formatted model card for the given MLflow run."""
    run = mlflow.get_run(run_id)
    params = run.data.params
    metrics = run.data.metrics
    grouped = _split_metrics_by_prefix(metrics)

    trained_at = datetime.fromtimestamp(run.info.start_time / 1000, tz=UTC).isoformat(
        timespec="seconds"
    )
    now = datetime.now(UTC).isoformat(timespec="seconds")

    lines: list[str] = [
        f"# Model card — `{run.info.run_name}`",
        "",
        f"- **MLflow run id:** `{run_id}`",
        f"- **Experiment:** `{run.info.experiment_id}`",
        f"- **Model family:** `{model_family}`",
        f"- **Trained at:** {trained_at}",
        f"- **Card generated:** {now}",
        "",
        "## Hyperparameters",
        "",
        "| Param | Value |",
        "|---|---|",
    ]
    for k, v in sorted(params.items()):
        lines.append(f"| `{k}` | `{v}` |")

    lines.append("")
    lines.append("## Metrics")

    for split_name, split_metrics in grouped.items():
        lines += [
            "",
            f"### `{split_name}`",
            "",
            "| Metric | Value |",
            "|---|---|",
        ]
        for k, v in sorted(split_metrics.items()):
            lines.append(f"| `{k}` | {v:.4f} |")

    return "\n".join(lines) + "\n"


def write_model_card(
    run_id: str,
    experiment_name: str,
    model_family: str = "xgboost",
) -> Path:
    """Write a model card to docs/model_cards/<experiment>/<run_id>.md."""
    md = generate_model_card(run_id, model_family)
    out_dir = MODEL_CARDS_DIR / experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{run_id}.md"
    path.write_text(md)
    return path
