"""Tests for the R2 sync helpers.

We don't hit real R2 — we pass in a mocked client and verify the correct
`upload_file` / `download_file` / `list_objects_v2` calls fire with the
right (bucket, key, path) args.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from statflow.storage import r2


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("R2_ACCOUNT_ID", "test-account")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "test-key")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "test-secret")
    monkeypatch.setenv("R2_BUCKET", "test-bucket")


def _touch(root: Path, rel: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"data")
    return path


def test_push_uploads_every_file_under_paths(tmp_path):
    _touch(tmp_path, "data/silver/games/games.parquet")
    _touch(tmp_path, "data/silver/team_game_stats/team_game_stats.parquet")
    _touch(tmp_path, "mlartifacts/mlflow.db")
    _touch(tmp_path, "data/bronze/schedule/date=2026-07-27/schedule.parquet")  # ignored

    client = MagicMock()
    n = r2.push(
        paths=("data/silver", "data/gold", "mlartifacts"),
        root=tmp_path,
        client=client,
    )

    assert n == 3
    uploaded_keys = {call.args[2] for call in client.upload_file.call_args_list}
    assert uploaded_keys == {
        "data/silver/games/games.parquet",
        "data/silver/team_game_stats/team_game_stats.parquet",
        "mlartifacts/mlflow.db",
    }
    for call in client.upload_file.call_args_list:
        assert call.args[1] == "test-bucket"


def test_push_skips_missing_directories(tmp_path):
    _touch(tmp_path, "data/silver/games/games.parquet")
    client = MagicMock()
    n = r2.push(paths=("data/silver", "mlartifacts"), root=tmp_path, client=client)
    assert n == 1  # only the silver file — mlartifacts didn't exist


def test_pull_downloads_all_listed_objects(tmp_path):
    client = MagicMock()
    client.get_paginator.return_value.paginate.side_effect = [
        # First call: prefix "data/silver"
        [{"Contents": [{"Key": "data/silver/games/games.parquet"}]}],
        # Second call: prefix "data/gold"
        [{"Contents": [{"Key": "data/gold/features/features.parquet"}]}],
        # Third call: prefix "mlartifacts"
        [{"Contents": [{"Key": "mlartifacts/mlflow.db"}]}],
    ]

    n = r2.pull(
        paths=("data/silver", "data/gold", "mlartifacts"),
        root=tmp_path,
        client=client,
    )

    assert n == 3
    downloaded_keys = {call.args[1] for call in client.download_file.call_args_list}
    assert downloaded_keys == {
        "data/silver/games/games.parquet",
        "data/gold/features/features.parquet",
        "mlartifacts/mlflow.db",
    }
    # Local dirs got created for each file's destination
    for call in client.download_file.call_args_list:
        local_path = Path(call.args[2])
        assert local_path.parent.exists()


def test_pull_handles_empty_bucket_gracefully(tmp_path):
    client = MagicMock()
    client.get_paginator.return_value.paginate.return_value = [{"Contents": []}]
    n = r2.pull(paths=("data/silver",), root=tmp_path, client=client)
    assert n == 0
    client.download_file.assert_not_called()
