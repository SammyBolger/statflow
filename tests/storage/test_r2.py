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
    # Remote HEAD returns None (via ClientError) — file doesn't exist yet remotely.
    from botocore.exceptions import ClientError

    client.head_object.side_effect = ClientError({"Error": {"Code": "404"}}, "HeadObject")

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
    from botocore.exceptions import ClientError

    client.head_object.side_effect = ClientError({"Error": {"Code": "404"}}, "HeadObject")

    n = r2.push(paths=("data/silver", "mlartifacts"), root=tmp_path, client=client)
    assert n == 1  # only the silver file — mlartifacts didn't exist


def test_push_skips_unchanged_files(tmp_path):
    """If the remote ETag matches the local file's MD5, don't re-upload."""
    _touch(tmp_path, "data/silver/games/games.parquet")

    client = MagicMock()
    # Simulate the remote object having the same content as the local file.
    local_md5 = r2._md5_of_file(tmp_path / "data" / "silver" / "games" / "games.parquet")
    client.head_object.return_value = {"ETag": f'"{local_md5}"'}

    n = r2.push(paths=("data/silver",), root=tmp_path, client=client)
    assert n == 0  # skipped — no uploads
    client.upload_file.assert_not_called()


def test_push_reuploads_when_content_changes(tmp_path):
    """If the remote ETag differs from the local file's MD5, re-upload."""
    _touch(tmp_path, "data/silver/games/games.parquet")

    client = MagicMock()
    # Simulate the remote object having *different* content.
    client.head_object.return_value = {"ETag": '"deadbeef"'}

    n = r2.push(paths=("data/silver",), root=tmp_path, client=client)
    assert n == 1
    client.upload_file.assert_called_once()


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


def test_pull_rewrites_mlflow_paths_to_current_root(tmp_path):
    """MLflow stores absolute artifact paths — after downloading a DB from
    another machine, we must rewrite them to point at the current root."""
    import sqlite3

    # Build a minimal MLflow-shaped DB with paths from a "different machine".
    db_path = tmp_path / "mlartifacts" / "mlflow.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE experiments (artifact_location TEXT)")
    conn.execute("CREATE TABLE runs (artifact_uri TEXT)")
    original = "/Users/original/Desktop/statflow"
    conn.execute(f"INSERT INTO experiments VALUES ('{original}/mlartifacts/statflow_winner')")
    conn.execute(
        f"INSERT INTO runs VALUES ('{original}/mlartifacts/statflow_winner/abc123/artifacts')"
    )
    conn.commit()
    conn.close()

    # Simulate a pull that already wrote the DB — we just need the rewrite step.
    client = MagicMock()
    client.get_paginator.return_value.paginate.return_value = [{"Contents": []}]
    r2.pull(paths=("mlartifacts",), root=tmp_path, client=client)

    # DB paths now point at tmp_path.
    conn = sqlite3.connect(db_path)
    exp_path = conn.execute("SELECT artifact_location FROM experiments").fetchone()[0]
    run_path = conn.execute("SELECT artifact_uri FROM runs").fetchone()[0]
    conn.close()
    assert exp_path == f"{tmp_path}/mlartifacts/statflow_winner"
    assert run_path == f"{tmp_path}/mlartifacts/statflow_winner/abc123/artifacts"
