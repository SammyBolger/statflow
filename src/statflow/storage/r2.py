"""Sync silver / gold / mlartifacts between the local project tree and R2.

The `pull()` helper also rewrites MLflow's SQLite DB after downloading, so
artifact paths (which MLflow always stores as absolute) resolve on the
current machine — otherwise a DB uploaded from a laptop wouldn't work on
a GH Actions runner and vice versa.

R2 is Cloudflare's S3-compatible object storage — we use it as the persistent
source of truth so multiple workflows (daily flow, weekly retrain) and the
hosted dashboard all see the same state.

Credentials come from env vars:
  R2_ACCOUNT_ID          - Cloudflare account id (used to build the endpoint URL)
  R2_ACCESS_KEY_ID       - R2 API token access key
  R2_SECRET_ACCESS_KEY   - R2 API token secret
  R2_BUCKET              - bucket name (defaults to 'statflow')

Usage from Python:
    from statflow.storage.r2 import push, pull
    pull()   # download current state
    push()   # upload updated state

Or from the shell:
    uv run python -m statflow.storage pull
    uv run python -m statflow.storage push
"""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
from collections.abc import Iterable
from pathlib import Path

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from statflow.config import PROJECT_ROOT

# Directories that need to persist across workflow runs. Bronze is deliberately
# excluded — it's large and rebuildable from re-ingesting; the GH Actions
# cache handles it between runs.
DEFAULT_SYNC_PATHS: tuple[str, ...] = (
    "data/silver",
    "data/gold",
    "mlartifacts",
)


def _client():
    """Build a boto3 S3 client pointed at R2 via env-var credentials."""
    account_id = os.environ["R2_ACCOUNT_ID"]
    return boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
        config=Config(signature_version="s3v4"),
    )


def _bucket() -> str:
    return os.environ.get("R2_BUCKET", "statflow")


def _md5_of_file(path: Path) -> str:
    """MD5 of a file's contents in hex — matches R2's ETag for non-multipart uploads."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _remote_etag(s3, bucket: str, key: str) -> str | None:
    """Return the R2 object's ETag (unquoted) or None if it doesn't exist."""
    try:
        head = s3.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in {"404", "NoSuchKey", "NotFound"}:
            return None
        return None  # any other error → err on side of uploading
    return head.get("ETag", "").strip('"')


def push(
    paths: Iterable[str] = DEFAULT_SYNC_PATHS,
    root: Path = PROJECT_ROOT,
    client=None,
) -> int:
    """Upload every file under each path to R2 — skipping files that are
    already present with matching content (ETag / MD5 equal).

    Incremental sync dramatically reduces daily-flow upload volume:
    most parquet files are re-generated identically each morning.
    """
    s3 = client or _client()
    bucket = _bucket()
    uploaded = 0
    for rel in paths:
        base = root / rel
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            key = str(path.relative_to(root))
            local_md5 = _md5_of_file(path)
            remote = _remote_etag(s3, bucket, key)
            if remote == local_md5:
                continue  # unchanged — skip
            s3.upload_file(str(path), bucket, key)
            uploaded += 1
    return uploaded


def pull(
    paths: Iterable[str] = DEFAULT_SYNC_PATHS,
    root: Path = PROJECT_ROOT,
    client=None,
) -> int:
    """Download every object under each path prefix from R2 to the local tree.

    After downloading, rewrite MLflow's DB so artifact paths point at the
    current machine's project root — MLflow stores absolute paths, so any
    DB moved between machines has otherwise-broken artifact URIs.
    """
    s3 = client or _client()
    bucket = _bucket()
    n = 0
    paginator = s3.get_paginator("list_objects_v2")
    for rel in paths:
        for page in paginator.paginate(Bucket=bucket, Prefix=rel):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                local = root / key
                local.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(bucket, key, str(local))
                n += 1

    _rewrite_mlflow_artifact_paths(root)
    return n


def _rewrite_mlflow_artifact_paths(root: Path) -> None:
    """Point MLflow's artifact paths at the current project root.

    MLflow always stores absolute paths for artifact_location (per experiment)
    and artifact_uri (per run). When a DB moves between machines, those paths
    are stale. We find the `mlartifacts/...` or `mlruns/...` segment inside
    each stored path and re-prefix it with the current project root.
    """
    db_path = root / "mlartifacts" / "mlflow.db"
    if not db_path.exists():
        return

    new_prefix = str(root)
    pattern = re.compile(r"^(.*?)/(mlartifacts|mlruns)(/.*|$)")

    # MLflow 3 splits absolute paths across several tables. All of these
    # need the same rewrite; MLflow otherwise can't find the model files.
    targets = [
        ("experiments", "artifact_location"),
        ("runs", "artifact_uri"),
        ("logged_models", "artifact_location"),
        ("model_versions", "storage_location"),
    ]

    conn = sqlite3.connect(db_path)
    try:
        existing_tables = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        for table, col in targets:
            if table not in existing_tables:
                continue  # older MLflow schema without this table
            rows = list(conn.execute(f"SELECT rowid, {col} FROM {table}"))
            for rowid, path in rows:
                if not path:
                    continue
                m = pattern.match(path)
                if not m:
                    continue
                new_path = f"{new_prefix}/{m.group(2)}{m.group(3)}"
                if new_path != path:
                    conn.execute(
                        f"UPDATE {table} SET {col} = ? WHERE rowid = ?",
                        (new_path, rowid),
                    )
        conn.commit()
    finally:
        conn.close()
