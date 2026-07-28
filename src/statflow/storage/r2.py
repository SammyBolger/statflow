"""Sync silver / gold / mlartifacts between the local project tree and R2.

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

import os
from collections.abc import Iterable
from pathlib import Path

import boto3
from botocore.client import Config

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


def push(
    paths: Iterable[str] = DEFAULT_SYNC_PATHS,
    root: Path = PROJECT_ROOT,
    client=None,
) -> int:
    """Upload every file under each path to R2 under the same relative key."""
    s3 = client or _client()
    bucket = _bucket()
    n = 0
    for rel in paths:
        base = root / rel
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            key = str(path.relative_to(root))
            s3.upload_file(str(path), bucket, key)
            n += 1
    return n


def pull(
    paths: Iterable[str] = DEFAULT_SYNC_PATHS,
    root: Path = PROJECT_ROOT,
    client=None,
) -> int:
    """Download every object under each path prefix from R2 to the local tree."""
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
    return n
