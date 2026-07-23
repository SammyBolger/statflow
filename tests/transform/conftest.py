"""Shared fixtures for silver-transform tests.

Each fixture returns a small helper that writes a synthetic bronze parquet
file so tests can mix and match which bronze tables have data.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def bronze_dir(tmp_path: Path) -> Path:
    return tmp_path / "bronze"


@pytest.fixture
def silver_dir(tmp_path: Path) -> Path:
    return tmp_path / "silver"


@pytest.fixture
def write_schedule(bronze_dir: Path) -> Callable[..., Path]:
    def _write(
        games: list[dict],
        target_date: str = "2026-07-22",
        ingested_at: datetime | None = None,
    ) -> Path:
        ingested_at = ingested_at or datetime.now(UTC)
        df = pd.DataFrame(
            {
                "game_pk": [g["gamePk"] for g in games],
                "game_date": target_date,
                "payload": [json.dumps(g) for g in games],
                "ingested_at": ingested_at,
            }
        )
        out = bronze_dir / "schedule" / f"date={target_date}" / "schedule.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        return out

    return _write


@pytest.fixture
def write_boxscores(bronze_dir: Path) -> Callable[..., Path]:
    def _write(
        boxscores: list[dict],
        target_date: str = "2026-07-22",
        ingested_at: datetime | None = None,
    ) -> Path:
        """`boxscores` is a list of (game_pk, payload_dict) tuples-or-dicts."""
        ingested_at = ingested_at or datetime.now(UTC)
        df = pd.DataFrame(
            {
                "game_pk": [b["game_pk"] for b in boxscores],
                "payload": [json.dumps(b["payload"]) for b in boxscores],
                "ingested_at": ingested_at,
            }
        )
        out = bronze_dir / "boxscores" / f"date={target_date}" / "boxscores.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        return out

    return _write


@pytest.fixture
def write_empty_bronze(bronze_dir: Path) -> Callable[[str], Path]:
    """Write an empty bronze parquet with the right schema for tables a test doesn't populate."""

    def _write(name: str, target_date: str = "2026-07-22") -> Path:
        if name in ("schedule", "boxscores", "plays"):
            df = pd.DataFrame(
                {
                    "game_pk": pd.Series([], dtype="int64"),
                    "payload": pd.Series([], dtype="object"),
                    "ingested_at": pd.Series([], dtype="datetime64[ns, UTC]"),
                }
            )
            if name == "schedule":
                df["game_date"] = pd.Series([], dtype="object")
        elif name == "transactions":
            df = pd.DataFrame(
                {
                    "transaction_id": pd.Series([], dtype="int64"),
                    "date": pd.Series([], dtype="object"),
                    "payload": pd.Series([], dtype="object"),
                    "ingested_at": pd.Series([], dtype="datetime64[ns, UTC]"),
                }
            )
        else:
            raise ValueError(f"unknown bronze table: {name}")

        out = bronze_dir / name / f"date={target_date}" / f"{name}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        return out

    return _write
