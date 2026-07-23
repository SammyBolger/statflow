"""CLI entry point for building the gold feature layer.

Usage:
    uv run python -m statflow.features
"""

from __future__ import annotations

import argparse

import pandas as pd

from statflow.config import GOLD_DIR
from statflow.features.runner import build_features


def _print_summary() -> None:
    """Print row counts for each gold table so you can eyeball the build."""
    for path in sorted(GOLD_DIR.glob("*/*.parquet")):
        n = len(pd.read_parquet(path))
        print(f"  {path.parent.name}: {n:,} rows")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m statflow.features",
        description="Rebuild the gold feature layer from silver parquet.",
    )
    parser.parse_args(argv)
    print("Building gold feature layer...")
    build_features()
    print("Done.")
    _print_summary()
