"""CLI entry point for building the silver layer.

Usage:
    uv run python -m statflow.transform
"""

from __future__ import annotations

import argparse

import pandas as pd

from statflow.config import SILVER_DIR
from statflow.transform.runner import build_silver


def _print_summary() -> None:
    """Print row counts for each silver table so you can eyeball the build."""
    for path in sorted(SILVER_DIR.glob("*/*.parquet")):
        n = len(pd.read_parquet(path))
        print(f"  {path.parent.name}: {n:,} rows")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m statflow.transform",
        description="Rebuild the silver layer from bronze parquet.",
    )
    parser.parse_args(argv)
    print("Building silver layer...")
    build_silver()
    print("Done.")
    _print_summary()
