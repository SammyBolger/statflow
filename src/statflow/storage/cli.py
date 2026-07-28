"""CLI for R2 sync — used by GH Actions workflows.

Usage:
    uv run python -m statflow.storage pull   # download state from R2
    uv run python -m statflow.storage push   # upload state to R2
"""

from __future__ import annotations

import argparse

from statflow.storage.r2 import pull, push


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m statflow.storage",
        description="Sync silver/gold/mlartifacts between the local tree and R2.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("pull", help="Download current state from R2.")
    sub.add_parser("push", help="Upload updated state to R2.")
    args = parser.parse_args(argv)

    if args.command == "pull":
        n = pull()
        print(f"pulled {n} file(s) from R2")
    else:
        n = push()
        print(f"pushed {n} file(s) to R2")
