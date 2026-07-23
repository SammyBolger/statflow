"""CLI: run all silver quality checks and print a report.

Usage:
    uv run python -m statflow.quality
    uv run python -m statflow.quality --fail-on-error   # non-zero exit if any fail
"""

from __future__ import annotations

import argparse
import sys

from statflow.quality.runner import run_checks, write_report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m statflow.quality",
        description="Run silver data-quality checks and append to the report.",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Exit with non-zero status if any check fails (for use in CI).",
    )
    args = parser.parse_args(argv)

    results = run_checks()
    write_report(results)

    failed = [r for r in results if not r.passed]
    passed = len(results) - len(failed)

    for r in results:
        marker = "PASS" if r.passed else "FAIL"
        print(f"[{marker}] {r.name}: {r.details}")
    print(f"\n{passed}/{len(results)} checks passed.")

    if failed and args.fail_on_error:
        sys.exit(1)
