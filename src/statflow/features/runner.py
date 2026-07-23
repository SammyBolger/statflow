"""Build the gold feature layer from silver parquet.

Each intermediate feature table is defined by a `.sql` file that reads from
DuckDB views over the silver parquet. A final assembly step (added later)
joins them into `data/gold/features/features.parquet` — the ML input.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from statflow.config import GOLD_DIR, SILVER_DIR

SQL_DIR = Path(__file__).parent / "sql"

SQL_TRANSFORMS: list[str] = ["team_rolling", "pitcher_form", "park_factors"]

SILVER_VIEWS = ("games", "team_game_stats", "pitcher_game_stats")


def build_features(
    silver_dir: Path = SILVER_DIR,
    gold_dir: Path = GOLD_DIR,
) -> None:
    """Rebuild all gold feature intermediates from silver parquet."""
    conn = duckdb.connect()
    _register_silver_views(conn, silver_dir)
    for name in SQL_TRANSFORMS:
        _run_sql_transform(conn, name, gold_dir)


def _register_silver_views(
    conn: duckdb.DuckDBPyConnection,
    silver_dir: Path,
) -> None:
    """Expose each existing silver table as a DuckDB view over its parquet file.

    Missing tables are skipped so tests can build a subset of silver. Any SQL
    that references a missing view will error at execute time with a clear
    "table not found" from DuckDB.
    """
    for name in SILVER_VIEWS:
        path = silver_dir / name / f"{name}.parquet"
        if not path.exists():
            continue
        conn.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{path}')")


def _run_sql_transform(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    gold_dir: Path,
) -> Path:
    """Execute a gold `.sql` file and write the result to parquet."""
    sql = (SQL_DIR / f"{table_name}.sql").read_text()
    df = conn.execute(sql).fetchdf()

    out_dir = gold_dir / table_name
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{table_name}.parquet"
    df.to_parquet(path, index=False)
    return path
