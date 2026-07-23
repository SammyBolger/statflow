"""Build the silver layer from bronze parquet using DuckDB.

Each silver table is defined by a `.sql` file in `transform/sql/` that reads
from a set of bronze views the runner registers ahead of time. The runner
executes the SQL, fetches a DataFrame, and writes it to silver parquet.

This split — SQL for logic, Python for glue — keeps the transformations
readable and diff-friendly while giving us DuckDB's JSON handling and
window functions.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from statflow.config import BRONZE_DIR, SILVER_DIR

SQL_DIR = Path(__file__).parent / "sql"

# Silver tables built purely from `.sql` files. Order doesn't matter here —
# each is derived from bronze only.
SQL_TRANSFORMS: list[str] = ["games"]

# Which bronze tables we expose as DuckDB views.
BRONZE_VIEWS = ("schedule", "boxscores", "plays", "transactions")


def build_silver(
    bronze_dir: Path = BRONZE_DIR,
    silver_dir: Path = SILVER_DIR,
) -> None:
    """Rebuild all silver tables from bronze parquet."""
    conn = duckdb.connect()
    _register_bronze_views(conn, bronze_dir)
    for name in SQL_TRANSFORMS:
        _run_sql_transform(conn, name, silver_dir)


def _register_bronze_views(
    conn: duckdb.DuckDBPyConnection,
    bronze_dir: Path,
) -> None:
    """Expose each bronze table as `bronze_<name>` reading its parquet partitions."""
    for name in BRONZE_VIEWS:
        pattern = str(bronze_dir / name / "**" / "*.parquet")
        conn.execute(
            f"CREATE OR REPLACE VIEW bronze_{name} AS SELECT * FROM read_parquet('{pattern}')"
        )


def _run_sql_transform(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    silver_dir: Path,
) -> Path:
    """Execute a silver `.sql` file and write the result to parquet."""
    sql = (SQL_DIR / f"{table_name}.sql").read_text()
    df = conn.execute(sql).fetchdf()

    out_dir = silver_dir / table_name
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{table_name}.parquet"
    df.to_parquet(path, index=False)
    return path
