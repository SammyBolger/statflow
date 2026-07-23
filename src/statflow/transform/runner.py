"""Build the silver layer from bronze parquet using DuckDB.

Most silver tables are defined by a `.sql` file in `transform/sql/` that reads
from bronze views the runner registers ahead of time. Tables where the shape
of the source JSON is a dict-of-dicts (e.g., boxscore.teams.<side>.players)
are built in Python instead — the SQL for iterating dynamic object keys is
harder to read than a small pandas loop.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd

from statflow.config import BRONZE_DIR, SILVER_DIR

SQL_DIR = Path(__file__).parent / "sql"

# Silver tables built purely from `.sql` files. Order doesn't matter here —
# each is derived from bronze only.
SQL_TRANSFORMS: list[str] = ["games", "team_game_stats"]

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
    _build_pitcher_game_stats(conn, silver_dir)


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


def _parse_innings_pitched(ip: str | None) -> float | None:
    """MLB reports innings as X.Y where Y is thirds: '6.1' = 6 1/3, '6.2' = 6 2/3.

    It's a common data-engineering gotcha — the field looks like a float but
    behaves like base-3. This helper does the conversion once.
    """
    if ip in (None, "", "0.0"):
        return None
    whole, _, thirds = ip.partition(".")
    return int(whole) + (int(thirds) / 3 if thirds else 0.0)


def _build_pitcher_game_stats(
    conn: duckdb.DuckDBPyConnection,
    silver_dir: Path,
) -> Path:
    """Flatten boxscore.teams.<side>.players into one row per pitcher appearance."""
    box_df = conn.execute("SELECT game_pk, payload, ingested_at FROM bronze_boxscores").fetchdf()

    rows: list[dict] = []
    for record in box_df.itertuples(index=False):
        payload = json.loads(record.payload)
        for side in ("home", "away"):
            team = payload.get("teams", {}).get(side, {})
            team_id = team.get("team", {}).get("id")
            pitchers_used = team.get("pitchers", [])
            starter_id = pitchers_used[0] if pitchers_used else None

            for player in team.get("players", {}).values():
                pitching = player.get("stats", {}).get("pitching", {})
                ip = _parse_innings_pitched(pitching.get("inningsPitched"))
                if ip is None:
                    continue  # not a pitcher in this game

                person = player.get("person", {})
                pitcher_id = person.get("id")
                rows.append(
                    {
                        "game_pk": record.game_pk,
                        "team_id": team_id,
                        "pitcher_id": pitcher_id,
                        "pitcher_name": person.get("fullName"),
                        "is_starter": pitcher_id == starter_id,
                        "innings_pitched": ip,
                        "hits_allowed": pitching.get("hits"),
                        "runs_allowed": pitching.get("runs"),
                        "earned_runs": pitching.get("earnedRuns"),
                        "walks": pitching.get("baseOnBalls"),
                        "strikeouts": pitching.get("strikeOuts"),
                        "pitches_thrown": pitching.get("numberOfPitches"),
                        "ingested_at": record.ingested_at,
                    }
                )

    df = pd.DataFrame(rows)
    if not df.empty:
        # Same dedup guarantee as the SQL tables: latest ingest wins.
        df = df.sort_values("ingested_at").drop_duplicates(
            subset=["game_pk", "pitcher_id"], keep="last"
        )

    out_dir = silver_dir / "pitcher_game_stats"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "pitcher_game_stats.parquet"
    df.to_parquet(path, index=False)
    return path
