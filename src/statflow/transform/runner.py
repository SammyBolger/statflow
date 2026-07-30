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
SQL_TRANSFORMS: list[str] = ["games", "team_game_stats", "transactions"]

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
    _build_odds_silver(bronze_dir, silver_dir)


def _register_bronze_views(
    conn: duckdb.DuckDBPyConnection,
    bronze_dir: Path,
) -> None:
    """Expose each bronze table as `bronze_<name>` reading its parquet partitions.

    Skips subdirectories with no parquet files — silver transforms that depend
    on a missing bronze table will fail with a clear "table not found" error.
    """
    for name in BRONZE_VIEWS:
        base = bronze_dir / name
        has_any = base.exists() and any(base.rglob("*.parquet"))
        if not has_any:
            continue
        pattern = str(base / "**" / "*.parquet")
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

    Note: "0.0" is a real value — a pitcher who faced batters but retired
    nobody (e.g., a starter pulled after being immediately shelled). It
    returns 0.0, not None. Only truly missing values return None.
    """
    if ip in (None, ""):
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
            # A player is a pitcher in this game iff their id appears in the
            # pitchers[] array. Filtering on innings_pitched would drop
            # starters who exited immediately (0.0 IP but very much a real
            # appearance) — data-quality checks caught this on real data.
            pitcher_ids = set(pitchers_used)

            for player in team.get("players", {}).values():
                person = player.get("person", {})
                pitcher_id = person.get("id")
                if pitcher_id not in pitcher_ids:
                    continue

                pitching = player.get("stats", {}).get("pitching", {})
                rows.append(
                    {
                        "game_pk": record.game_pk,
                        "team_id": team_id,
                        "pitcher_id": pitcher_id,
                        "pitcher_name": person.get("fullName"),
                        "is_starter": pitcher_id == starter_id,
                        "innings_pitched": _parse_innings_pitched(pitching.get("inningsPitched")),
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


def _american_to_implied_prob(price: float | int) -> float:
    """American moneyline odds → implied probability (0..1), pre-devig.

    Positive numbers (e.g. +150) pay $150 on a $100 bet; negative numbers
    (e.g. -150) require a $150 bet to win $100. Both encode the book's
    price for the outcome; the raw prob still carries the vig, so callers
    normalize a two-outcome market by dividing each raw prob by their sum.
    """
    price = float(price)
    if price < 0:
        return -price / (-price + 100.0)
    return 100.0 / (price + 100.0)


def _summarize_game_odds(payload: dict) -> dict[str, float | None]:
    """Average moneyline (as devigged probs) + total line across books.

    Returns a dict with `market_home_win_prob`, `market_away_win_prob`,
    `market_total_line`, and per-market book counts. Missing values are
    None. Assumes the payload came from The Odds API v4.
    """
    home_team = payload.get("home_team")
    away_team = payload.get("away_team")

    home_probs: list[float] = []
    away_probs: list[float] = []
    totals: list[float] = []

    for book in payload.get("bookmakers", []):
        for market in book.get("markets", []):
            outcomes = market.get("outcomes", [])
            if market.get("key") == "h2h":
                h_raw = a_raw = None
                for o in outcomes:
                    if o.get("name") == home_team and o.get("price") is not None:
                        h_raw = _american_to_implied_prob(o["price"])
                    elif o.get("name") == away_team and o.get("price") is not None:
                        a_raw = _american_to_implied_prob(o["price"])
                if h_raw is not None and a_raw is not None and (h_raw + a_raw) > 0:
                    # Devig by normalizing so the two probs sum to 1.
                    total = h_raw + a_raw
                    home_probs.append(h_raw / total)
                    away_probs.append(a_raw / total)
            elif market.get("key") == "totals":
                # Over/under share the same `point`; take either outcome's point.
                for o in outcomes:
                    if o.get("point") is not None:
                        totals.append(float(o["point"]))
                        break

    def _mean(xs: list[float]) -> float | None:
        return sum(xs) / len(xs) if xs else None

    return {
        "market_home_win_prob": _mean(home_probs),
        "market_away_win_prob": _mean(away_probs),
        "market_total_line": _mean(totals),
        "n_books_moneyline": len(home_probs),
        "n_books_totals": len(totals),
    }


def _build_odds_silver(bronze_dir: Path, silver_dir: Path) -> Path | None:
    """Flatten bronze odds parquet to one row per game, joined to game_pk.

    Bronze holds one row per (game, ingested_at) with the raw Odds API
    payload. Silver averages moneyline (devigged) + total line across
    bookmakers and left-joins to silver.games on (game_date, home_team).
    Returns None if no bronze odds partitions exist yet.
    """
    odds_root = bronze_dir / "odds"
    paths = sorted(odds_root.rglob("*.parquet")) if odds_root.exists() else []
    if not paths:
        return None

    # Read one file at a time and concat, so Hive-style partition-key
    # inference doesn't collide with our internal `date` column.
    bronze_df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    rows: list[dict] = []
    for record in bronze_df.itertuples(index=False):
        payload = json.loads(record.payload)
        summary = _summarize_game_odds(payload)
        rows.append(
            {
                "odds_event_id": record.odds_event_id,
                "odds_date": record.date,
                "home_team": record.home_team,
                "away_team": record.away_team,
                "commence_time": record.commence_time,
                "fetched_at": record.fetched_at,
                **summary,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        # Latest fetch per (date, home_team, away_team) wins — same
        # dedup pattern as the SQL silver tables.
        df = df.sort_values("fetched_at").drop_duplicates(
            subset=["odds_date", "home_team", "away_team"], keep="last"
        )

    # Join to silver.games to attach game_pk when possible. Team names from
    # The Odds API match the MLB Stats API's `team.name` field for MLB in
    # the vast majority of cases; unmatched rows keep game_pk=NULL and
    # surface as a data-quality issue.
    games_path = silver_dir / "games" / "games.parquet"
    if games_path.exists() and not df.empty:
        games = pd.read_parquet(games_path)[
            ["game_pk", "game_date", "home_team_name", "away_team_name"]
        ].copy()
        games["game_date"] = pd.to_datetime(games["game_date"]).dt.date.astype(str)
        df = df.merge(
            games,
            how="left",
            left_on=["odds_date", "home_team", "away_team"],
            right_on=["game_date", "home_team_name", "away_team_name"],
        ).drop(columns=["home_team_name", "away_team_name"])
    else:
        df["game_pk"] = pd.NA
        df["game_date"] = pd.NA

    out_dir = silver_dir / "odds"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "odds.parquet"
    df.to_parquet(path, index=False)
    return path
