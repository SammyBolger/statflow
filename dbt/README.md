# dbt-duckdb project

Optional, coexisting alternative to the Python runner in
`src/statflow/transform/` + `src/statflow/features/`.

## Why both?

- **Python runner (primary):** what the daily Prefect flow calls. Fast,
  no external tools, minimal deps. This is what runs in production.
- **dbt project (this dir):** the same silver + gold models re-expressed
  as a proper dbt project. Demonstrates the SQL layers translate cleanly
  to the industry-standard toolchain (schema tests, DAG resolution,
  `dbt docs`, etc.). Extendable to full parity by copying more `.sql`
  files from `src/statflow/{transform,features}/sql/` following the
  patterns shown.

## Currently in the dbt project

Silver:
- `games` — full port of the Python-runner version, with schema tests

Gold:
- `team_rolling` — pattern demonstration using `{{ ref('games') }}`

The remaining silver models (`team_game_stats`, `pitcher_game_stats`,
`transactions`) and gold intermediates (`pitcher_form`, `bullpen_form`,
`park_factors`, `roster_activity`, `features`) haven't been ported —
this project intentionally stops at "the pattern works, extend as needed".

## Run

```bash
uv sync --extra dev --extra dbt
uv run dbt build --project-dir dbt --profiles-dir dbt
```

Writes silver/gold parquets into the same paths the Python runner uses:
- `data/silver/games/games.parquet`
- `data/gold/team_rolling/team_rolling.parquet`

## Design notes

- Uses **`materialized: external`** with `format: parquet` so each model
  writes a single parquet file (matches the Python runner's output layout).
- Uses **in-memory DuckDB** — no persistent DB file since materializations
  are external.
- **`{{ source('bronze', 'X') }}`** patterns point at raw parquet under
  `data/bronze/`. The Python runner registers the same paths as `bronze_X`
  views; dbt does it via `sources.yml`.
- Schema tests (unique, not_null) live in `models/silver/schema.yml`.
