# `sports/` — per-sport ingest, transform, features

Home for sport-specific data pipelines. Each sport gets its own subpackage
that owns the concrete API client + silver SQL + gold features. Shared
abstractions live under `../core/` (splits, metrics, MLflow helpers,
storage, dashboard/API framework).

## Structure

```
sports/
├── mlb/            ← Baseball. See mlb/README.md for status.
├── nba/            ← Basketball. Scaffolded, not yet implemented.
├── wnba/           ← Future.
├── nhl/            ← Future.
└── nfl/            ← Future.
```

## Why this layout

- **`core/`** holds anything that isn't sport-specific — data splits by
  season/date, evaluation metrics, MLflow experiment helpers, R2 sync,
  Prefect flow scaffolding, Streamlit component library.
- **`sports/<name>/`** holds anything that IS sport-specific — the API
  client for that league, silver SQL that knows what stats exist, gold
  features that encode domain knowledge (park factors are MLB-only,
  pace/four-factors are NBA-only, etc.).

Adding a new sport = create `sports/<new>/` with its own ingest + silver
+ features + a `flows/daily.py` that wires it together. No changes to
`core/`, no changes to any other sport.

## Current status

The MLB pipeline currently lives at the top level
(`src/statflow/{ingest,transform,features,models,...}`) as a legacy
of when this was an MLB-only project. Migration into `sports/mlb/`
is planned — this directory is the destination.

See the migration plan in `../../../docs/adrs/0001-multi-sport-layout.md`
(TBD) or the notes below.

## Migration plan

Roughly in order:

1. `ingest/` → `sports/mlb/ingest/` (5 files)
2. `transform/` → `sports/mlb/transform/`
3. `features/` → `sports/mlb/features/`
4. `models/` → mostly stays as `core/models/`; MLB-specific `FEATURE_COLS`
   moves into `sports/mlb/models/features.py`
5. Update all `python -m statflow.X` invocations (in workflows, README,
   docker-compose) to `python -m statflow.sports.mlb.X`
6. Same for tests

Each step should keep the daily flow and dashboard working. Recommend
doing one directory at a time, running `uv run pytest` + a manual
daily-flow run after each.
