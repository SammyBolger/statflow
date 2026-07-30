# `sports/` — per-sport ingest, transform, features

Home for sport-specific data pipelines. Each sport gets its own
subpackage that owns the concrete API client + silver SQL + gold
features. Shared abstractions live under `../core/` (splits, metrics,
MLflow helpers, storage, dashboard/API framework).

## Structure

```
sports/
├── mlb/            ← Baseball. See mlb/README.md.
├── nba/            ← Basketball. Not implemented — see nba/README.md.
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

## Current status

The MLB pipeline currently lives at the top level
(`src/statflow/{ingest,transform,features,models,...}`) as a legacy
of when this was an MLB-only project. Two migration options exist:

**Option A — "slot NBA in first, migrate MLB later" (recommended).**
Build the new sport under `sports/<new>/` while MLB stays at the top
level. Coexistence is fine because Python module paths are independent.
Migrate MLB into `sports/mlb/` afterward, once NBA is proven out and the
shape of the sport-specific interface is clear from a real second
implementation.

**Option B — "migrate MLB first, then add NBA."** Do the full refactor
described below before writing any NBA code. Cleaner end state but
delays the fun work.

## Adding a new sport (checklist)

For a sport `<new>` (e.g., `nba`):

1. Create `sports/<new>/` with `ingest/`, `transform/`, `features/`
   subpackages, mirroring the top-level MLB layout.
2. Write an API client — a thin wrapper around the sport's data source
   (see `mlb_api.py` for the shape: session with retries, `.get()` method
   returning JSON).
3. Ingest functions — one per source table, each writing to
   `data/bronze/<new>/<source>/date=YYYY-MM-DD/*.parquet`.
   **Note:** current MLB bronze is at `data/bronze/<source>/...` without
   a sport prefix. When the second sport ships, refactor MLB bronze to
   `data/bronze/mlb/<source>/...` in the same commit that adds the new
   sport, so both sports use the same convention going forward.
4. Silver SQL — bronze → typed tables. Views registered by a
   sport-specific `runner.py`.
5. Gold features SQL — the sport's specific rolling stats / domain
   features. `features.parquet` output should conform to the shared
   `FEATURE_COLS` interface (or extend it).
6. Add a `flows/<new>_daily.py` mirroring `flows/daily.py`. Register the
   flow schedule via a new GitHub Actions workflow file.
7. Dashboard: either a per-sport tab or a per-sport home page. The
   Streamlit app already reads gold parquet directly — the new sport
   just adds another data path.
8. Tests: copy the MLB test layout, seed with fixtures for the new
   sport's silver schema.

## Migration plan (MLB → `sports/mlb/`, if/when we go Option B)

Done in small commits so the daily flow keeps working between each:

1. Move `ingest/` → `sports/mlb/ingest/`. Update workflow YAML +
   README + tests. Verify daily flow runs.
2. Move `transform/` → `sports/mlb/transform/`.
3. Move `features/` → `sports/mlb/features/`.
4. Split `models/`: sport-agnostic parts (data splits, metrics, MLflow
   wrappers) → `core/models/`; sport-specific pieces (FEATURE_COLS,
   feature interpretation labels) → `sports/mlb/models/`.
5. Move `storage/`, `quality/`, `dashboard/`, `api/` → `core/`.

Each step: `git commit`, push, verify CI + a manual daily-flow run,
before starting the next.
