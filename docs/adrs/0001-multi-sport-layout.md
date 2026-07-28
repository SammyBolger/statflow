# ADR 0001 — Multi-sport package layout

**Status:** Accepted, migration in progress.
**Date:** 2026-07-28.

## Context

StatFlow started as an MLB-only project. The top-level structure
(`src/statflow/{ingest,transform,features,models,dashboard,api,...}`) is
fine for one sport but doesn't scale to five (NBA, WNBA, NHL, NFL are on
the roadmap).

Each sport has:
- A different data source (MLB Stats API vs stats.nba.com vs nflfastR vs
  scraped NHL data).
- A different silver schema (baseball has innings/pitchers; basketball
  has quarters/lineups; football has downs/drives).
- Different features that encode domain knowledge (park factors for MLB,
  Four Factors for NBA, EPA/play for NFL).

But every sport also shares a lot:
- Time-based train/val/test splitting.
- MLflow tracking, promotion gates, calibration wrappers.
- R2 sync.
- Data quality check framework.
- Streamlit component library + dashboard scaffolding.

## Decision

Split into `core/` (sport-agnostic) and `sports/<name>/` (sport-specific).

```
src/statflow/
├── core/          # splits, metrics, MLflow helpers, storage,
│                  # dashboard components, API scaffolding
├── sports/
│   ├── mlb/       # everything MLB — ingest, silver, features, sport-specific model config
│   ├── nba/       # (future)
│   └── ...
└── flows/         # per-sport Prefect flows composing core + sport
```

Adding a sport = create `sports/<new>/` + a `flows/<new>_daily.py` — no
edits to `core/` or other sports.

## Consequences

**Positive:**
- Adding NBA (planned) becomes a self-contained milestone.
- Sport-specific code is discoverable via its directory.
- Cross-sport code reuse is enforced (if you copy code between sports,
  it belongs in core).

**Negative:**
- One-time refactor of imports throughout the codebase.
- Longer module paths (`python -m statflow.sports.mlb.ingest.backfill`
  vs `python -m statflow.ingest.backfill`).
- Existing MLflow runs / R2 layout / model cards keep their old paths
  — no data migration needed, only code.

## Migration plan

Done in small commits so the daily flow keeps working between each:

1. Ship the target directory tree with placeholder `__init__.py` files
   and READMEs describing intent. ✅ **Done in this ADR's commit.**
2. Move `ingest/` → `sports/mlb/ingest/`. Update workflow YAML +
   README + tests. Verify daily flow runs.
3. Move `transform/` → `sports/mlb/transform/`. Same verification.
4. Move `features/` → `sports/mlb/features/`.
5. Split `models/`: sport-agnostic parts (data splits, metrics, MLflow
   wrappers) → `core/models/`; sport-specific pieces (FEATURE_COLS,
   feature interpretation labels) → `sports/mlb/models/`.
6. Move `storage/`, `quality/`, `dashboard/`, `api/` → `core/`.
7. Add NBA in `sports/nba/` reusing `core/` primitives.

Each step: `git commit`, push, verify CI + a manual daily flow run,
before starting the next.
