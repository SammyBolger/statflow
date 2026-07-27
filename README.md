# StatFlow

End-to-end MLB game prediction pipeline. Ingests data from the free MLB
Stats API, transforms it through a medallion architecture using DuckDB
and parquet, trains XGBoost models to predict game winners and total
runs, tracks experiments in MLflow, orchestrates the daily pipeline with
Prefect, and serves predictions through a Streamlit dashboard.

Built as an entry-level portfolio project — every layer emphasizes
explainable, defensible engineering choices over clever abstractions.

## Stack

Python 3.11 · uv · MLB Stats API · DuckDB · parquet · pandas · XGBoost
· MLflow · Prefect · Streamlit · Docker · GitHub Actions

## Architecture

```
┌──────────────────────┐
│  MLB Stats API       │  (free, no auth)
│  statsapi.mlb.com    │
└──────────┬───────────┘
           │  HTTP + JSON
           ▼
┌──────────────────────────────────────────────────────────────┐
│  PREFECT DAILY FLOW (statflow-daily)                         │
│  ingest ─▶ transform ─▶ features ─▶ predict ─▶ outcomes      │
└─────────┬────────────────┬────────────────┬──────────────────┘
          ▼                ▼                ▼
   ┌──────────────────────────────────────────────┐
   │  PARQUET LAKE (Hive-partitioned by date)     │
   │  data/bronze/  data/silver/  data/gold/      │
   └────────────────┬─────────────────────────────┘
                    │ SQL over parquet
                    ▼
           ┌────────────────┐
           │    DuckDB      │◀── silver + gold transforms
           └────────┬───────┘
                    │
        ┌───────────┴─────────────┐
        ▼                         ▼
  ┌───────────┐            ┌──────────────┐
  │  XGBoost  │────logs───▶│    MLflow    │
  │  models   │            │  tracking    │
  │           │◀───load────│  (SQLite)    │
  └─────┬─────┘            └──────────────┘
        │
        ▼
  ┌─────────────────────────────────┐
  │  predictions + outcomes parquet │
  └───────────────┬─────────────────┘
                  ▼
          ┌──────────────┐
          │  Streamlit   │
          │  dashboard   │
          └──────────────┘

Scheduled daily by GitHub Actions cron.
```

## Pipeline layers

**Bronze — raw JSON stored as-is in parquet.** One partition per date
per source (schedule, boxscores, plays, transactions). The `payload`
column holds the raw JSON blob so silver can re-extract without hitting
the API again.

**Silver — typed, deduplicated tables via DuckDB SQL.** Three tables:
- `games` — one row per game with targets `home_win`, `total_runs`
- `team_game_stats` — home + away batting/fielding per game
- `pitcher_game_stats` — one row per pitcher appearance

Deduplication uses `QUALIFY ROW_NUMBER() OVER (... ORDER BY ingested_at DESC)`
so re-ingesting a game overwrites cleanly.

**Gold — features + monitoring.** Four intermediate tables plus the
final feature table:
- `team_rolling` — L10 team stats + rest days (anti-leakage windowing)
- `pitcher_form` — L5 SP ERA/K9/rest via cumulative-rate formula
- `park_factors` — venue vs league runs, trailing 82 games
- `features` — final ML input, 18 features, one row per game
- `predictions` + `prediction_outcomes` — model output + actuals for monitoring

Anti-leakage is enforced structurally with `ROWS BETWEEN N PRECEDING AND
1 PRECEDING` on every rolling aggregate — the current game is excluded
from its own window. Killer tests would fail loudly if the SQL ever
regressed.

**Models.** Baselines (`home_always_wins`, `mean_runs`) set the bar.
Logistic Regression + Ridge are the linear baselines. XGBoost handles
both targets. Every training run logs params, per-split metrics, and
the model artifact to MLflow (SQLite backend under `mlartifacts/`).

**Orchestration.** A single Prefect flow (`statflow-daily`) wraps
ingest → transform → features → predict → refresh_outcomes. The ingest
step retries with exponential backoff because it's the only step
touching the network. GitHub Actions runs the flow at 11:00 UTC daily
and caches accumulated state across runs.

**Dashboard.** Streamlit with two tabs: today's games (matchups +
predictions) and model performance (rolling metrics + calibration
plot). Data loaders are pure functions in a separate module — tested
without a Streamlit runtime.

## Quickstart

### Local (recommended for development)

```bash
uv sync --extra dev

# One-time historical backfill (~1-2 hours, resumable)
uv run python -m statflow.ingest.backfill --start-season 2024 --end-season 2026

# Build silver + gold, run quality checks
uv run python -m statflow.transform
uv run python -m statflow.features
uv run python -m statflow.quality

# Train models
uv run python -m statflow.models train

# Launch the dashboard
uv run streamlit run src/statflow/dashboard/app.py       # http://localhost:8501

# (optional) MLflow tracking UI
uv run mlflow ui --backend-store-uri sqlite:///mlartifacts/mlflow.db  # http://localhost:5000
```

Or from now on, run the daily flow to keep things fresh:
```bash
uv run python -m statflow.flows            # today
uv run python -m statflow.flows --date 2026-07-27   # a specific date
```

### Docker (for a demo)

```bash
# Populate data first from the host (Docker container doesn't ship data)
uv run python -m statflow.ingest.backfill --start-season 2024 --end-season 2026
uv run python -m statflow.transform && uv run python -m statflow.features
uv run python -m statflow.models train

# Then bring up the dashboard + MLflow UI
docker-compose up --build
```

- Dashboard: <http://localhost:8501>
- MLflow UI: <http://localhost:5000>

## Testing

```bash
uv run pytest        # 130+ hermetic tests, no network
uv run ruff check    # lint
uv run ruff format --check
```

## Milestones

- [x] **M0** — Repo scaffolding, uv, ruff, pytest, CI
- [x] **M1** — MLB API client + bronze ingest (schedule, boxscores, plays, transactions)
- [x] **M2** — Silver layer (DuckDB SQL, 3 tables)
- [x] **M3** — Historical backfill (7,337 games across 3 seasons) + 8 data quality checks
- [x] **M4** — Gold feature layer (18 features, anti-leakage tests)
- [x] **M5** — Baselines + LR/Ridge + XGBoost + MLflow tracking + feature importance
- [x] **M6** — Prefect daily flow + GitHub Actions cron
- [x] **M7** — Streamlit dashboard (today's games + model performance monitoring)
- [x] **M8** — Docker + polish

## A note on model performance

Beating baselines in MLB prediction is legitimately hard.

- Home teams win ~54% of games historically — that's the naive baseline
- Vegas books achieve ~55% with far more data (bullpen, weather, umpires, line-move signal)
- My XGBoost matches the baseline within 0.001 log-loss on validation

That's honest, not a bug. The model is well-calibrated but the feature
set is signal-limited. To meaningfully improve, the next round of work
would be:

- Bullpen state and per-reliever fatigue
- Point-in-time roster + injury reconstruction from the transactions feed
- Weather / wind / temperature
- Umpire strike-zone tendencies

The infrastructure is deliberately built to make those additions cheap
— each new feature is one `.sql` file (or Python file) plus a test.

## Data source

MLB Stats API (`statsapi.mlb.com`). Public, free, no auth. Portfolio
use only.
