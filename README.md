# StatFlow

An end-to-end MLB game prediction pipeline. Ingests data from the free
MLB Stats API, transforms it through a medallion (bronze/silver/gold)
architecture on DuckDB + parquet, trains XGBoost models for game winners
and total runs, tracks experiments in MLflow, orchestrates the daily run
with Prefect, and serves predictions through a Streamlit dashboard and a
FastAPI service.

## What this is (and isn't)

This is a personal project I built to see how close I could get to Vegas
using only public data. I'm not trying to beat the closing line — sharp
books have proprietary injury reports, weather models, sharp-money signal,
and full-time quants. Beating them with the free MLB API isn't the goal.
The goal is to build a rigorous pipeline, honestly measure how the model
performs against a market benchmark, and see where the gap closes and
where it doesn't.

I built it as a portfolio project for entry-level / new-grad data
engineering, data science, and ML engineering roles — the fun of watching
predictions vs. reality every morning is the bonus.

**Currently MLB-only.** NBA is next; the ingest + storage layout is
already namespaced under `src/statflow/sports/` so adding a sport is a
matter of dropping in a new module rather than reshaping the pipeline.
See [`docs/adrs/0001-multi-sport-layout.md`](docs/adrs/0001-multi-sport-layout.md).

## Stack

Python 3.11 · uv · MLB Stats API · DuckDB · parquet · pandas · XGBoost
· MLflow · Prefect · Streamlit · FastAPI · Docker · GitHub Actions
· Cloudflare R2 · dbt (showcase)

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
        ┌────────────────────────┐
        │  Streamlit + FastAPI   │
        └────────────────────────┘

Scheduled daily by GitHub Actions; state synced to Cloudflare R2.
```

## Pipeline layers

**Bronze — raw JSON stored as-is in parquet.** One partition per date per
source (schedule, boxscores, plays, transactions). The `payload` column
holds the raw JSON blob so silver can re-extract fields without hitting
the API again.

**Silver — typed, deduplicated tables via DuckDB SQL.** Three tables:
- `games` — one row per game with targets `home_win`, `total_runs`
- `team_game_stats` — home + away batting/fielding per game
- `pitcher_game_stats` — one row per pitcher appearance

Deduplication uses `QUALIFY ROW_NUMBER() OVER (... ORDER BY ingested_at DESC)`
so re-ingesting a game overwrites cleanly.

**Gold — features + monitoring.** Six intermediate tables plus the final
feature table:
- `team_rolling` — L10 team stats + rest days (anti-leakage windowing)
- `pitcher_form` — L5 SP ERA/K9/rest via cumulative-rate formula
- `bullpen_form` — L10 team bullpen ERA + L3 workload (fatigue proxy)
- `park_factors` — venue vs league runs, trailing 82 games
- `roster_activity` — rolling IL-transaction proxy for team availability
- `features` — final ML input, ~23 features, one row per game
- `predictions` + `prediction_outcomes` — model output + actuals for monitoring

Anti-leakage is enforced structurally with `ROWS BETWEEN N PRECEDING AND
1 PRECEDING` on every rolling aggregate — the current game is excluded
from its own window. Killer tests fail loudly if the SQL ever regresses.

**Models.** Baselines (`home_always_wins`, `mean_runs`) set the bar.
Logistic Regression + Ridge are the linear baselines. XGBoost handles
both targets. Every training run logs params, per-split metrics, feature
importance, and the model artifact to MLflow. A promotion gate tags the
best-performing run per experiment — that's the run `predict.py` serves.
Auto-generated markdown model cards commit to `docs/model_cards/` on
every retrain.

**Orchestration.** A single Prefect flow (`statflow-daily`) wraps
ingest → transform → features → predict → refresh_outcomes. Ingest
re-runs a rolling 3-day window so yesterday's boxscore is guaranteed to
land as `Final` before the next round of predictions. GitHub Actions runs
the flow at 11:00 UTC daily, pulls prior state from Cloudflare R2, pushes
updated state back, and emails on failure.

**Serving.**
- **Streamlit dashboard** — three tabs: today's games (with per-game
  SHAP-equivalent explanations), model performance monitoring (rolling
  metrics + calibration + confidence-tier breakdown + model-vs-baseline),
  and a historical explorer.
- **FastAPI service** — `/health`, `/predictions/today`, `/predictions/{date}`,
  `/performance/rolling` — reads gold parquet directly, no model
  inference at request time.

## Quickstart

### Local (recommended for development)

```bash
uv sync --extra dev

# One-time historical backfill (~6-10 hours for 5 seasons, resumable)
uv run python -m statflow.ingest.backfill --start-season 2019 --end-season 2026

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

# (optional) FastAPI service
uv run uvicorn statflow.api.app:app --reload --port 8000
```

Or, from now on, run the daily flow to keep things fresh:
```bash
uv run python -m statflow.flows            # today
uv run python -m statflow.flows --date 2026-07-27   # a specific date
```

### Docker (for a demo)

```bash
# Populate data first from the host (Docker container doesn't ship data)
uv run python -m statflow.ingest.backfill --start-season 2019 --end-season 2026
uv run python -m statflow.transform && uv run python -m statflow.features
uv run python -m statflow.models train

# Then bring up the dashboard + MLflow UI
docker-compose up --build
```

- Dashboard: <http://localhost:8501>
- MLflow UI: <http://localhost:5000>

## Testing

```bash
uv run pytest        # hermetic — no network
uv run ruff check    # lint
uv run ruff format --check
```

## Milestones

- [x] **M0** — Repo scaffolding, uv, ruff, pytest, CI
- [x] **M1** — MLB API client + bronze ingest (schedule, boxscores, plays, transactions)
- [x] **M2** — Silver layer (DuckDB SQL, 3 tables)
- [x] **M3** — Historical backfill (multi-season) + 8 data quality checks
- [x] **M4** — Gold feature layer (~23 features, anti-leakage tests)
- [x] **M5** — Baselines + LR/Ridge + XGBoost + MLflow tracking + feature importance
- [x] **M6** — Prefect daily flow + GitHub Actions cron
- [x] **M7** — Streamlit dashboard (today's games + model performance + historical explorer)
- [x] **M8** — Docker + polish
- [x] **Post-M8** — Cloudflare R2 as source-of-truth storage; hosted on Streamlit Community Cloud
- [x] **Post-M8** — Weekly retrain workflow + promotion gate + auto model cards
- [x] **Post-M8** — FastAPI service, per-game SHAP-equivalent explanations, dbt showcase project
- [ ] **Next** — Closing-line ingestion + "Model vs. Market" dashboard tab
- [ ] **Next** — NBA extension via `src/statflow/sports/nba/`

Current data volumes and metrics are visible live on the dashboard —
they change every day as the pipeline ingests new games.

## A note on model performance

MLB is genuinely one of the hardest major sports to predict from
public data.

- Home teams win ~54% of games historically — that's the naive baseline.
- Vegas books close around 55–57% accuracy with proprietary data, sharp
  money signal, weather models, and full-time quants.
- The public-data ceiling most rigorous models land at is somewhere in
  the 52–55% range on accuracy and ~2.3–2.6 MAE on total runs.

I'd rather report an honest calibration curve than an inflated headline
number. The dashboard's Model Performance tab shows the real thing —
log loss, Brier score, and MAE, always benchmarked against the naive
baselines on the same completed games. Next iteration adds closing lines
from The Odds API so you can also see the model vs the market.

Highest-leverage feature additions still on the list:
- Point-in-time roster + injury reconstruction from the transactions feed
- Weather / wind / temperature
- Umpire strike-zone tendencies
- Batting order strength (starting lineup)

Each addition is one `.sql` file (or one Python file) plus a test — the
infrastructure is built to make experimentation cheap.

## Data source

MLB Stats API (`statsapi.mlb.com`). Public, free, no auth. Portfolio
use only.
