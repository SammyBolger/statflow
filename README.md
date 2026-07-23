# StatFlow

MLB game outcome prediction pipeline. Ingests data from the MLB Stats API,
transforms it through a medallion architecture (bronze → silver → gold) with
DuckDB and parquet, trains models to predict game winners and total runs,
tracks experiments with MLflow, and serves predictions through a Streamlit
dashboard.

**Status:** in progress, built one milestone at a time. See milestone plan below.

## Stack

Python 3.11 · uv · MLB Stats API · DuckDB · parquet · pandas · XGBoost ·
MLflow · Prefect · Streamlit · Docker · GitHub Actions

## Getting started

```bash
# install uv if you don't have it: https://docs.astral.sh/uv/
uv sync --extra dev

# run tests
uv run pytest

# run linter
uv run ruff check
uv run ruff format --check
```

## Milestone plan

- [ ] **0** — Repo scaffolding, tooling, CI
- [ ] **1** — MLB Stats API client + bronze ingest to parquet
- [ ] **2** — Silver layer via DuckDB SQL transforms
- [ ] **3** — Historical backfill (2019–present) + data quality checks
- [ ] **4** — Gold feature layer (rolling team stats, pitcher form, injuries)
- [ ] **5** — Baseline models + XGBoost + MLflow tracking (time-series CV)
- [ ] **6** — Prefect flows + GitHub Actions daily schedule
- [ ] **7** — Streamlit dashboard + model performance monitoring
- [ ] **8** — Docker packaging + polish

## Architecture

_TBD — architecture diagram will be added here once Milestone 1 is in place._

## Notes

Data is from the public MLB Stats API. Portfolio project only, not for
commercial use.
