# `core/` — cross-sport shared abstractions

Everything in this directory should be sport-agnostic. If it references
"MLB" or "runs" or "innings," it doesn't belong here — put it under
`../sports/mlb/` instead.

## What lives here (after the multi-sport migration)

- **`models/`** — data splits by season/date, evaluation metrics (log
  loss, Brier, MAE, RMSE), MLflow experiment helpers, promotion gate,
  calibration wrappers, per-prediction explanations. All operate on
  a generic `Split(train, val, test)` of tabular features + target.
- **`storage/`** — R2 sync utilities (upload/download/rewrite). Same
  code regardless of what tables are being synced.
- **`quality/`** — CheckResult dataclass + runner. Each sport plugs in
  its own checks.
- **`dashboard/`** — Streamlit component library (metric cards,
  calibration plot, model-vs-baseline chart, historical explorer).
  Each sport composes these into its own page.

## Not-yet-migrated pointers

Right now these live at `src/statflow/{models,storage,quality,dashboard}/`.
The migration is the parallel counterpart to `sports/mlb/` — see
`../sports/README.md`.
