# CLAUDE.md — house rules for this repo

Read this at the start of every session. If your instinct fights any of
these rules, ask before overriding.

## What this project is

StatFlow is a personal MLB (soon: multi-sport) game-prediction pipeline
Sammy is building as a portfolio piece for entry-level / new-grad data
engineering, data science, and ML engineering roles.

The interview story matters as much as the code. Every line has to be
one Sammy can read, explain, and defend cold. Prefer boring correctness
over clever abstraction — see "Code style" below.

## Never do

- **Never add `Co-Authored-By: Claude ...` trailers to commits.** GitHub
  surfaces co-authors as repo contributors; this needs to look like
  Sammy's project on his GitHub. Not negotiable.
- **Never `git push --force`, `git reset --hard`, or discard uncommitted
  changes** without an explicit ask.
- **Never skip pre-commit hooks** (`--no-verify`). If a hook fails, fix
  the underlying issue.
- **Never introduce new top-level dependencies** without asking. `uv add`
  is a real decision, not a reflex.

## Code style — write early-career

Sammy is 22 and needs to defend every line in interviews. Code that's
"too good" sounds implausible and is hard to discuss.

**Yes:**
- Plain functions and classes with clear names.
- Simple type hints (`str`, `list[dict]`, `Optional[X]`).
- ABCs where they earn their keep (one obvious base class, not a
  hierarchy).
- Standard library-and-framework-docs patterns (FastAPI, SQLAlchemy,
  pytest as shown in their docs).
- Short docstrings on public functions explaining what's non-obvious.
- Small obvious helpers.

**No:**
- Metaclasses, descriptors, `__init_subclass__` tricks.
- Heavy generics (`ParamSpec`, complex `TypeVar` bounds, variance
  gymnastics).
- Protocol classes when an ABC is fine.
- Dependency-injection containers, factory-factories, registries.
- Abstracting "just in case" — wait until the second use case appears.
- Walrus operators or dense one-liners chasing cleverness.
- Custom decorators when a function call is fine.

When in doubt: pick the boring option.

## Modeling discipline

- **Time-based splits only.** Random shuffles leak future outcomes into
  training. Split by season: earlier seasons train, later seasons
  validate/test. `split_by_season()` is the only correct entry point.
- **Anti-leakage windows.** Every rolling aggregate uses
  `ROWS BETWEEN N PRECEDING AND 1 PRECEDING` — the current game is
  excluded from its own window. There are killer tests that fail loudly
  if this regresses; don't touch them lightly.
- **Baselines matter.** `HomeAlwaysWinsBaseline` and `MeanRunsBaseline`
  are the bar. If a change makes the model worse than baseline on val,
  that's a red flag — investigate before merging.

## Before committing

- Run `uv run pytest`. All 166+ tests must pass.
- Run `uv run ruff check` and `uv run ruff format --check`.
- Never commit `.env`, `secrets.toml`, `mlartifacts/mlflow.db`, or
  anything under `data/`. `.gitignore` covers these, but sanity-check
  with `git status` before staging.
- Commit messages follow the existing style
  (`fix(features): ...`, `feat(dashboard): ...`, etc.).

## When Sammy asks for a change

- If it touches multiple files or is architecturally non-trivial,
  propose the plan first (`ExitPlanMode` in Claude Code). Don't dive in.
- Prefer editing existing files to creating new ones.
- When you finish, tell Sammy what changed in one or two sentences —
  not a paragraph. He can read the diff.
- After non-trivial ML changes, offer to walk him through the code so
  he understands what shipped — this is the "make it teach you" habit.

## Multi-sport plans

The pipeline is currently MLB-only. NBA is next. See
`src/statflow/sports/README.md` for the layout and the "add a new sport"
checklist. Don't refactor toward multi-sport until Sammy asks for the
NBA work — the ADR captures the intent.
