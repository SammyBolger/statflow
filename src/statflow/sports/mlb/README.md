# `sports/mlb/`

MLB (Major League Baseball) pipeline. Uses the free
[MLB Stats API](https://statsapi.mlb.com) — no auth, no rate limits
published, ~5 req/s polite in practice.

## Current status

**Actual code still lives at the top level of `src/statflow/`** — this
directory is where it is being migrated to as part of the multi-sport
refactor. See `../README.md` for the migration plan.

## Sport-specific features (once migrated)

- Park factors — venue offensive/pitcher friendliness (trailing 82 games)
- Starting pitcher form — ERA / K/9 / days rest (trailing 5 starts, base-3
  innings-pitched quirk handled)
- Bullpen form — relief ERA + recent innings (fatigue proxy)
- Rolling team offense — runs scored / allowed / win% (trailing 10 games)
- Rest days between games
- Rough IL activity (30-day rolling count of availability-affecting
  transactions — see `../../features/sql/roster_activity.sql` for the
  honest caveats)
