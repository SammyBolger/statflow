# `sports/nba/`

Placeholder for the NBA pipeline.

## Not implemented yet

This directory exists to show the multi-sport structure — NBA is the
next intended sport to add after MLB migration completes.

## Design notes for the future

- **Data source:** unofficial `stats.nba.com` API via a Python wrapper
  like `nba_api`. Free but rate-limited (~1 request per second polite).
- **Season shape:** ~1,230 regular-season games + playoffs. Fewer games
  than MLB (~2,430) but each game has more information density.
- **Model targets:** winner (spread), total points, potentially per-team
  points.
- **Sport-specific features:**
  - Team pace (possessions/game — huge for total-points prediction)
  - Four Factors (eFG%, TOV%, ORB%, FT/FGA) rolling
  - Back-to-back flag (previous-day game)
  - Rest advantage vs opponent
  - Lineup availability (star players IN/OUT — much bigger signal in
    NBA than MLB because 5 starters vs 9 batters)
  - Home/road splits
- **Season overlap:** NBA runs Oct–June; overlaps with the tail of
  MLB (Oct) + all of NFL. Daily flow needs to know which sports are
  in season on any given day.

## Minimum viable NBA milestone

1. Ingest client for `stats.nba.com` + boxscore + play-by-play endpoints
2. Bronze parquet layout mirroring MLB
3. Silver: games + team_game_stats + player_game_stats (analog of MLB
   pitcher_game_stats)
4. Gold: rolling team form + four factors + rest/travel
5. Model reusing `core/models/` training loop
6. Dashboard tab per sport, or per-sport home page
