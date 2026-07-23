-- Gold intermediate: park_factors
--
-- Grain: one row per game_pk (each game is played at exactly one venue).
-- Purpose: capture how offensive-friendly a ballpark is, relative to the
-- league. A value > 1.0 means the park boosts run scoring; < 1.0 suppresses.
--
-- Anti-leakage: both the venue window and the league window exclude the
-- current game via `ROWS BETWEEN 82 PRECEDING AND 1 PRECEDING`. 82 games is
-- roughly one full home schedule at a venue — enough for a stable estimate
-- without being so wide that stadium renovations get washed out.
--
-- Only Final games are used as inputs — scheduled games have NULL total_runs
-- and would dilute the average. But we compute a row for every game_pk
-- (including scheduled ones) so downstream joins are clean.

WITH final_games AS (
    SELECT
        game_pk,
        game_date,
        venue_id,
        total_runs
    FROM games
    WHERE status = 'Final' AND total_runs IS NOT NULL
),
venue_stats AS (
    SELECT
        game_pk,
        venue_id,
        AVG(total_runs) OVER venue_window AS venue_avg_runs,
        AVG(total_runs) OVER league_window AS league_avg_runs,
        COUNT(*) OVER venue_window AS n_prior_venue_games
    FROM final_games
    WINDOW
        venue_window AS (
            PARTITION BY venue_id
            ORDER BY game_date, game_pk
            ROWS BETWEEN 82 PRECEDING AND 1 PRECEDING
        ),
        league_window AS (
            ORDER BY game_date, game_pk
            ROWS BETWEEN 82 PRECEDING AND 1 PRECEDING
        )
)
SELECT
    vs.game_pk,
    vs.venue_id,
    fg.game_date,  -- needed for ASOF joins in features.sql
    vs.venue_avg_runs,
    vs.league_avg_runs,
    vs.venue_avg_runs / NULLIF(vs.league_avg_runs, 0) AS venue_park_factor_runs,
    vs.n_prior_venue_games
FROM venue_stats vs
JOIN final_games fg ON fg.game_pk = vs.game_pk
ORDER BY vs.game_pk;
