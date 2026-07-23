-- Gold: features
--
-- Grain: one row per game_pk (all games — Final + Scheduled).
-- Purpose: final ML-ready feature table. This is what the model reads.
--
-- Structure:
--   * Direct join to team_rolling (twice — home and away) using game_pk +
--     team_id. Every game has a team_rolling row for each team.
--   * ASOF join to pitcher_form (twice — home probable + away probable).
--     ASOF is required because the probable pitcher for a scheduled game
--     hasn't started this game yet — we look up their most recent prior
--     start's rolling stats. For Final games this returns the same row
--     you'd get from a direct join.
--   * ASOF join to park_factors keyed by venue_id and game_date, for the
--     same reason — scheduled games get the venue's park factor from its
--     most recent prior Final game.
--
-- Depends on registered views: games, team_rolling, pitcher_form,
-- park_factors.

SELECT
    g.game_pk,
    g.game_date,
    g.season,
    g.home_team_id,
    g.away_team_id,
    g.venue_id,

    -- Team rolling — home
    home_tr.runs_scored_l10 AS home_runs_scored_l10,
    home_tr.runs_allowed_l10 AS home_runs_allowed_l10,
    home_tr.win_pct_l10 AS home_win_pct_l10,
    home_tr.days_rest AS home_days_rest,

    -- Team rolling — away
    away_tr.runs_scored_l10 AS away_runs_scored_l10,
    away_tr.runs_allowed_l10 AS away_runs_allowed_l10,
    away_tr.win_pct_l10 AS away_win_pct_l10,
    away_tr.days_rest AS away_days_rest,

    -- Starting pitcher form — home probable
    home_pf.era_l5 AS home_sp_era_l5,
    home_pf.k_per_9_l5 AS home_sp_k_per_9_l5,
    home_pf.days_rest AS home_sp_days_rest,

    -- Starting pitcher form — away probable
    away_pf.era_l5 AS away_sp_era_l5,
    away_pf.k_per_9_l5 AS away_sp_k_per_9_l5,
    away_pf.days_rest AS away_sp_days_rest,

    -- Venue
    pf.venue_park_factor_runs,

    -- Trivial context
    (g.day_night = 'day') AS is_day_game,
    EXTRACT(MONTH FROM g.game_date) AS month,
    g.game_type AS season_type,

    -- Targets (NULL until game is Final)
    g.home_win AS target_home_win,
    g.total_runs AS target_total_runs
FROM games g
LEFT JOIN team_rolling home_tr
    ON home_tr.game_pk = g.game_pk AND home_tr.team_id = g.home_team_id
LEFT JOIN team_rolling away_tr
    ON away_tr.game_pk = g.game_pk AND away_tr.team_id = g.away_team_id
ASOF LEFT JOIN pitcher_form home_pf
    ON home_pf.pitcher_id = g.home_probable_pitcher_id
    AND home_pf.game_date <= g.game_date
ASOF LEFT JOIN pitcher_form away_pf
    ON away_pf.pitcher_id = g.away_probable_pitcher_id
    AND away_pf.game_date <= g.game_date
ASOF LEFT JOIN park_factors pf
    ON pf.venue_id = g.venue_id
    AND pf.game_date <= g.game_date
ORDER BY g.game_date, g.game_pk;
