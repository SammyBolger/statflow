-- Gold intermediate: bullpen_form
--
-- Grain: one row per (game_pk, team_id) for every game (Final + Scheduled).
-- Purpose: team bullpen quality + recent fatigue, going into each game.
--
-- Bullpen features are widely regarded as the single biggest missing signal
-- in a basic MLB model built on pre-game stats alone. Two features:
--   * bullpen_era_l10 — cumulative ERA over the last 10 team-games (form)
--   * bullpen_ip_l3   — total bullpen innings over the last 3 (fatigue proxy)
--
-- Anti-leakage: ROWS BETWEEN N PRECEDING AND 1 PRECEDING excludes the current row.
--
-- Anti-scaffold: pitcher_game_stats has no rows for scheduled games (no
-- relievers have thrown yet), so joining directly would drop scheduled
-- games entirely. Instead we start from a team-per-game skeleton derived
-- from silver.games, then LEFT JOIN to reliever stats. Non-Final rows
-- contribute NULL to the aggregates and are ignored by SUM.
--
-- Depends on registered views: games, pitcher_game_stats.

WITH team_game_pairs AS (
    -- Skeleton: one row per (game, team) for every game in silver.games.
    -- Scheduled games get a row here so features.sql's direct join finds them.
    SELECT g.game_pk, g.game_date, g.status, g.home_team_id AS team_id FROM games g
    UNION ALL
    SELECT g.game_pk, g.game_date, g.status, g.away_team_id AS team_id FROM games g
),
team_bullpen AS (
    -- Reliever totals per (game, team). CASE-guard: only Final games
    -- contribute real numbers; everything else is NULL and gets ignored
    -- by the window SUMs below.
    SELECT
        tgp.game_pk,
        tgp.team_id,
        tgp.game_date,
        SUM(CASE WHEN tgp.status = 'Final' THEN p.innings_pitched END) AS bp_ip,
        SUM(CASE WHEN tgp.status = 'Final' THEN p.earned_runs END) AS bp_er
    FROM team_game_pairs tgp
    LEFT JOIN pitcher_game_stats p
        ON p.game_pk = tgp.game_pk
        AND p.team_id = tgp.team_id
        AND p.is_starter = FALSE
    GROUP BY tgp.game_pk, tgp.team_id, tgp.game_date
)
SELECT
    game_pk,
    team_id,
    SUM(bp_er) OVER w_form * 9.0 / NULLIF(SUM(bp_ip) OVER w_form, 0) AS bullpen_era_l10,
    SUM(bp_ip) OVER w_fatigue AS bullpen_ip_l3,
    COUNT(bp_ip) OVER w_form AS bullpen_prior_games
FROM team_bullpen
WINDOW
    w_form AS (
        PARTITION BY team_id
        ORDER BY game_date, game_pk
        ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
    ),
    w_fatigue AS (
        PARTITION BY team_id
        ORDER BY game_date, game_pk
        ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
    )
ORDER BY team_id, game_date, game_pk;
