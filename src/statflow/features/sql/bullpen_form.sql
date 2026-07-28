-- Gold intermediate: bullpen_form
--
-- Grain: one row per (game_pk, team_id) — for games where relievers appeared.
-- Purpose: team bullpen quality + recent fatigue, going into each game.
--
-- Bullpen features are widely regarded as the single biggest missing signal
-- in a basic MLB model built on pre-game stats alone. Adding two:
--   * bullpen_era_l10 — cumulative ERA over the last 10 team-games (form)
--   * bullpen_ip_l3   — total bullpen innings over the last 3 (fatigue proxy)
--
-- Same anti-leakage discipline as team_rolling and pitcher_form:
-- ROWS BETWEEN N PRECEDING AND 1 PRECEDING excludes the current row.

WITH team_bullpen AS (
    -- One row per (game, team) with that team's relief-pitching totals.
    SELECT
        p.game_pk,
        p.team_id,
        g.game_date,
        SUM(p.innings_pitched) AS bp_ip,
        SUM(p.earned_runs) AS bp_er
    FROM pitcher_game_stats p
    JOIN games g ON g.game_pk = p.game_pk
    WHERE p.is_starter = FALSE
    GROUP BY p.game_pk, p.team_id, g.game_date
)
SELECT
    game_pk,
    team_id,
    SUM(bp_er) OVER w_form * 9.0 / NULLIF(SUM(bp_ip) OVER w_form, 0) AS bullpen_era_l10,
    SUM(bp_ip) OVER w_fatigue AS bullpen_ip_l3,
    COUNT(*) OVER w_form AS bullpen_prior_games
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
