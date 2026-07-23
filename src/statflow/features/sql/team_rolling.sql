-- Gold intermediate: team_rolling
--
-- Grain: one row per (game_pk, team_id) — two rows per game.
-- Purpose: for each team-game, the team's rolling form BEFORE this game
-- (last 10 completed games). The final features.sql joins this table
-- twice — once for the home side, once for the away side.
--
-- Anti-leakage: `ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING` excludes the
-- current row from its own window. Scheduled games are included in the
-- CTE so they get rolling values (with NULLs) but their NULL runs are
-- ignored by AVG when they appear in a later window.
--
-- Depends on registered views: games, team_game_stats.

WITH team_perspective AS (
    SELECT
        g.game_pk,
        g.game_date,
        ts.team_id,
        ts.runs AS runs_scored,
        opp.runs AS runs_allowed,
        CASE
            WHEN g.status = 'Final' AND ts.is_home THEN g.home_win
            WHEN g.status = 'Final' AND NOT ts.is_home THEN NOT g.home_win
        END AS won
    FROM games g
    JOIN team_game_stats ts ON ts.game_pk = g.game_pk
    JOIN team_game_stats opp
        ON opp.game_pk = g.game_pk
        AND opp.team_id != ts.team_id
)
SELECT
    game_pk,
    team_id,
    AVG(runs_scored) OVER w AS runs_scored_l10,
    AVG(runs_allowed) OVER w AS runs_allowed_l10,
    AVG(CAST(won AS INTEGER)) OVER w AS win_pct_l10,
    -- Count only rows with an actual runs value — scheduled games (NULL runs)
    -- shouldn't inflate the "how much history do we have" signal.
    COUNT(runs_scored) OVER w AS n_prior_games
FROM team_perspective
WINDOW w AS (
    PARTITION BY team_id
    ORDER BY game_date, game_pk
    ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
)
ORDER BY team_id, game_date, game_pk;
