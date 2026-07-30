-- Gold intermediate: team_rolling
--
-- Grain: one row per (game_pk, team_id) — two rows per game, for EVERY game
-- (Final and Scheduled). Rolling stats reflect the team's form going into
-- that game.
--
-- Anti-leakage: `ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING` excludes the
-- current row from its own window.
--
-- Anti-scaffold: runs_scored / runs_allowed / won are only populated for
-- Final games. Scheduled and other non-Final rows contribute NULL and are
-- ignored by AVG / COUNT. This matters because the daily flow re-ingests
-- today's boxscore at 7am — before games have started — so bronze contains
-- scaffold rows with all zeros for scheduled games. Reading scores from
-- silver.games (which respects status) keeps those zeros out of the average.
--
-- Depends on registered views: games.

WITH team_perspective AS (
    SELECT
        g.game_pk,
        g.game_date,
        g.status,
        g.home_team_id AS team_id,
        CASE WHEN g.status = 'Final' THEN g.home_score END AS runs_scored,
        CASE WHEN g.status = 'Final' THEN g.away_score END AS runs_allowed,
        CASE WHEN g.status = 'Final' THEN g.home_win END AS won
    FROM games g
    UNION ALL
    SELECT
        g.game_pk,
        g.game_date,
        g.status,
        g.away_team_id AS team_id,
        CASE WHEN g.status = 'Final' THEN g.away_score END AS runs_scored,
        CASE WHEN g.status = 'Final' THEN g.home_score END AS runs_allowed,
        CASE WHEN g.status = 'Final' THEN NOT g.home_win END AS won
    FROM games g
)
SELECT
    game_pk,
    team_id,
    AVG(runs_scored) OVER w AS runs_scored_l10,
    AVG(runs_allowed) OVER w AS runs_allowed_l10,
    AVG(CAST(won AS INTEGER)) OVER w AS win_pct_l10,
    -- COUNT(col) counts only non-NULL values, so this is "how many prior
    -- Final games contributed to the average" — not just "how many rows
    -- were in the physical window."
    COUNT(runs_scored) OVER w AS n_prior_games,
    -- Days since this team's previous game. LAG uses the ordered window,
    -- ignoring the ROWS BETWEEN clause on `w`.
    DATE_DIFF(
        'day',
        LAG(game_date, 1) OVER team_order,
        game_date
    ) AS days_rest
FROM team_perspective
WINDOW
    w AS (
        PARTITION BY team_id
        ORDER BY game_date, game_pk
        ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
    ),
    team_order AS (
        PARTITION BY team_id ORDER BY game_date, game_pk
    )
ORDER BY team_id, game_date, game_pk;
