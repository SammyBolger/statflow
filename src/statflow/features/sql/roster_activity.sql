-- Gold intermediate: roster_activity
--
-- MINIMAL PROXY for team injury/roster instability.
--
-- Grain: one row per (team_id, game_date) — for each date, how many
-- availability-affecting transactions has this team had in the last 30 days?
--
-- This is NOT true point-in-time roster reconstruction — that would require
-- replaying every IL move + activation from a known starting roster to
-- materialize who was actually available on each date. That's a milestone-
-- sized effort (see notes/roster_pit.md future work).
--
-- What this DOES give us: a rough "team X has been unusually IL-active
-- lately" signal — high values often mean multiple players hurt, low means
-- healthy roster. Imperfect but nonzero signal, cheap to compute.
--
-- Depends on registered views: transactions, games.

WITH team_dates AS (
    -- One row per (team, game_date) — for each game a team plays.
    SELECT g.home_team_id AS team_id, g.game_date FROM games g
    UNION
    SELECT g.away_team_id AS team_id, g.game_date FROM games g
),
team_daily_events AS (
    -- Per (team, date), count IL/availability-affecting transactions.
    SELECT
        COALESCE(t.from_team_id, t.to_team_id) AS team_id,
        t.transaction_date,
        COUNT(*) AS n_events
    FROM transactions t
    WHERE t.affects_availability
      AND COALESCE(t.from_team_id, t.to_team_id) IS NOT NULL
    GROUP BY 1, 2
),
joined AS (
    SELECT
        td.team_id,
        td.game_date,
        COALESCE(tde.n_events, 0) AS n_events
    FROM team_dates td
    LEFT JOIN team_daily_events tde
        ON tde.team_id = td.team_id AND tde.transaction_date = td.game_date
)
SELECT
    team_id,
    game_date,
    SUM(n_events) OVER (
        PARTITION BY team_id
        ORDER BY game_date
        RANGE BETWEEN INTERVAL 30 DAY PRECEDING AND INTERVAL 1 DAY PRECEDING
    ) AS il_events_l30
FROM joined
ORDER BY team_id, game_date;
