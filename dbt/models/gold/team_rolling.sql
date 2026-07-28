-- Gold: team_rolling (dbt version of src/statflow/features/sql/team_rolling.sql).
--
-- Demonstrates a gold-layer dbt model — depends on silver.games and
-- silver.team_game_stats via {{ ref() }}. In a full dbt migration all six
-- gold intermediates + the final features model would follow this pattern.
--
-- Note: this doesn't reference team_game_stats yet because that silver
-- model isn't in the dbt project (yet). The Python runner's version does.
-- Kept simpler here as a pattern demonstration.

{{ config(materialized='external', format='parquet') }}

WITH team_perspective AS (
    SELECT
        g.game_pk,
        g.game_date,
        g.home_team_id AS team_id,
        g.home_score AS runs_scored,
        g.away_score AS runs_allowed,
        CASE WHEN g.status = 'Final' THEN g.home_win END AS won
    FROM {{ ref('games') }} g
    UNION ALL
    SELECT
        g.game_pk,
        g.game_date,
        g.away_team_id AS team_id,
        g.away_score AS runs_scored,
        g.home_score AS runs_allowed,
        CASE WHEN g.status = 'Final' THEN NOT g.home_win END AS won
    FROM {{ ref('games') }} g
)
SELECT
    game_pk,
    team_id,
    AVG(runs_scored) OVER w AS runs_scored_l10,
    AVG(runs_allowed) OVER w AS runs_allowed_l10,
    AVG(CAST(won AS INTEGER)) OVER w AS win_pct_l10,
    COUNT(runs_scored) OVER w AS n_prior_games
FROM team_perspective
WINDOW w AS (
    PARTITION BY team_id
    ORDER BY game_date, game_pk
    ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
)
