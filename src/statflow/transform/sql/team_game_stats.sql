-- Silver: team_game_stats
--
-- Grain: two rows per game (one for the home team, one for the away team).
-- Source: bronze_boxscores.
-- Depends on the runner having registered `bronze_boxscores` as a view.
--
-- Each row is a team's batting + fielding line for a single game — the
-- raw numbers the gold layer will roll into rolling-form features.

WITH home AS (
    SELECT
        game_pk,
        CAST(json_extract_string(payload, '$.teams.home.team.id') AS INTEGER) AS team_id,
        TRUE AS is_home,
        CAST(json_extract_string(payload, '$.teams.home.teamStats.batting.runs') AS INTEGER)
            AS runs,
        CAST(json_extract_string(payload, '$.teams.home.teamStats.batting.hits') AS INTEGER)
            AS hits,
        CAST(json_extract_string(payload, '$.teams.home.teamStats.batting.atBats') AS INTEGER)
            AS at_bats,
        CAST(json_extract_string(payload, '$.teams.home.teamStats.batting.strikeOuts') AS INTEGER)
            AS strikeouts,
        CAST(json_extract_string(payload, '$.teams.home.teamStats.batting.baseOnBalls') AS INTEGER)
            AS walks,
        CAST(json_extract_string(payload, '$.teams.home.teamStats.batting.homeRuns') AS INTEGER)
            AS home_runs,
        CAST(json_extract_string(payload, '$.teams.home.teamStats.batting.leftOnBase') AS INTEGER)
            AS left_on_base,
        CAST(json_extract_string(payload, '$.teams.home.teamStats.fielding.errors') AS INTEGER)
            AS errors,
        ingested_at
    FROM bronze_boxscores
),
away AS (
    SELECT
        game_pk,
        CAST(json_extract_string(payload, '$.teams.away.team.id') AS INTEGER) AS team_id,
        FALSE AS is_home,
        CAST(json_extract_string(payload, '$.teams.away.teamStats.batting.runs') AS INTEGER)
            AS runs,
        CAST(json_extract_string(payload, '$.teams.away.teamStats.batting.hits') AS INTEGER)
            AS hits,
        CAST(json_extract_string(payload, '$.teams.away.teamStats.batting.atBats') AS INTEGER)
            AS at_bats,
        CAST(json_extract_string(payload, '$.teams.away.teamStats.batting.strikeOuts') AS INTEGER)
            AS strikeouts,
        CAST(json_extract_string(payload, '$.teams.away.teamStats.batting.baseOnBalls') AS INTEGER)
            AS walks,
        CAST(json_extract_string(payload, '$.teams.away.teamStats.batting.homeRuns') AS INTEGER)
            AS home_runs,
        CAST(json_extract_string(payload, '$.teams.away.teamStats.batting.leftOnBase') AS INTEGER)
            AS left_on_base,
        CAST(json_extract_string(payload, '$.teams.away.teamStats.fielding.errors') AS INTEGER)
            AS errors,
        ingested_at
    FROM bronze_boxscores
),
combined AS (
    SELECT * FROM home
    UNION ALL
    SELECT * FROM away
),
deduped AS (
    SELECT * FROM combined
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY game_pk, team_id
        ORDER BY ingested_at DESC
    ) = 1
)
SELECT
    game_pk,
    team_id,
    is_home,
    runs,
    hits,
    at_bats,
    strikeouts,
    walks,
    home_runs,
    left_on_base,
    errors,
    ingested_at
FROM deduped
ORDER BY game_pk, is_home DESC;
