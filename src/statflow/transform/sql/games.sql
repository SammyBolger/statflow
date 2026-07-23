-- Silver: games
--
-- Grain: one row per game (game_pk).
-- Source: bronze_schedule (raw JSON payload per game).
-- Depends on the runner having registered `bronze_schedule` as a view.
--
-- Targets we care about downstream:
--   * home_win   (BOOLEAN) — classification target
--   * total_runs (INTEGER) — regression target
-- Both are NULL until status = 'Final' so the model can filter cleanly.

WITH parsed AS (
    SELECT
        game_pk,
        CAST(game_date AS DATE) AS game_date,
        CAST(json_extract_string(payload, '$.gameDate') AS TIMESTAMP) AS game_datetime_utc,
        CAST(json_extract_string(payload, '$.season') AS INTEGER) AS season,
        json_extract_string(payload, '$.gameType') AS game_type,
        json_extract_string(payload, '$.status.detailedState') AS status,
        CAST(json_extract_string(payload, '$.teams.home.team.id') AS INTEGER) AS home_team_id,
        json_extract_string(payload, '$.teams.home.team.name') AS home_team_name,
        CAST(json_extract_string(payload, '$.teams.away.team.id') AS INTEGER) AS away_team_id,
        json_extract_string(payload, '$.teams.away.team.name') AS away_team_name,
        CAST(json_extract_string(payload, '$.teams.home.score') AS INTEGER) AS home_score,
        CAST(json_extract_string(payload, '$.teams.away.score') AS INTEGER) AS away_score,
        CAST(json_extract_string(payload, '$.venue.id') AS INTEGER) AS venue_id,
        json_extract_string(payload, '$.venue.name') AS venue_name,
        json_extract_string(payload, '$.dayNight') AS day_night,
        CAST(json_extract_string(payload, '$.teams.home.probablePitcher.id') AS INTEGER)
            AS home_probable_pitcher_id,
        json_extract_string(payload, '$.teams.home.probablePitcher.fullName')
            AS home_probable_pitcher_name,
        CAST(json_extract_string(payload, '$.teams.away.probablePitcher.id') AS INTEGER)
            AS away_probable_pitcher_id,
        json_extract_string(payload, '$.teams.away.probablePitcher.fullName')
            AS away_probable_pitcher_name,
        ingested_at
    FROM bronze_schedule
),
deduped AS (
    -- If the same game appears in multiple bronze partitions (e.g., we re-ingested
    -- after the game finished), keep the most recent version by ingested_at.
    SELECT * FROM parsed
    QUALIFY ROW_NUMBER() OVER (PARTITION BY game_pk ORDER BY ingested_at DESC) = 1
)
SELECT
    game_pk,
    game_date,
    game_datetime_utc,
    season,
    game_type,
    status,
    home_team_id,
    home_team_name,
    away_team_id,
    away_team_name,
    home_score,
    away_score,
    CASE WHEN status = 'Final' THEN home_score + away_score END AS total_runs,
    CASE
        WHEN status = 'Final' AND home_score > away_score THEN TRUE
        WHEN status = 'Final' AND home_score < away_score THEN FALSE
    END AS home_win,
    venue_id,
    venue_name,
    day_night,
    home_probable_pitcher_id,
    home_probable_pitcher_name,
    away_probable_pitcher_id,
    away_probable_pitcher_name,
    ingested_at
FROM deduped
ORDER BY game_date, game_pk;
