-- Gold intermediate: pitcher_form
--
-- Grain: one row per (game_pk, pitcher_id) — the starting pitcher's form
-- going into each of their actual starts. Same anti-leakage discipline as
-- team_rolling: `ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING` excludes the
-- pitcher's current start from their own rolling window.
--
-- Rate stats are aggregated over the window (sum ER / sum IP * 9) rather
-- than averaging per-start rates — that's the standard cumulative-ERA
-- definition and gives the correct weighted rate over the span.
--
-- Depends on registered views: games, pitcher_game_stats.

WITH starter_appearances AS (
    -- One row per actual starter appearance, joined to game_date for ordering.
    SELECT
        p.game_pk,
        g.game_date,
        p.pitcher_id,
        p.team_id,
        p.innings_pitched,
        p.earned_runs,
        p.strikeouts
    FROM pitcher_game_stats p
    JOIN games g ON g.game_pk = p.game_pk
    WHERE p.is_starter = TRUE
)
SELECT
    game_pk,
    pitcher_id,
    -- Cumulative ERA / K/9 over the trailing 5 starts (excluding current).
    -- NULLIF guards divide-by-zero if a pitcher had 5 straight 0.0-IP starts
    -- (extreme edge case, but real — see the 0.0-IP starter bug fix).
    SUM(earned_runs) OVER w * 9.0 / NULLIF(SUM(innings_pitched) OVER w, 0) AS era_l5,
    SUM(strikeouts) OVER w * 9.0 / NULLIF(SUM(innings_pitched) OVER w, 0) AS k_per_9_l5,
    -- Days since previous start. LAG on the ordered starts gives the immediate
    -- prior date, regardless of the ROWS BETWEEN clause on `w`. Using
    -- DATE_DIFF avoids the DATE-DATE vs INTERVAL vs INTEGER wart across
    -- DuckDB versions.
    DATE_DIFF(
        'day',
        LAG(game_date, 1) OVER pitcher_order,
        game_date
    ) AS days_rest,
    COUNT(*) OVER w AS n_prior_starts
FROM starter_appearances
WINDOW
    w AS (
        PARTITION BY pitcher_id
        ORDER BY game_date, game_pk
        ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
    ),
    pitcher_order AS (
        PARTITION BY pitcher_id ORDER BY game_date, game_pk
    )
ORDER BY pitcher_id, game_date, game_pk;
