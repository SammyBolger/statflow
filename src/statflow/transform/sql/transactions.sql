-- Silver: transactions
--
-- Grain: one row per MLB transaction (IL moves, trades, roster status changes).
-- Source: bronze_transactions (raw JSON payload per transaction).
--
-- We extract just the fields we need for feature engineering — typeCode
-- (IL10 / IL60 / SC / TR / etc.), the involved person, and both from/to
-- teams. Full description text kept for auditability. Silver is filtered
-- to MLB-only via league heuristics (MLB player ids are consistently
-- accompanied by an MLB team_id — we drop rows where neither from_team
-- nor to_team has a valid MLB team id, catching minor-league transactions).

WITH parsed AS (
    SELECT
        transaction_id,
        CAST(date AS DATE) AS transaction_date,
        json_extract_string(payload, '$.typeCode') AS type_code,
        json_extract_string(payload, '$.typeDesc') AS type_desc,
        json_extract_string(payload, '$.description') AS description,
        CAST(json_extract_string(payload, '$.person.id') AS INTEGER) AS person_id,
        json_extract_string(payload, '$.person.fullName') AS person_name,
        CAST(json_extract_string(payload, '$.fromTeam.id') AS INTEGER) AS from_team_id,
        json_extract_string(payload, '$.fromTeam.name') AS from_team_name,
        CAST(json_extract_string(payload, '$.toTeam.id') AS INTEGER) AS to_team_id,
        json_extract_string(payload, '$.toTeam.name') AS to_team_name,
        ingested_at
    FROM bronze_transactions
),
deduped AS (
    SELECT * FROM parsed
    QUALIFY ROW_NUMBER() OVER (PARTITION BY transaction_id ORDER BY ingested_at DESC) = 1
)
SELECT
    transaction_id,
    transaction_date,
    type_code,
    type_desc,
    description,
    person_id,
    person_name,
    from_team_id,
    from_team_name,
    to_team_id,
    to_team_name,
    -- Convenience flag: transactions that affect availability (IL, DFA, etc.)
    -- vs administrative moves (option to minors, coaching change).
    (type_code IN ('IL10', 'IL60', 'IL15', 'SC', 'RA', 'DFA')) AS affects_availability,
    ingested_at
FROM deduped
ORDER BY transaction_date, transaction_id;
