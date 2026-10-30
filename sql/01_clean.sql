-- =============================================================================
-- 01 · Data quality checks and cleaning
-- raw_* views are created in src/pipeline.py from the CSVs in data/raw/.
-- =============================================================================

CREATE OR REPLACE TABLE data_quality AS
SELECT 1 AS ord, 'Duplicate rows (same ticket ID loaded twice)' AS check_name,
       count(*) - count(DISTINCT ticket_id) AS rows_affected,
       'Removed (one row kept per ticket)' AS action
FROM raw_tickets
UNION ALL
SELECT 2, 'Resolved before it was created (bad timestamp)',
       count(DISTINCT ticket_id) FILTER (WHERE resolved_at < created_at), 'Kept for volume, excluded from time metrics'
FROM raw_tickets
UNION ALL
SELECT 3, 'CSAT outside the 1–5 scale',
       count(DISTINCT ticket_id) FILTER (WHERE csat IS NOT NULL AND csat NOT BETWEEN 1 AND 5), 'Set to NULL (no valid survey)'
FROM raw_tickets
UNION ALL
SELECT 4, 'Agent ID not found in the agent list',
       count(DISTINCT ticket_id) FILTER (WHERE agent_id NOT IN (SELECT agent_id FROM raw_agents)), 'Kept, team = Unknown'
FROM raw_tickets
UNION ALL
SELECT 5, 'Channel not recorded',
       count(DISTINCT ticket_id) FILTER (WHERE channel IS NULL OR trim(channel) = ''), 'Labelled "unknown"'
FROM raw_tickets
ORDER BY ord;

CREATE OR REPLACE TABLE tickets AS
WITH dedup AS (
    SELECT * FROM raw_tickets
    QUALIFY row_number() OVER (PARTITION BY ticket_id ORDER BY created_at) = 1
),
base AS (
    SELECT t.ticket_id,
           t.created_at,
           date_trunc('month', t.created_at)::DATE                     AS month,
           coalesce(nullif(trim(t.channel), ''), 'unknown')             AS channel,
           t.category,
           t.priority,
           t.agent_id,
           coalesce(a.team, 'Unknown')                                  AS team,
           datediff('day', a.hire_date, t.created_at::DATE)             AS tenure_days,
           t.handle_time_min,
           t.escalated,
           t.reopened,
           CASE WHEN t.csat BETWEEN 1 AND 5 THEN t.csat END             AS csat,
           -- Bad timestamps give NULL durations, so they drop out of averages and SLA.
           CASE WHEN t.first_response_at >= t.created_at
                THEN datediff('second', t.created_at, t.first_response_at) / 60.0 END   AS first_response_min,
           CASE WHEN t.resolved_at >= t.created_at
                THEN datediff('second', t.created_at, t.resolved_at) / 3600.0 END       AS resolution_hours,
           CASE t.priority WHEN 'P1' THEN 4 WHEN 'P2' THEN 24 ELSE 72 END   AS sla_target_hours
    FROM dedup t
    LEFT JOIN raw_agents a USING (agent_id)
)
SELECT *,
       CASE WHEN resolution_hours IS NOT NULL THEN resolution_hours <= sla_target_hours END AS sla_met,
       CASE WHEN tenure_days IS NULL THEN 'Unknown'
            WHEN tenure_days < 90  THEN '0-3 months'
            WHEN tenure_days < 180 THEN '3-6 months'
            ELSE '6+ months' END                                                       AS tenure_band
FROM base;

-- QA evaluations scored with the rubric weights.
-- Score = % of rubric points passed. A failed critical criterion (verification,
-- accuracy, policy compliance) is also flagged as an "auto-fail".
CREATE OR REPLACE TABLE rubric AS
SELECT * FROM (VALUES
    ('greeting',          10, false),
    ('verification',      20, true),
    ('accuracy',          25, true),
    ('policy_compliance', 20, true),
    ('empathy',           10, false),
    ('documentation',     15, false)
) AS r(criterion, weight, is_critical);

-- Long format: one row per evaluation x criterion (easier to aggregate).
CREATE OR REPLACE TABLE qa_long AS
WITH u AS (
    UNPIVOT raw_qa
    ON greeting, verification, accuracy, policy_compliance, empathy, documentation
    INTO NAME criterion VALUE result
)
SELECT u.evaluation_id, u.ticket_id, u.agent_id, u.evaluator_id, u.evaluated_at,
       u.criterion, u.result, r.weight, r.is_critical
FROM u
JOIN rubric r USING (criterion);

CREATE OR REPLACE TABLE qa_scored AS
SELECT l.evaluation_id,
       l.ticket_id,
       l.agent_id,
       l.evaluator_id,
       t.team,
       t.tenure_band,
       t.month,
       100.0 * sum(l.weight) FILTER (WHERE l.result = 'pass') / sum(l.weight)    AS qa_score,
       bool_or(l.result = 'fail' AND l.is_critical)                              AS auto_fail,
       count(*) FILTER (WHERE l.result = 'fail')                                  AS criteria_failed
FROM qa_long l
JOIN tickets t USING (ticket_id)
GROUP BY ALL;
