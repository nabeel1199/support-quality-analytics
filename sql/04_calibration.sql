-- =============================================================================
-- 04 · Evaluator calibration: are QA evaluators scoring consistently?
-- Every month, all 6 evaluators score the same 10 tickets independently.
-- The "consensus" for each ticket x criterion is the majority view
-- (4+ of 6 evaluators). Exact 3-3 splits are treated as unclear and excluded.
-- =============================================================================

CREATE OR REPLACE TABLE calibration_long AS
WITH u AS (
    UNPIVOT raw_calibration
    ON greeting, verification, accuracy, policy_compliance, empathy, documentation
    INTO NAME criterion VALUE result
)
SELECT u.*, r.is_critical
FROM u JOIN rubric r USING (criterion);

CREATE OR REPLACE TABLE calibration_consensus AS
SELECT session_month, ticket_id, criterion,
       count(*) FILTER (WHERE result = 'fail') AS fail_votes,
       CASE WHEN count(*) FILTER (WHERE result = 'fail') >= 4 THEN 'fail'
            WHEN count(*) FILTER (WHERE result = 'fail') <= 2 THEN 'pass'
       END AS consensus
FROM calibration_long
GROUP BY ALL;

CREATE OR REPLACE TABLE evaluator_calibration AS
SELECT c.evaluator_id,
       CASE WHEN c.is_critical THEN 'Critical criteria' ELSE 'Soft-skill criteria' END AS criteria_type,
       count(*)                                                                         AS judgements,
       round(100.0 * avg((c.result = k.consensus)::INT), 1)                             AS agreement_pct,
       -- Direction of disagreement: failing what others passed vs passing what others failed
       count(*) FILTER (WHERE c.result = 'fail' AND k.consensus = 'pass')               AS stricter_than_consensus,
       count(*) FILTER (WHERE c.result = 'pass' AND k.consensus = 'fail')               AS more_lenient_than_consensus
FROM calibration_long c
JOIN calibration_consensus k USING (session_month, ticket_id, criterion)
WHERE k.consensus IS NOT NULL
GROUP BY ALL
ORDER BY criteria_type, evaluator_id;

-- Does the same bias show up in day-to-day QA scores? Compare each evaluator's
-- average score with the average of everyone else (tickets are assigned randomly).
CREATE OR REPLACE TABLE evaluator_scoring AS
SELECT evaluator_id,
       count(*)                                                            AS evaluations,
       round(avg(qa_score), 1)                                             AS avg_qa_score,
       round(avg(qa_score) - (SELECT avg(qa_score) FROM qa_scored), 1)     AS diff_vs_overall
FROM qa_scored
GROUP BY evaluator_id
ORDER BY evaluator_id;
