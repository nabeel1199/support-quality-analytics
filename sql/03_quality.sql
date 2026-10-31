-- =============================================================================
-- 03 · Quality: where do errors happen, who makes them, and do they matter?
-- =============================================================================

CREATE OR REPLACE TABLE qa_monthly AS
SELECT month,
       count(*)                                     AS evaluations,
       round(avg(qa_score), 1)                      AS avg_qa_score,
       round(100.0 * avg(auto_fail::INT), 1)        AS auto_fail_pct
FROM qa_scored
GROUP BY month
ORDER BY month;

-- Error Pareto: which rubric criteria fail most often?
CREATE OR REPLACE TABLE qa_by_criterion AS
SELECT criterion,
       is_critical,
       weight,
       count(*) FILTER (WHERE result = 'fail')                                    AS fails,
       round(100.0 * avg((result = 'fail')::INT), 1)                              AS fail_rate_pct,
       round(100.0 * count(*) FILTER (WHERE result = 'fail')
             / sum(count(*) FILTER (WHERE result = 'fail')) OVER (), 1)           AS share_of_all_fails_pct
FROM qa_long
GROUP BY ALL
ORDER BY fails DESC, criterion;

-- New agents vs experienced agents.
CREATE OR REPLACE TABLE qa_by_tenure AS
SELECT tenure_band,
       count(*)                                     AS evaluations,
       round(avg(qa_score), 1)                      AS avg_qa_score,
       round(100.0 * avg(auto_fail::INT), 1)        AS auto_fail_pct
FROM qa_scored
WHERE tenure_band <> 'Unknown'
GROUP BY tenure_band
ORDER BY tenure_band;

-- Do QA findings show up in what customers experience? Join QA results to ticket outcomes.
CREATE OR REPLACE TABLE qa_vs_outcomes AS
SELECT CASE WHEN q.auto_fail THEN 'Critical error found' ELSE 'No critical error' END AS qa_result,
       count(*)                                                                     AS evaluated_tickets,
       round(100.0 * avg(t.reopened::INT), 1)                                       AS reopen_pct,
       round(100.0 * avg(t.escalated::INT), 1)                                      AS escalation_pct,
       count(t.csat)                                                                AS csat_responses,
       round(100.0 * avg((t.csat >= 4)::INT) FILTER (WHERE t.csat IS NOT NULL), 1)  AS csat_pct
FROM qa_scored q
JOIN tickets t USING (ticket_id)
GROUP BY ALL
ORDER BY qa_result;

-- Per criterion: how much does failing it change the customer outcome?
CREATE OR REPLACE TABLE criterion_impact AS
SELECT l.criterion,
       round(100.0 * avg((t.csat >= 4)::INT) FILTER (WHERE l.result = 'pass' AND t.csat IS NOT NULL), 1) AS csat_pct_when_passed,
       round(100.0 * avg((t.csat >= 4)::INT) FILTER (WHERE l.result = 'fail' AND t.csat IS NOT NULL), 1) AS csat_pct_when_failed,
       round(100.0 * avg(t.reopened::INT)    FILTER (WHERE l.result = 'pass'), 1)                       AS reopen_pct_when_passed,
       round(100.0 * avg(t.reopened::INT)    FILTER (WHERE l.result = 'fail'), 1)                       AS reopen_pct_when_failed,
       count(t.csat) FILTER (WHERE l.result = 'fail')                                                   AS csat_responses_when_failed
FROM qa_long l
JOIN tickets t USING (ticket_id)
GROUP BY l.criterion
ORDER BY csat_pct_when_passed - csat_pct_when_failed DESC, criterion;
