-- =============================================================================
-- LLM evaluator: score every rater against the hidden truth and the human majority.
-- Raters are the six human QA evaluators (monthly calibration sessions) plus each
-- LLM judge from src/llm_evaluator.py. Run by `llm_evaluator.py analyse` after the
-- main pipeline (it needs rubric, calibration_long and calibration_consensus). Inputs:
--   raw_calibration_truth  hidden truth for each calibration ticket (from the generator)
--   llm_judgments          one row per ticket x criterion x LLM judge, with its evidence
--   llm_calls              token use and cost of every API call
-- =============================================================================

CREATE OR REPLACE TABLE calibration_truth_long AS
SELECT u.ticket_id, u.criterion, u.truth, r.is_critical
FROM (
    UNPIVOT raw_calibration_truth
    ON greeting, verification, accuracy, policy_compliance, empathy, documentation
    INTO NAME criterion VALUE truth
) u
JOIN rubric r USING (criterion);

-- Every judgement with the truth and the human majority beside it. As in 04_calibration.sql,
-- each human is compared with a majority that includes their own vote; the LLM is not part of it.
CREATE OR REPLACE TABLE rater_judgements AS
WITH j AS (
    SELECT evaluator_id AS rater, 'Human' AS rater_type, ticket_id, criterion, result FROM calibration_long
    UNION ALL
    SELECT rater, 'LLM', ticket_id, criterion, result FROM llm_judgments
)
SELECT j.rater, j.rater_type, j.ticket_id, j.criterion, t.is_critical, j.result, t.truth, k.consensus
FROM j
JOIN calibration_truth_long t USING (ticket_id, criterion)
LEFT JOIN calibration_consensus k USING (ticket_id, criterion);

-- Errors are rare (64 of 720 judgements), so plain agreement flatters everyone: a rater who
-- passes everything still agrees 91% of the time. Errors caught and kappa are the fairer measures.
CREATE OR REPLACE TABLE llm_rater_agreement AS
WITH s AS (
    SELECT *, CASE WHEN is_critical THEN 'Critical criteria' ELSE 'Soft-skill criteria' END AS ctype
    FROM rater_judgements
), g AS (
    SELECT rater, rater_type,
           coalesce(ctype, 'All criteria')                                         AS criteria_type,
           count(*)                                                                AS judgements,
           count(*) FILTER (WHERE truth = 'fail')                                  AS true_fails,
           count(*) FILTER (WHERE result = 'fail' AND truth = 'fail')              AS caught,
           count(*) FILTER (WHERE result = 'fail' AND truth = 'pass')              AS false_flags,
           avg((result = truth)::DOUBLE)                                           AS po,
           avg((result = 'fail')::DOUBLE)                                          AS p_rater,
           avg((truth = 'fail')::DOUBLE)                                           AS p_truth,
           avg((result = consensus)::DOUBLE) FILTER (WHERE consensus IS NOT NULL)  AS agree_consensus
    FROM s
    GROUP BY GROUPING SETS ((rater, rater_type, ctype), (rater, rater_type))
), k AS (
    SELECT *, p_rater * p_truth + (1 - p_rater) * (1 - p_truth) AS pe FROM g   -- chance agreement
)
SELECT rater, rater_type, criteria_type, judgements, true_fails, caught, false_flags,
       round(100 * po, 1)                                                  AS agree_truth_pct,
       round(100.0 * caught / nullif(true_fails, 0), 1)                    AS errors_caught_pct,
       round(100.0 * false_flags / nullif(judgements - true_fails, 0), 1)  AS false_flag_pct,
       round((po - pe) / nullif(1 - pe, 0), 3)                             AS kappa_vs_truth,
       round(100 * agree_consensus, 1)                                     AS agree_consensus_pct
FROM k
ORDER BY criteria_type, rater_type, rater;

CREATE OR REPLACE TABLE llm_rater_by_criterion AS
SELECT rater, rater_type, criterion,
       count(*) FILTER (WHERE truth = 'fail')                                       AS true_fails,
       count(*) FILTER (WHERE result = 'fail' AND truth = 'fail')                   AS caught,
       count(*) FILTER (WHERE result = 'fail' AND truth = 'pass')                   AS false_flags,
       round(100.0 * count(*) FILTER (WHERE result = 'fail' AND truth = 'fail')
             / nullif(count(*) FILTER (WHERE truth = 'fail'), 0), 1)                AS errors_caught_pct,
       round(100 * avg((result = truth)::DOUBLE), 1)                                 AS agree_truth_pct
FROM rater_judgements
GROUP BY ALL
ORDER BY criterion, rater_type, rater;

-- Every LLM judgement that differs from the truth, with the judge's own evidence: the list to
-- read by hand. A disagreement can be the judge's mistake or a transcript that missed its spec.
CREATE OR REPLACE TABLE llm_disagreements AS
SELECT j.rater, j.ticket_id, j.criterion, r.truth, r.consensus AS human_majority, j.result AS llm_result,
       CASE WHEN j.result = 'fail' THEN 'false flag' ELSE 'missed error' END AS kind,
       j.evidence
FROM llm_judgments j
JOIN rater_judgements r ON r.rater = j.rater AND r.ticket_id = j.ticket_id AND r.criterion = j.criterion
WHERE j.result <> r.truth
ORDER BY j.rater, j.ticket_id, j.criterion;

CREATE OR REPLACE TABLE llm_run_cost AS
SELECT stage, model,
       count(*)                                                                    AS calls,
       sum(input_tokens + cache_read_input_tokens + cache_creation_input_tokens)   AS input_tokens,
       sum(cache_read_input_tokens)                                                AS cached_input_tokens,
       sum(output_tokens)                                                          AS output_tokens,
       round(sum(cost_usd), 2)                                                     AS cost_usd,
       round(sum(cost_usd) / count(*), 4)                                          AS cost_per_call_usd
FROM llm_calls
GROUP BY ALL
ORDER BY stage DESC, model;
