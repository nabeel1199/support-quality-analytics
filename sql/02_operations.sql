-- =============================================================================
-- 02 · Operational KPIs: volume, speed, SLA, escalations, reopens, CSAT
-- CSAT % = share of survey answers that are 4 or 5 (the usual "% satisfied").
-- =============================================================================

CREATE OR REPLACE TABLE headline_kpis AS
SELECT count(*)                                                         AS tickets,
       count(DISTINCT agent_id) FILTER (WHERE team <> 'Unknown')       AS agents,
       round(100.0 * avg(sla_met::INT), 1)                              AS sla_met_pct,
       round(median(resolution_hours), 1)                               AS median_resolution_hours,
       round(avg(handle_time_min), 1)                                   AS avg_handle_min,
       round(100.0 * avg(escalated::INT), 1)                            AS escalation_pct,
       round(100.0 * avg(reopened::INT), 1)                             AS reopen_pct,
       round(100.0 * count(csat) / count(*), 1)                         AS survey_response_pct,
       round(100.0 * avg((csat >= 4)::INT) FILTER (WHERE csat IS NOT NULL), 1) AS csat_pct,
       (SELECT round(avg(qa_score), 1) FROM qa_scored)                  AS avg_qa_score,
       (SELECT round(100.0 * avg(auto_fail::INT), 1) FROM qa_scored)    AS auto_fail_pct,
       (SELECT count(*) FROM qa_scored)                                 AS qa_evaluations
FROM tickets;

CREATE OR REPLACE TABLE monthly_ops AS
SELECT month,
       count(*)                                                         AS tickets,
       round(100.0 * avg(sla_met::INT), 1)                              AS sla_met_pct,
       round(median(resolution_hours), 1)                               AS median_resolution_hours,
       round(avg(handle_time_min), 1)                                   AS avg_handle_min,
       round(100.0 * avg(escalated::INT), 1)                            AS escalation_pct,
       round(100.0 * avg(reopened::INT), 1)                             AS reopen_pct,
       round(100.0 * avg((csat >= 4)::INT) FILTER (WHERE csat IS NOT NULL), 1) AS csat_pct,
       count(csat)                                                      AS csat_responses
FROM tickets
GROUP BY month
ORDER BY month;

CREATE OR REPLACE TABLE channel_ops AS
SELECT channel,
       count(*)                                                         AS tickets,
       round(median(first_response_min), 1)                             AS median_first_response_min,
       round(quantile_cont(first_response_min, 0.9), 1)                 AS p90_first_response_min,
       round(100.0 * avg(sla_met::INT), 1)                              AS sla_met_pct,
       round(100.0 * avg((csat >= 4)::INT) FILTER (WHERE csat IS NOT NULL), 1) AS csat_pct
FROM tickets
GROUP BY channel
ORDER BY tickets DESC;

CREATE OR REPLACE TABLE category_ops AS
SELECT category,
       count(*)                                                         AS tickets,
       round(avg(handle_time_min), 1)                                   AS avg_handle_min,
       round(100.0 * avg(escalated::INT), 1)                            AS escalation_pct,
       round(100.0 * avg(reopened::INT), 1)                             AS reopen_pct,
       round(100.0 * avg(sla_met::INT), 1)                              AS sla_met_pct,
       round(100.0 * avg((csat >= 4)::INT) FILTER (WHERE csat IS NOT NULL), 1) AS csat_pct
FROM tickets
GROUP BY category
ORDER BY tickets DESC;

-- Does the day a ticket arrives affect whether it meets SLA? (weekend staffing / Monday backlog)
CREATE OR REPLACE TABLE sla_by_weekday AS
SELECT isodow(created_at)                                               AS weekday_num,
       dayname(created_at)                                              AS weekday,
       priority,
       count(*)                                                         AS tickets,
       round(100.0 * avg(sla_met::INT), 1)                              AS sla_met_pct
FROM tickets
GROUP BY ALL
ORDER BY priority, weekday_num;
