-- =============================================================================
-- 05 · Did the Team C coaching programme work? And which agents need support?
-- =============================================================================

-- Coaching started on 1 Apr 2026 for Team C only. A simple before/after on Team C
-- would also pick up anything that changed for everyone (seasonality, new
-- policies), so we compare Team C's change with the change for all other teams
-- (difference-in-differences). Only agents hired before Oct 2025 are included,
-- so the new-hire waves do not distort the comparison.
CREATE OR REPLACE TABLE coaching_impact AS
WITH scoped AS (
    SELECT q.*,
           CASE WHEN q.team = 'Team C' THEN 'Team C (coached)' ELSE 'Other teams' END AS grp,
           CASE WHEN q.month < DATE '2026-04-01' THEN 'before' ELSE 'after' END      AS period,
           t.reopened
    FROM qa_scored q
    JOIN tickets t USING (ticket_id)
    JOIN raw_agents a ON a.agent_id = q.agent_id
    WHERE a.hire_date < DATE '2025-10-01'
),
agg AS (
    SELECT grp,
           avg(qa_score)        FILTER (WHERE period = 'before') AS score_before,
           avg(qa_score)        FILTER (WHERE period = 'after')  AS score_after,
           avg(auto_fail::INT)  FILTER (WHERE period = 'before') AS autofail_before,
           avg(auto_fail::INT)  FILTER (WHERE period = 'after')  AS autofail_after,
           count(*)                                              AS evaluations
    FROM scoped
    GROUP BY grp
)
SELECT grp,
       evaluations,
       round(score_before, 1)                     AS qa_score_before,
       round(score_after, 1)                      AS qa_score_after,
       round(score_after - score_before, 1)       AS qa_score_change,
       round(100 * autofail_before, 1)            AS auto_fail_pct_before,
       round(100 * autofail_after, 1)             AS auto_fail_pct_after,
       round(100 * (autofail_after - autofail_before), 1) AS auto_fail_change_pts
FROM agg
ORDER BY grp DESC;

CREATE OR REPLACE TABLE coaching_did AS
SELECT round(max(qa_score_change)     FILTER (WHERE grp = 'Team C (coached)')
           - max(qa_score_change)     FILTER (WHERE grp = 'Other teams'), 1) AS qa_score_effect_pts,
       round(max(auto_fail_change_pts) FILTER (WHERE grp = 'Team C (coached)')
           - max(auto_fail_change_pts) FILTER (WHERE grp = 'Other teams'), 1) AS auto_fail_effect_pts
FROM coaching_impact;

-- Quarterly trend (monthly samples per team are small and noisy).
CREATE OR REPLACE TABLE coaching_trend AS
SELECT date_trunc('quarter', q.month)::DATE AS quarter,
       CASE WHEN q.team = 'Team C' THEN 'Team C (coached)' ELSE 'Other teams' END AS grp,
       round(100.0 * avg(q.auto_fail::INT), 1) AS auto_fail_pct,
       round(avg(q.qa_score), 1)               AS avg_qa_score
FROM qa_scored q
JOIN raw_agents a ON a.agent_id = q.agent_id
WHERE a.hire_date < DATE '2025-10-01'
GROUP BY ALL
ORDER BY grp, quarter;

-- Agent scorecard for the last quarter (Jul–Sep 2026).
-- Agents with fewer than 12 evaluations are not ranked: too few to be fair.
CREATE OR REPLACE TABLE agent_scorecard AS
WITH ops AS (
    SELECT agent_id, team, max(tenure_band) AS tenure_band,
           count(*)                                                          AS tickets,
           round(avg(handle_time_min), 1)                                    AS avg_handle_min,
           round(100.0 * avg(sla_met::INT), 1)                               AS sla_met_pct,
           round(100.0 * avg(reopened::INT), 1)                              AS reopen_pct,
           round(100.0 * avg((csat >= 4)::INT) FILTER (WHERE csat IS NOT NULL), 1) AS csat_pct,
           count(csat)                                                       AS csat_responses
    FROM tickets
    WHERE month >= DATE '2026-07-01' AND team <> 'Unknown'
    GROUP BY agent_id, team
),
qa AS (
    SELECT agent_id,
           count(*)                               AS evaluations,
           round(avg(qa_score), 1)                AS avg_qa_score,
           round(100.0 * avg(auto_fail::INT), 1)  AS auto_fail_pct
    FROM qa_scored
    WHERE month >= DATE '2026-07-01'
    GROUP BY agent_id
)
SELECT o.*, q.evaluations, q.avg_qa_score, q.auto_fail_pct,
       CASE WHEN q.evaluations >= 12
            THEN rank() OVER (PARTITION BY q.evaluations >= 12 ORDER BY q.avg_qa_score DESC) END AS qa_rank,
       CASE WHEN q.evaluations >= 12
             AND q.avg_qa_score < (SELECT quantile_cont(avg_qa_score, 0.1) FROM qa WHERE evaluations >= 12)
            THEN 'Coaching priority' END                                                     AS flag
FROM ops o
LEFT JOIN qa q USING (agent_id)
ORDER BY qa_rank NULLS LAST;
