# Rebuilding the dashboard in Tableau Public

Tableau Public is free and runs on macOS. Run `python src/pipeline.py` first so that `exports/` is up to date.

## Option A: aggregated tables (quickest)

Add each CSV as its own data source:

| Sheet | File | Build |
|---|---|---|
| KPI tiles | `headline_kpis.csv` | One sheet per measure on **Text** (`sla_met_pct`, `csat_pct`, `avg_qa_score`, `auto_fail_pct`, `reopen_pct`). |
| SLA by month | `monthly_ops.csv` | `month` (continuous) on Columns and `sla_met_pct` on Rows, as a line. Put `tickets` on Tooltip. |
| Error Pareto | `qa_by_criterion.csv` | `criterion` on Rows and `share_of_all_fails_pct` on Columns, sorted. Put `is_critical` on Color. |
| Tenure | `qa_by_tenure.csv` | `tenure_band` on Columns and `auto_fail_pct` on Rows. |
| QA vs outcomes | `qa_vs_outcomes.csv` | Measure Names (`csat_pct`, `reopen_pct`, `escalation_pct`) on Columns and Measure Values on Rows. Put `qa_result` on Color. |
| Calibration | `evaluator_calibration.csv` | Filter to soft-skill criteria. `evaluator_id` on Columns and `agreement_pct` on Rows. Add a constant reference line at 95. |
| Coaching | `coaching_trend.csv` | `quarter` on Columns, `auto_fail_pct` on Rows and `grp` on Color. |
| Scorecard | `agent_scorecard.csv` | Text table, filtered to `flag` = "Coaching priority" plus Top 5 by `qa_rank`. |

## Option B: row-level data (more interactive)

Connect to `data/raw/tickets.csv`, `agents.csv` and `qa_evaluations.csv` (after running `scripts/generate_data.py`). In the data model, relate tickets ↔ agents on `agent_id` and tickets ↔ QA on `ticket_id`. Then add:

- Calculated fields: `SLA Met = DATEDIFF('hour', [created_at], [resolved_at]) <= [SLA target]`, with `SLA target` = `CASE [priority] WHEN 'P1' THEN 4 WHEN 'P2' THEN 24 ELSE 72 END`.
- `CSAT % = SUM(IIF([csat] >= 4, 1, 0)) / COUNT([csat])`.
- Filters for team, channel, category and month, applied to all sheets on the dashboard.

Option B is a good way to show Tableau skills (LODs, parameters, actions). For example, `{FIXED [agent_id] : AVG([qa_score])}` compares each agent with their team average.

## Publish

**File → Save to Tableau Public**. In the description, note that the data is synthetic, then link the workbook from the README and your CV.

> Power BI Desktop is Windows-only. On a Mac, use Tableau Public, or build the same model in Power BI on a Windows machine or VM.
