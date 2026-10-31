"""Unit tests for sql/01_clean.sql, run on the hand-made data in tests/fixtures/.

Each fixture ticket is built to exercise one rule:
    T1  normal P1 ticket, resolved in 3h      -> meets the 4h SLA
    T2  loaded twice, resolved in 6h          -> de-duplicated, misses the 4h SLA
    T3  resolved before it was created        -> no resolution time, no SLA result
    T4  blank channel, unknown agent, CSAT 9  -> 'unknown', team 'Unknown', CSAT NULL
    T5  normal P3 ticket, resolved in 10h     -> meets the 72h SLA
"""
import pandas as pd


def rows(con, sql):
    return con.sql(sql).fetchall()


def ticket(con, ticket_id):
    df = con.sql(f"SELECT * FROM tickets WHERE ticket_id = '{ticket_id}'").df()
    assert len(df) == 1
    return df.iloc[0]


def test_data_quality_counts_one_of_each_problem(cleaned):
    counts = rows(cleaned, "SELECT check_name, rows_affected FROM data_quality ORDER BY ord")
    assert [n for _, n in counts] == [1, 1, 1, 1, 1]


def test_duplicates_are_removed(cleaned):
    assert rows(cleaned, "SELECT count(*), count(DISTINCT ticket_id) FROM tickets") == [(5, 5)]


def test_sla_uses_the_priority_target(cleaned):
    assert ticket(cleaned, "T1").sla_met          # P1, 3h <= 4h
    assert not ticket(cleaned, "T2").sla_met      # P1, 6h > 4h
    assert ticket(cleaned, "T5").sla_met          # P3, 10h <= 72h
    assert ticket(cleaned, "T2").resolution_hours == 6


def test_bad_timestamp_is_kept_but_has_no_time_metrics(cleaned):
    t3 = ticket(cleaned, "T3")
    assert pd.isna(t3.resolution_hours)
    assert pd.isna(t3.sla_met)


def test_invalid_values_are_labelled_not_dropped(cleaned):
    t4 = ticket(cleaned, "T4")
    assert t4.channel == "unknown"
    assert t4.team == "Unknown"
    assert t4.tenure_band == "Unknown"
    assert pd.isna(t4.csat)   # CSAT 9 is outside 1-5, so NULL


def test_tenure_bands(cleaned):
    assert ticket(cleaned, "T1").tenure_band == "0-3 months"
    assert ticket(cleaned, "T3").tenure_band == "6+ months"


def test_qa_score_uses_rubric_weights(cleaned):
    scored = dict(rows(cleaned, "SELECT evaluation_id, qa_score FROM qa_scored"))
    assert scored == {"E1": 100.0, "E2": 75.0, "E3": 90.0}


def test_only_critical_failures_are_auto_fails(cleaned):
    auto_fail = dict(rows(cleaned, "SELECT evaluation_id, auto_fail FROM qa_scored"))
    assert auto_fail == {"E1": False, "E2": True, "E3": False}   # accuracy is critical, empathy is not
