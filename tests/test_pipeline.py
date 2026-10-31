"""End-to-end tests: generate the full synthetic dataset, run every SQL file, check the results."""
import filecmp

import pytest

import generate_data
import pipeline


def test_generator_is_reproducible(raw_dir, tmp_path):
    generate_data.main(tmp_path)
    for name in pipeline.RAW_VIEWS.values():
        assert filecmp.cmp(raw_dir / name, tmp_path / name, shallow=False), name


def test_exports_are_identical_on_every_run(full, raw_dir):
    # Ties in a sort (e.g. two agents with the same QA score) must not reorder rows between runs.
    for _ in range(3):
        again = pipeline.build(raw_dir=raw_dir)
        for table in pipeline.EXPORT_TABLES:
            assert full.table(table).df().equals(again.table(table).df()), table
        again.close()


def test_missing_raw_data_fails_clearly(tmp_path):
    with pytest.raises(SystemExit, match="generate_data.py"):
        pipeline.build(raw_dir=tmp_path)


@pytest.mark.parametrize("table", pipeline.EXPORT_TABLES)
def test_every_export_table_has_rows(full, table):
    assert full.sql(f"SELECT count(*) FROM {table}").fetchone()[0] > 0


def test_injected_dirty_data_is_all_found(full):
    counts = [n for (n,) in full.sql("SELECT rows_affected FROM data_quality ORDER BY ord").fetchall()]
    assert counts == [316, 40, 60, 25, 632]


def test_ticket_ids_are_unique_after_cleaning(full):
    total, distinct = full.sql("SELECT count(*), count(DISTINCT ticket_id) FROM tickets").fetchone()
    assert total == distinct == 63_180


def test_scores_and_percentages_are_in_range(full):
    assert full.sql("SELECT min(qa_score) >= 0 AND max(qa_score) <= 100 FROM qa_scored").fetchone()[0]
    assert full.sql("SELECT bool_and(sla_met_pct BETWEEN 0 AND 100 AND csat_pct BETWEEN 0 AND 100) FROM monthly_ops").fetchone()[0]


def test_headline_numbers_match_the_readme(full):
    kpis = full.sql("SELECT * FROM headline_kpis").df().iloc[0]
    assert kpis.sla_met_pct == 89.3
    assert kpis.csat_pct == 73.4
    assert kpis.avg_qa_score == 92.0
    assert kpis.auto_fail_pct == 15.2
    assert kpis.qa_evaluations == 7_304


def test_scorecard_only_flags_agents_with_enough_evaluations(full):
    assert full.sql("SELECT count(*) FROM agent_scorecard WHERE flag IS NOT NULL AND evaluations < 12").fetchone()[0] == 0
