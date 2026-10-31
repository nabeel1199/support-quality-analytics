import sys
from pathlib import Path

import duckdb
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import generate_data  # noqa: E402
import pipeline  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def cleaned():
    """Small hand-made dataset with one example of every data problem, after 01_clean.sql."""
    con = duckdb.connect()
    pipeline.load_raw(con, FIXTURES)
    pipeline.run_sql(con, [pipeline.SQL_DIR / "01_clean.sql"])
    yield con
    con.close()


@pytest.fixture(scope="session")
def raw_dir(tmp_path_factory):
    """The full synthetic dataset, generated once per test session."""
    out = tmp_path_factory.mktemp("raw")
    generate_data.main(out)
    return out


@pytest.fixture(scope="session")
def full(raw_dir):
    """The whole pipeline (every SQL file) run on the full synthetic dataset."""
    con = pipeline.build(raw_dir=raw_dir)
    yield con
    con.close()
