"""Run every SQL file in order against the raw (synthetic) data and export the results.

Usage:
    python src/pipeline.py

All analysis lives in the numbered files under sql/. This script registers
each CSV in data/raw/ as a view called raw_<name>, executes the SQL files in
order, and exports the result tables to CSV (used by the notebook, the
dashboard and the Tableau guide).
"""
from pathlib import Path
from typing import Iterable, Optional

import duckdb

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SQL_DIR = ROOT / "sql"
EXPORTS = ROOT / "exports"

RAW_VIEWS = {
    "raw_tickets": "tickets.csv",
    "raw_agents": "agents.csv",
    "raw_qa": "qa_evaluations.csv",
    "raw_calibration": "calibration.csv",
}

EXPORT_TABLES = [
    "data_quality",
    "headline_kpis",
    "monthly_ops",
    "channel_ops",
    "category_ops",
    "sla_by_weekday",
    "qa_monthly",
    "qa_by_criterion",
    "qa_by_tenure",
    "qa_vs_outcomes",
    "criterion_impact",
    "evaluator_calibration",
    "evaluator_scoring",
    "coaching_impact",
    "coaching_did",
    "coaching_trend",
    "agent_scorecard",
]


def load_raw(con: duckdb.DuckDBPyConnection, raw_dir: Path = RAW) -> None:
    """Register each raw CSV as a view called raw_<name>."""
    for view, file in RAW_VIEWS.items():
        con.execute(f"CREATE OR REPLACE VIEW {view} AS SELECT * FROM read_csv_auto('{(raw_dir / file).as_posix()}')")


def run_sql(con: duckdb.DuckDBPyConnection, sql_files: Optional[Iterable[Path]] = None) -> None:
    """Execute the given SQL files (default: every file in sql/, in numbered order)."""
    for sql_file in sql_files or sorted(SQL_DIR.glob("*.sql")):
        con.execute(sql_file.read_text())


def build(con: Optional[duckdb.DuckDBPyConnection] = None, raw_dir: Path = RAW) -> duckdb.DuckDBPyConnection:
    if not (raw_dir / "tickets.csv").exists():
        raise SystemExit("Raw data missing - run `python scripts/generate_data.py` first.")
    con = con or duckdb.connect()
    # Parallel aggregation adds floats in a different order on each run, which can flip a rounded
    # average (11.85 -> 11.8 or 11.9). One thread keeps every export identical; the data is small.
    con.execute("SET threads TO 1")
    load_raw(con, raw_dir)
    run_sql(con)
    return con


def export(con: duckdb.DuckDBPyConnection) -> None:
    EXPORTS.mkdir(exist_ok=True)
    for table in EXPORT_TABLES:
        con.table(table).df().to_csv(EXPORTS / f"{table}.csv", index=False)


if __name__ == "__main__":
    connection = build()
    export(connection)
    print(f"Exported {len(EXPORT_TABLES)} tables to {EXPORTS}")
    print(connection.table("data_quality").df().to_string(index=False))
