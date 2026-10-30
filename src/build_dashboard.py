"""Build the interactive dashboard (docs/index.html, served by GitHub Pages).

Usage:
    python src/build_dashboard.py      # reads the CSVs in exports/

The page is a single static HTML file: the data is embedded as JSON and the
charts are drawn in the browser with Plotly.js, so there is no server.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPORTS = ROOT / "exports"
TEMPLATE = Path(__file__).with_name("dashboard_template.html")
OUT = ROOT / "docs" / "index.html"


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(EXPORTS / f"{name}.csv")


def records(df: pd.DataFrame) -> list:
    return json.loads(df.to_json(orient="records"))


def build_payload() -> dict:
    weekday = load("sla_by_weekday")
    weekday["met"] = weekday["tickets"] * weekday["sla_met_pct"] / 100
    weekday = (weekday.groupby(["weekday_num", "weekday"], as_index=False)[["tickets", "met"]].sum()
               .assign(sla_met_pct=lambda d: (100 * d["met"] / d["tickets"]).round(1))
               .drop(columns="met"))
    scorecard = load("agent_scorecard")
    ranked = scorecard[scorecard["qa_rank"].notna()]
    return {
        "kpis": records(load("headline_kpis"))[0],
        "monthly": records(load("monthly_ops")),
        "weekday": records(weekday),
        "criteria": records(load("qa_by_criterion")),
        "tenure": records(load("qa_by_tenure")),
        "outcomes": records(load("qa_vs_outcomes")),
        "impact": records(load("criterion_impact")),
        "calibration": records(load("evaluator_calibration")),
        "evaluators": records(load("evaluator_scoring")),
        "coaching": records(load("coaching_impact")),
        "did": records(load("coaching_did"))[0],
        "coaching_trend": records(load("coaching_trend")),
        "categories": records(load("category_ops")),
        "top_agents": records(ranked.head(5)),
        "flagged_agents": records(scorecard[scorecard["flag"].notna()]),
        "quality": records(load("data_quality")),
    }


if __name__ == "__main__":
    html = TEMPLATE.read_text().replace("__DATA__", json.dumps(build_payload(), separators=(",", ":")))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(html)
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")
