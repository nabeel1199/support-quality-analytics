"""Static PNG charts for the README (reports/figures/).

Usage:
    python src/charts.py        # reads the CSVs in exports/

The interactive versions live in the dashboard (src/build_dashboard.py).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mtick  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXPORTS = ROOT / "exports"
FIGURES = ROOT / "reports" / "figures"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED,
    "axes.titlecolor": INK,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
})


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(EXPORTS / f"{name}.csv")


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES / name, bbox_inches="tight")
    plt.close(fig)


def error_pareto() -> None:
    df = load("qa_by_criterion").sort_values("share_of_all_fails_pct")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    colors = [ORANGE if c else BLUE for c in df["is_critical"]]
    labels = df["criterion"].str.replace("_", " ").str.capitalize()
    ax.barh(labels, df["share_of_all_fails_pct"], color=colors, height=0.6)
    for y, (share, rate) in enumerate(zip(df["share_of_all_fails_pct"], df["fail_rate_pct"])):
        ax.text(share + 0.4, y, f"{share:.1f}%  (fails in {rate:.1f}% of evals)", va="center", fontsize=8, color=MUTED)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.set_xlim(0, df["share_of_all_fails_pct"].max() * 1.6)
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE)],
              labels=["Soft skill", "Critical"], frameon=False, loc="lower right")
    ax.set_title("What fails in QA (share of all failed criteria)")
    save(fig, "qa_error_pareto.png")


def qa_vs_outcomes() -> None:
    df = load("qa_vs_outcomes").set_index("qa_result")
    metrics = [("CSAT %", "csat_pct"), ("Reopen %", "reopen_pct"), ("Escalation %", "escalation_pct")]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    w = 0.38
    for i, (name, color) in enumerate([("No critical error", BLUE), ("Critical error found", ORANGE)]):
        vals = [df.loc[name, m] for _, m in metrics]
        xs = [j + (i - 0.5) * w for j in range(len(metrics))]
        ax.bar(xs, vals, width=w, color=color, label=name)
        for x, v in zip(xs, vals):
            ax.text(x, v + 1, f"{v:.1f}%", ha="center", fontsize=8, color=MUTED)
    ax.set_xticks(range(len(metrics)), labels=[m for m, _ in metrics])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False)
    ax.set_title("Do QA errors reach the customer?")
    save(fig, "qa_vs_outcomes.png")


def calibration() -> None:
    df = load("evaluator_calibration").query("criteria_type == 'Soft-skill criteria'")
    fig, ax = plt.subplots(figsize=(7, 3.4))
    colors = [ORANGE if v < 95 else BLUE for v in df["agreement_pct"]]
    ax.bar(df["evaluator_id"], df["agreement_pct"], color=colors, width=0.55)
    ax.axhline(95, color=MUTED, linestyle=":", linewidth=1)
    ax.set_xlim(-0.6, len(df) - 0.4)
    ax.text(len(df) - 0.35, 95, "95%\ntarget", ha="left", va="center", fontsize=8, color=MUTED, clip_on=False)
    for x, v in enumerate(df["agreement_pct"]):
        ax.text(x, v - 2.5, f"{v:.1f}%", ha="center", fontsize=8, color="white", fontweight="bold")
    ax.set_ylim(70, 103)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.grid(axis="x", visible=False)
    ax.set_title("Evaluator agreement with the majority (soft-skill criteria)")
    save(fig, "evaluator_calibration.png")


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    error_pareto()
    qa_vs_outcomes()
    calibration()
    print(f"Saved figures to {FIGURES}")
