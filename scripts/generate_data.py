"""Generate the SYNTHETIC customer-support dataset used in this project.

Usage:
    python scripts/generate_data.py          # writes data/raw/*.csv (seeded, reproducible)

Why synthetic: real contact-centre data (tickets, agent scores, customer
messages) is confidential. Every row here is invented by this script. It is
not taken from, or modelled on, any real company's data. The script builds in
realistic relationships (new agents make more mistakes, errors cause reopens
and bad CSAT, one evaluator is harsher than the rest, one team gets coaching)
so that the analysis has something real to find, and the README explains how.

It also injects a small amount of dirty data (duplicates, impossible
timestamps, invalid CSAT values, unknown agent IDs, blank channels) so the SQL
cleaning step has real work to do.

Outputs (data/raw/):
    agents.csv            one row per agent
    tickets.csv           one row per support ticket (~60k)
    qa_evaluations.csv    one row per QA evaluation of a sampled ticket
    calibration.csv       monthly calibration: the same tickets scored by every evaluator
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
START, END = pd.Timestamp("2025-10-01"), pd.Timestamp("2026-09-30")

CHANNELS = {"chat": 0.45, "email": 0.30, "phone": 0.20, "social": 0.05}
CATEGORIES = {  # share, median handle minutes, escalation probability
    "Order status":     (0.22, 6, 0.03),
    "Billing question": (0.20, 9, 0.04),
    "Account access":   (0.18, 11, 0.05),
    "Refund request":   (0.13, 12, 0.06),
    "Technical issue":  (0.12, 18, 0.15),
    "Product question": (0.08, 8, 0.02),
    "Billing dispute":  (0.07, 21, 0.25),
}
PRIORITY_SLA_HOURS = {"P1": 4, "P2": 24, "P3": 72}
# QA rubric: criterion -> (weight, is_critical, base error probability)
RUBRIC = {
    "greeting":           (10, False, 0.04),
    "verification":       (20, True,  0.03),
    "accuracy":           (25, True,  0.05),
    "policy_compliance":  (20, True,  0.04),
    "empathy":            (10, False, 0.10),
    "documentation":      (15, False, 0.12),
}
TEAMS = [f"Team {c}" for c in "ABCDEFGH"]
COACHED_TEAM, COACHING_START = "Team C", pd.Timestamp("2026-04-01")
EVALUATORS = [f"QE{i:02d}" for i in range(1, 7)]
HARSH_EVALUATOR, LENIENT_EVALUATOR = "QE04", "QE06"


def make_agents(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    agent_no = 1
    for team in TEAMS:
        for _ in range(10):
            # Most agents are established; two hiring waves join during the year.
            r = rng.random()
            if r < 0.70:
                hire = START - pd.Timedelta(days=int(rng.integers(120, 1500)))
            elif r < 0.85:
                hire = pd.Timestamp("2026-01-05")
            else:
                hire = pd.Timestamp("2026-06-01")
            rows.append({
                "agent_id": f"AG{agent_no:03d}",
                "team": team,
                "hire_date": hire.date(),
                "skill": rng.normal(0, 0.35),          # hidden: agent-level quality, used only to simulate
            })
            agent_no += 1
    return pd.DataFrame(rows)


def daily_volumes(rng: np.random.Generator) -> pd.Series:
    days = pd.date_range(START, END, freq="D")
    weekday = np.where(days.dayofweek >= 5, 0.55, 1.0)
    weekday = np.where(days.dayofweek == 0, 1.15, weekday)     # Monday backlog
    season = np.where(days.month.isin([11, 12]), 1.30, 1.0)     # year-end sales peak
    trend = np.linspace(1.0, 1.12, len(days))
    lam = 175 * weekday * season * trend
    return pd.Series(rng.poisson(lam), index=days)


def error_probabilities(skill, tenure_days, coached_months):
    """Probability that each rubric criterion is failed on a ticket."""
    tenure_effect = np.where(tenure_days < 90, 1.9, np.where(tenure_days < 180, 1.35, 1.0))
    coaching_effect = 1 - 0.40 * np.clip(coached_months / 3, 0, 1)   # ramps up over 3 months
    skill_effect = np.exp(-skill)
    return {c: np.clip(p * tenure_effect * coaching_effect * skill_effect, 0, 0.9)
            for c, (_, _, p) in RUBRIC.items()}


def make_tickets(rng: np.random.Generator, agents: pd.DataFrame) -> pd.DataFrame:
    vol = daily_volumes(rng)
    n = int(vol.sum())
    day = np.repeat(vol.index.values, vol.values)
    hour_weights = np.array([1, 1, 1, 1, 1, 2, 3, 5, 8, 10, 11, 11, 10, 10, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1], float)
    hours = rng.choice(24, size=n, p=hour_weights / hour_weights.sum())
    created = pd.to_datetime(day) + pd.to_timedelta(hours * 60 + rng.integers(0, 60, n), unit="m")

    channel = rng.choice(list(CHANNELS), size=n, p=list(CHANNELS.values()))
    cat_names = list(CATEGORIES)
    cat_p = np.array([CATEGORIES[c][0] for c in cat_names])
    category = rng.choice(cat_names, size=n, p=cat_p / cat_p.sum())
    p1_share = np.where(np.isin(category, ["Billing dispute", "Technical issue", "Account access"]), 0.20, 0.06)
    r = rng.random(n)
    priority = np.where(r < p1_share, "P1", np.where(r < p1_share + 0.30, "P2", "P3"))

    # Assign to an agent who was already hired on that day.
    hire = pd.to_datetime(agents["hire_date"]).values
    agent_idx = np.empty(n, dtype=int)
    for d in np.unique(day):
        mask = day == d
        eligible = np.flatnonzero(hire <= d)
        agent_idx[mask] = rng.choice(eligible, size=mask.sum())
    a = agents.iloc[agent_idx].reset_index(drop=True)
    tenure_days = (created - pd.to_datetime(a["hire_date"])).dt.days.values
    coached_months = np.where(
        (a["team"].values == COACHED_TEAM) & (created >= COACHING_START),
        (created - COACHING_START).days / 30.4, 0)

    # Hidden "true" quality of each interaction: which rubric criteria were failed.
    probs = error_probabilities(a["skill"].values, tenure_days, coached_months)
    errors = {c: rng.random(n) < p for c, p in probs.items()}

    # Operational measures
    load = pd.Series(1, index=created).groupby(created.normalize()).transform("count").values
    load_factor = load / np.median(load)
    frt_median = pd.Series(channel).map({"chat": 1.5, "phone": 0.8, "email": 180, "social": 45}).values
    first_response_min = rng.lognormal(np.log(frt_median * load_factor ** 1.5), 0.6)
    handle_median = pd.Series(category).map({c: v[1] for c, v in CATEGORIES.items()}).values
    handle_min = rng.lognormal(np.log(handle_median * np.where(tenure_days < 90, 1.25, 1.0)), 0.45)
    esc_p = pd.Series(category).map({c: v[2] for c, v in CATEGORIES.items()}).values
    escalated = rng.random(n) < np.clip(esc_p + 0.03 * (tenure_days < 90) + 0.05 * errors["accuracy"], 0, 1)
    base_res_h = pd.Series(priority).map({"P1": 1.5, "P2": 9, "P3": 26}).values
    resolution_h = rng.lognormal(np.log(base_res_h * load_factor ** 1.2 * np.where(escalated, 2.2, 1.0)), 0.7)
    resolution_h = np.maximum(resolution_h, (first_response_min + handle_min) / 60)
    reopened = rng.random(n) < np.clip(0.05 + 0.14 * errors["accuracy"] + 0.08 * errors["policy_compliance"]
                                       + 0.03 * errors["documentation"], 0, 1)

    # CSAT: only ~22% of customers answer the survey.
    sla_breached = resolution_h > pd.Series(priority).map(PRIORITY_SLA_HOURS).values
    sat = (4.3 - 1.1 * errors["accuracy"] - 0.6 * errors["policy_compliance"] - 0.5 * errors["empathy"]
           - 0.25 * errors["greeting"] - 0.6 * sla_breached - 0.5 * escalated - 0.8 * reopened
           + rng.normal(0, 0.7, n))
    csat = np.where(rng.random(n) < 0.22, np.clip(np.rint(sat), 1, 5), np.nan)

    tickets = pd.DataFrame({
        "ticket_id": [f"T{i:06d}" for i in range(1, n + 1)],
        "created_at": created,
        "first_response_at": created + pd.to_timedelta(first_response_min, unit="m"),
        "resolved_at": created + pd.to_timedelta(resolution_h, unit="h"),
        "channel": channel,
        "category": category,
        "priority": priority,
        "agent_id": a["agent_id"].values,
        "handle_time_min": handle_min.round(1),
        "escalated": escalated,
        "reopened": reopened,
        "csat": csat,
    })
    hidden = pd.DataFrame(errors)
    return tickets, hidden


def evaluate(rng, true_errors: pd.DataFrame, evaluator: str) -> pd.DataFrame:
    """What an evaluator records: usually the truth, with evaluator-specific bias on soft criteria."""
    out = {}
    for c, (_, critical, _) in RUBRIC.items():
        truth = true_errors[c].values
        miss = 0.05                                    # everyone occasionally misses an error
        false_flag = 0.01
        if not critical and evaluator == HARSH_EVALUATOR:
            false_flag = 0.18
        if not critical and evaluator == LENIENT_EVALUATOR:
            miss = 0.45
        seen = np.where(truth, rng.random(len(truth)) > miss, rng.random(len(truth)) < false_flag)
        out[c] = np.where(seen, "fail", "pass")
    return pd.DataFrame(out)


def make_qa(rng, tickets: pd.DataFrame, hidden: pd.DataFrame) -> pd.DataFrame:
    # QA samples about 2 tickets per agent per week.
    t = tickets.assign(week=tickets["created_at"].dt.to_period("W").astype(str))
    shuffled = t.sample(frac=1, random_state=int(rng.integers(1e9)))
    sampled = shuffled.groupby(["agent_id", "week"]).head(2).sort_values("created_at")
    hid = hidden.loc[sampled.index].reset_index(drop=True)
    sampled = sampled.reset_index(drop=True)
    evaluator = rng.choice(EVALUATORS, size=len(sampled))
    parts = []
    for ev in EVALUATORS:
        m = evaluator == ev
        parts.append(evaluate(rng, hid[m], ev).set_index(np.flatnonzero(m)))
    scores = pd.concat(parts).sort_index()
    qa = pd.DataFrame({
        "evaluation_id": [f"QA{i:06d}" for i in range(1, len(sampled) + 1)],
        "ticket_id": sampled["ticket_id"],
        "agent_id": sampled["agent_id"],
        "evaluator_id": evaluator,
        "evaluated_at": (sampled["resolved_at"] + pd.to_timedelta(rng.integers(1, 72, len(sampled)), unit="h")).dt.floor("min"),
    })
    return pd.concat([qa, scores], axis=1)


def make_calibration(rng, tickets: pd.DataFrame, hidden: pd.DataFrame) -> pd.DataFrame:
    """Each month, 10 tickets are scored independently by every evaluator."""
    rows = []
    for month, g in tickets.groupby(tickets["created_at"].dt.to_period("M")):
        pick = g.sample(10, random_state=int(rng.integers(1e9))).index
        for ev in EVALUATORS:
            s = evaluate(rng, hidden.loc[pick].reset_index(drop=True), ev)
            s.insert(0, "evaluator_id", ev)
            s.insert(0, "ticket_id", tickets.loc[pick, "ticket_id"].values)
            s.insert(0, "session_month", str(month))
            rows.append(s)
    return pd.concat(rows, ignore_index=True)


def add_dirty_data(rng, tickets: pd.DataFrame) -> pd.DataFrame:
    t = tickets.copy()
    bad_time = t.sample(40, random_state=2).index                       # resolved before created
    t.loc[bad_time, "resolved_at"] = t.loc[bad_time, "created_at"] - pd.Timedelta(hours=2)
    bad_csat = t[t["csat"].notna()].sample(60, random_state=3).index   # out-of-range survey values
    t.loc[bad_csat, "csat"] = rng.choice([0, 6, 10], size=len(bad_csat))
    orphan = t.sample(25, random_state=4).index                         # agent IDs missing from HR file
    t.loc[orphan, "agent_id"] = "AG999"
    blank = t.sample(frac=0.01, random_state=5).index                   # channel not captured
    t.loc[blank, "channel"] = ""
    dupes = t.sample(frac=0.005, random_state=1)                       # double-ingested rows
    return pd.concat([t, dupes]).sort_values("created_at").reset_index(drop=True)


def main(out_dir: Path = RAW) -> None:
    rng = np.random.default_rng(SEED)
    out_dir.mkdir(parents=True, exist_ok=True)
    agents = make_agents(rng)
    tickets, hidden = make_tickets(rng, agents)
    qa = make_qa(rng, tickets, hidden)
    calibration = make_calibration(rng, tickets, hidden)
    dirty = add_dirty_data(rng, tickets)

    agents.drop(columns="skill").to_csv(out_dir / "agents.csv", index=False)
    dirty.to_csv(out_dir / "tickets.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
    qa.to_csv(out_dir / "qa_evaluations.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
    calibration.to_csv(out_dir / "calibration.csv", index=False)
    print(f"agents: {len(agents)}, tickets: {len(dirty):,}, QA evaluations: {len(qa):,}, "
          f"calibration rows: {len(calibration):,}")


if __name__ == "__main__":
    main()
