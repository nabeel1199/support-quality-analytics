"""LLM QA evaluator: can Claude score support transcripts the way the QA team does?

Usage:
    python src/llm_evaluator.py estimate              # pending calls and estimated cost (no API calls)
    python src/llm_evaluator.py generate --limit 10   # write transcripts for calibration tickets (API)
    python src/llm_evaluator.py judge --limit 10      # score the transcripts blind (API)
    python src/llm_evaluator.py analyse               # agreement tables -> exports/llm_*.csv (no API calls)

The experiment uses the 120 monthly calibration tickets, where all six human
evaluators scored the same tickets and the generator knows the hidden truth:

1. generate  An LLM writes a support transcript for each ticket in which the
             agent fails exactly the criteria that are failed in the truth.
2. judge     A separate, blind call scores each transcript against the same
             rubric. The judge never sees the truth or the failure list.
3. analyse   The LLM judge joins the six humans as a seventh rater. Every rater
             is scored against the truth (errors caught, false flags, Cohen's
             kappa) and against the human majority, reusing 04_calibration.sql.

API responses are saved to llm_eval/*.jsonl and committed, so `analyse` and the
tests replay them with no API key. A run only calls the API for tickets that
are missing or out of date (the spec, prompt version or transcript changed).
Needs `pip install -r requirements-llm.txt` and ANTHROPIC_API_KEY for the API steps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

import pipeline

ROOT = Path(__file__).resolve().parents[1]
LLM_DIR = ROOT / "llm_eval"
TRANSCRIPTS = LLM_DIR / "transcripts.jsonl"
JUDGMENTS = LLM_DIR / "judgments.jsonl"
SQL_FILE = ROOT / "sql" / "llm" / "agreement.sql"

DEFAULT_MODEL = "claude-opus-5-5"
WRITER_EFFORT = "low"          # writing a transcript to a fixed spec is easy
DEFAULT_JUDGE_EFFORT = "medium"
WRITER_VERSION = "writer-v1"   # bump when the writer prompt changes: transcripts are regenerated
JUDGE_VERSION = "judge-v1"     # bump when the judge prompt changes: transcripts are re-judged

# Server-side refusal fallback (beta). Not offered for Haiku models, so it is only sent for these.
FALLBACK_BETA = "server-side-fallback-2026-07-01"
FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}

# USD per million tokens, Anthropic API list prices (Oct 2026). Thinking tokens are billed as output.
PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}
# Rough per-call token use for `estimate` before any real calls exist (input, output incl. thinking).
GUESS_TOKENS = {"writer": (1_800, 2_500), "judge": (2_200, 2_000)}

CRITERIA = ["greeting", "verification", "accuracy", "policy_compliance", "empathy", "documentation"]
CHANNELS = {"chat", "email", "phone", "social"}

COMPANY = """\
## Company facts and procedures

Fernvale is a fictional online homeware store. Its support agents must work to these rules.

Facts
- Standard delivery takes 3-5 working days; express delivery takes 1-2 working days.
- Items can be returned within 30 days of delivery for a full refund.
- Refunds reach the original payment method 5-7 working days after the return is received.
- Password reset links expire after 24 hours.

Procedures
- Identity check: before sharing or changing anything on a customer's account (orders, payments,
  addresses, refunds, login), confirm the registered email address AND either the order number or
  the last four digits of the payment card.
- Never ask for a full card number, a security code or a password.
- A refund outside the 30-day window needs team-lead approval; an agent must not approve it alone.
- Goodwill credit is capped at GBP 10 per ticket.
- A billing dispute over GBP 100 must be escalated to the Billing team, and the customer told to
  expect a reply within 2 working days.
- A technical issue the agent cannot fix is escalated to Tier 2 with the steps already tried.
"""

RUBRIC = """\
## QA rubric

Score each criterion pass or fail.

- greeting. Pass: the agent opens with a greeting, gives their first name and offers help.
  Fail: no greeting, or no name.
- verification. Pass: the agent completes the identity check before sharing or changing account
  information, or the conversation never needs account information. Fail: account information is
  shared or changed before the check is complete, or the check uses only one identifier.
- accuracy. Pass: everything the agent tells the customer matches the company facts, and the problem
  gets a correct resolution or next step. Fail: any wrong fact (time frame, window, amount) or a wrong fix.
- policy_compliance. Pass: every procedure is followed. Fail: any procedure is broken, such as approving
  an out-of-window refund alone, offering more than GBP 10 credit, not escalating a dispute over GBP 100,
  or asking for a full card number or password. A skipped identity check is scored under verification only.
- empathy. Pass: the agent acknowledges the customer's situation or frustration where it is expressed
  and stays courteous. Fail: ignores clear frustration, or is curt, dismissive or blaming.
- documentation. Pass: the internal note states the customer's issue, what the agent did and the next
  step, and matches the conversation. Fail: the note is missing any of these, is too vague to act on,
  or contradicts the conversation.
"""

TRANSCRIPT_FORMAT = """\
## Transcript format

Channel: <channel> | Category: <category> | Priority: <priority>
---
Customer: ...
Agent: ...
(6 to 14 turns in total. Email: one message per turn. Phone: a call transcript.)
---
Internal note: <the note the agent wrote on the ticket>
"""

WRITER_SYSTEM = (
    "You write realistic synthetic customer-support transcripts. They are used to test QA evaluators, "
    "so every transcript must match its specification exactly.\n\n" + COMPANY + "\n" + RUBRIC + "\n" + TRANSCRIPT_FORMAT
)

JUDGE_SYSTEM = (
    "You are a quality evaluator on Fernvale's customer support team. You will be given one support "
    "transcript. Score it against the QA rubric, using the company facts and procedures. Judge only what "
    "the transcript shows: a criterion passes unless the transcript gives a reason to fail it. For each "
    "criterion, write one sentence of evidence from the transcript, then the result.\n\n" + COMPANY + "\n" + RUBRIC
)

WRITER_SCHEMA = {
    "type": "object",
    "properties": {"transcript": {"type": "string"}},
    "required": ["transcript"],
    "additionalProperties": False,
}

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        c: {
            "type": "object",
            "properties": {"evidence": {"type": "string"}, "result": {"type": "string", "enum": ["pass", "fail"]}},
            "required": ["evidence", "result"],
            "additionalProperties": False,
        }
        for c in CRITERIA
    },
    "required": CRITERIA,
    "additionalProperties": False,
}

EXPORT_TABLES = ["llm_rater_agreement", "llm_rater_by_criterion", "llm_disagreements", "llm_run_cost"]


class BadOutput(Exception):
    """The model answered, but the answer can't be used (refusal, cut off, wrong shape, leaked spec)."""


# --------------------------------------------------------------------------- specs and prompts

def load_specs(raw_dir: Path = pipeline.RAW) -> list:
    """One spec per calibration ticket: its channel, category, priority and the criteria it truly fails."""
    truth_file = raw_dir / "calibration_truth.csv"
    if not truth_file.exists():
        raise SystemExit("calibration_truth.csv missing - run `python scripts/generate_data.py` first.")
    truth = pd.read_csv(truth_file)
    tickets = (pd.read_csv(raw_dir / "tickets.csv", usecols=["ticket_id", "channel", "category", "priority"],
                           keep_default_na=False)
               .drop_duplicates("ticket_id"))
    merged = truth.merge(tickets, on="ticket_id", how="left", validate="one_to_one")
    specs = []
    for row in merged.sort_values("ticket_id").itertuples(index=False):
        specs.append({
            "ticket_id": row.ticket_id,
            "session_month": row.session_month,
            # A channel the ticket system didn't record is written as chat, the most common channel.
            "channel": row.channel if row.channel in CHANNELS else "chat",
            "category": row.category,
            "priority": row.priority,
            "failures": [c for c in CRITERIA if getattr(row, c) == "fail"],
        })
    return specs


def spec_key(spec: dict) -> dict:
    """The parts of a spec a transcript depends on (a change means the transcript must be rewritten)."""
    return {k: spec[k] for k in ("channel", "category", "priority", "failures")}


def writer_prompt(spec: dict) -> str:
    passes = [c for c in CRITERIA if c not in spec["failures"]]
    if spec["failures"]:
        target = (f"The agent must FAIL exactly these criteria: {', '.join(spec['failures'])}.\n"
                  f"The agent must PASS every other criterion: {', '.join(passes)}.")
    else:
        target = "The agent must PASS every criterion."
    return (
        "Write one transcript for this ticket.\n\n"
        f"Channel: {spec['channel']}\nCategory: {spec['category']}\nPriority: {spec['priority']}\n\n"
        f"{target}\n\n"
        "Make each failure clear to a careful evaluator applying the rubric, but natural, the way real agents "
        "slip up. Do not mention the rubric, scores or criteria anywhere in the transcript. Invent the customer's "
        "name and order details; use no real people or companies."
    )


def judge_prompt(transcript: str) -> str:
    return f"<transcript>\n{transcript}\n</transcript>"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- API calls

def make_client():
    """An Anthropic client, or a clear message if the SDK or credentials are missing."""
    try:
        import anthropic
    except ImportError:
        raise SystemExit("The LLM steps need the Anthropic SDK: pip install -r requirements-llm.txt")
    has_credentials = (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
                       or (Path.home() / ".config" / "anthropic").exists())
    if not has_credentials:
        raise SystemExit("No Anthropic credentials found. Set ANTHROPIC_API_KEY in this shell "
                         "(or run `ant auth login`), then run the command again.")
    return anthropic.Anthropic(max_retries=4)


def call_claude(client, *, model: str, system: str, user: str, schema: dict, effort: str,
                max_tokens: int = 16_000):
    """One structured-output request. Returns (parsed JSON, usage dict, model that served it)."""
    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
        "output_config": {"effort": effort, "format": {"type": "json_schema", "schema": schema}},
        "cache_control": {"type": "ephemeral"},   # the system prompt is identical on every call
    }
    if model in FALLBACK_MODELS:
        kwargs.update(betas=[FALLBACK_BETA], fallbacks="default")
    response = client.beta.messages.create(**kwargs)
    if response.stop_reason == "refusal":
        raise BadOutput("the model declined the request")
    if response.stop_reason == "max_tokens":
        raise BadOutput("the response hit max_tokens")
    text = next((b.text for b in response.content if b.type == "text"), None)
    if text is None:
        raise BadOutput("no text block in the response")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise BadOutput(f"invalid JSON: {e}")
    u = response.usage
    usage = {
        "input_tokens": u.input_tokens or 0,
        "output_tokens": u.output_tokens or 0,
        "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
        "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
    }
    return data, usage, response.model


def cost_usd(model: str, usage: dict) -> Optional[float]:
    if model not in PRICES:
        return None
    p_in, p_out = PRICES[model]
    return (usage["input_tokens"] * p_in
            + usage.get("cache_creation_input_tokens", 0) * p_in * 1.25
            + usage.get("cache_read_input_tokens", 0) * p_in * 0.1
            + usage["output_tokens"] * p_out) / 1e6


def check_transcript(text: str) -> str:
    text = text.strip()
    if "Customer:" not in text or "Agent:" not in text:
        raise BadOutput("transcript is missing Customer/Agent turns")
    lowered = text.lower()
    if "rubric" in lowered or any(c in lowered for c in ("policy_compliance", "criteria", "criterion")):
        raise BadOutput("transcript mentions the rubric, so it would leak the answer to the judge")
    return text


def check_judgment(data: dict) -> dict:
    if set(data) != set(CRITERIA):
        raise BadOutput(f"judgment has criteria {sorted(data)}, expected {CRITERIA}")
    for c in CRITERIA:
        if data[c].get("result") not in ("pass", "fail"):
            raise BadOutput(f"{c}: result must be pass or fail")
    return {c: {"result": data[c]["result"], "evidence": data[c].get("evidence", "").strip()} for c in CRITERIA}


# --------------------------------------------------------------------------- cache (JSONL files)

def read_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, records: list, sort_key: Callable) -> None:
    """Sorted, one record per line, written atomically so an interrupted run can't corrupt the cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in sorted(records, key=sort_key)))
    tmp.replace(path)


def transcript_is_current(record: Optional[dict], spec: dict) -> bool:
    return bool(record) and record["writer_version"] == WRITER_VERSION and record["spec"] == spec_key(spec)


def current_transcripts(specs: list, path: Path = TRANSCRIPTS) -> dict:
    """ticket_id -> transcript record, for tickets whose saved transcript still matches its spec."""
    saved = {r["ticket_id"]: r for r in read_jsonl(path)}
    return {s["ticket_id"]: saved[s["ticket_id"]] for s in specs if transcript_is_current(saved.get(s["ticket_id"]), s)}


def judgment_is_current(record: Optional[dict], transcript: dict) -> bool:
    return (bool(record) and record["judge_version"] == JUDGE_VERSION
            and record["transcript_sha"] == sha(transcript["transcript"]))


def run_parallel(todo: list, work: Callable, store: dict, key: Callable, path: Path, sort_key: Callable,
                 workers: int, label: str) -> dict:
    """Run work(item) for each item, saving every result to the cache as soon as it arrives."""
    lock = threading.Lock()
    failed, spent = [], 0.0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(work, item): item for item in todo}
        try:
            for n, fut in enumerate(as_completed(futures), 1):
                item = futures[fut]
                try:
                    record = fut.result()
                except BadOutput as e:
                    failed.append(item["ticket_id"])
                    print(f"[{n}/{len(todo)}] {label} {item['ticket_id']}: skipped ({e})")
                    continue
                with lock:
                    store[key(record)] = record
                    write_jsonl(path, list(store.values()), sort_key)
                spent += cost_usd(record["model"], record["usage"]) or 0.0
                print(f"[{n}/{len(todo)}] {label} {item['ticket_id']}: ok (running cost ${spent:.2f})")
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)
            print(f"Stopped. Finished results are saved in {path}; run again to continue.")
            raise
    if failed:
        print(f"{len(failed)} {label} call(s) gave unusable output and were not saved: {', '.join(failed)}. "
              "Run the command again to retry them.")
    return {"done": len(todo) - len(failed), "failed": failed, "cost_usd": round(spent, 4)}


def generate(client, specs: list, model: str = DEFAULT_MODEL, limit: Optional[int] = None,
             workers: int = 4, force: bool = False, path: Path = TRANSCRIPTS) -> dict:
    """Write a transcript for every calibration ticket that doesn't have a current one."""
    store = {r["ticket_id"]: r for r in read_jsonl(path)}
    todo = [s for s in specs if force or not transcript_is_current(store.get(s["ticket_id"]), s)][:limit]

    def work(spec):
        data, usage, served = call_claude(client, model=model, system=WRITER_SYSTEM, user=writer_prompt(spec),
                                          schema=WRITER_SCHEMA, effort=WRITER_EFFORT)
        return {"ticket_id": spec["ticket_id"], "spec": spec_key(spec), "writer_version": WRITER_VERSION,
                "model": model, "served_by": served, "effort": WRITER_EFFORT,
                "transcript": check_transcript(data["transcript"]), "usage": usage}

    print(f"Writing {len(todo)} transcript(s) with {model}.")
    return run_parallel(todo, work, store, lambda r: r["ticket_id"], path, lambda r: r["ticket_id"], workers, "write")


def judge(client, specs: list, model: str = DEFAULT_MODEL, effort: str = DEFAULT_JUDGE_EFFORT,
          limit: Optional[int] = None, workers: int = 4, force: bool = False,
          transcripts_path: Path = TRANSCRIPTS, path: Path = JUDGMENTS) -> dict:
    """Score every current transcript blind. The judge sees only the transcript, never the spec."""
    transcripts = current_transcripts(specs, transcripts_path)
    store = {(r["ticket_id"], r["model"], r["effort"]): r for r in read_jsonl(path)}
    todo = [t for t in transcripts.values()
            if force or not judgment_is_current(store.get((t["ticket_id"], model, effort)), t)][:limit]

    def work(t):
        data, usage, served = call_claude(client, model=model, system=JUDGE_SYSTEM, user=judge_prompt(t["transcript"]),
                                          schema=JUDGE_SCHEMA, effort=effort)
        return {"ticket_id": t["ticket_id"], "judge_version": JUDGE_VERSION, "model": model, "served_by": served,
                "effort": effort, "transcript_sha": sha(t["transcript"]), "results": check_judgment(data),
                "usage": usage}

    print(f"Judging {len(todo)} transcript(s) with {model} at {effort} effort.")
    key = lambda r: (r["ticket_id"], r["model"], r["effort"])  # noqa: E731
    return run_parallel(todo, work, store, key, path, key, workers, "judge")


# --------------------------------------------------------------------------- estimate and analyse

def estimate(specs: list, model: str = DEFAULT_MODEL, effort: str = DEFAULT_JUDGE_EFFORT,
             transcripts_path: Path = TRANSCRIPTS, judgments_path: Path = JUDGMENTS) -> pd.DataFrame:
    """Pending calls and their estimated cost, from measured token use when calls have already been made."""
    saved_t = {r["ticket_id"]: r for r in read_jsonl(transcripts_path)}
    pending_write = sum(not transcript_is_current(saved_t.get(s["ticket_id"]), s) for s in specs)
    saved_j = {(r["ticket_id"], r["model"], r["effort"]): r for r in read_jsonl(judgments_path)}
    current = current_transcripts(specs, transcripts_path)
    judged = sum(judgment_is_current(saved_j.get((tid, model, effort)), t) for tid, t in current.items())
    pending_judge = len(specs) - judged
    rows = []
    for stage, n, past in [("write", pending_write, list(saved_t.values())), ("judge", pending_judge, list(saved_j.values()))]:
        past = [r for r in past if r["model"] == model]
        if past:
            tok_in = sum(r["usage"]["input_tokens"] + r["usage"]["cache_read_input_tokens"]
                         + r["usage"]["cache_creation_input_tokens"] for r in past) / len(past)
            tok_out = sum(r["usage"]["output_tokens"] for r in past) / len(past)
            basis = f"measured on {len(past)} calls"
        else:
            tok_in, tok_out = GUESS_TOKENS["writer" if stage == "write" else "judge"]
            basis = "rough guess"
        per_call = cost_usd(model, {"input_tokens": tok_in, "output_tokens": tok_out})
        rows.append({"stage": stage, "model": model, "pending_calls": n, "tokens_in_per_call": round(tok_in),
                     "tokens_out_per_call": round(tok_out), "basis": basis,
                     "est_cost_usd": None if per_call is None else round(n * per_call, 2)})
    return pd.DataFrame(rows)


def analyse(raw_dir: Path = pipeline.RAW, transcripts_path: Path = TRANSCRIPTS, judgments_path: Path = JUDGMENTS,
            exports: Optional[Path] = pipeline.EXPORTS):
    """Score every rater (six humans + each LLM judge) against the truth and the human majority."""
    specs = load_specs(raw_dir)
    transcripts = current_transcripts(specs, transcripts_path)
    judgments = [j for j in read_jsonl(judgments_path)
                 if j["ticket_id"] in transcripts and judgment_is_current(j, transcripts[j["ticket_id"]])]
    if not judgments:
        print("No current judgments yet - run `generate` and `judge` first.")
        return None

    con = pipeline.build(raw_dir=raw_dir)
    con.execute(f"CREATE OR REPLACE VIEW raw_calibration_truth AS "
                f"SELECT * FROM read_csv_auto('{(raw_dir / 'calibration_truth.csv').as_posix()}')")
    judged = pd.DataFrame(
        [(j["ticket_id"], f"{j['model']} @ {j['effort']}", c, j["results"][c]["result"], j["results"][c]["evidence"])
         for j in judgments for c in CRITERIA],
        columns=["ticket_id", "rater", "criterion", "result", "evidence"])
    calls = pd.DataFrame(
        [{"stage": stage, "model": r["model"], **r["usage"], "cost_usd": cost_usd(r["model"], r["usage"])}
         for stage, records in [("write transcripts", list(transcripts.values())), ("judge", judgments)]
         for r in records])
    con.register("judged_df", judged)
    con.register("calls_df", calls)
    con.execute("CREATE OR REPLACE TABLE llm_judgments AS SELECT * FROM judged_df")
    con.execute("CREATE OR REPLACE TABLE llm_calls AS SELECT * FROM calls_df")
    con.execute(SQL_FILE.read_text())
    if exports is not None:
        exports.mkdir(exist_ok=True)
        for table in EXPORT_TABLES:
            con.table(table).df().to_csv(exports / f"{table}.csv", index=False)
    return con


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("estimate", "generate", "judge", "analyse"):
        p = sub.add_parser(name)
        if name != "analyse":
            p.add_argument("--model", default=DEFAULT_MODEL)
        if name in ("estimate", "judge"):
            p.add_argument("--effort", default=DEFAULT_JUDGE_EFFORT, choices=["low", "medium", "high", "xhigh", "max"])
        if name in ("generate", "judge"):
            p.add_argument("--limit", type=int, help="only make this many calls (try 5 first)")
            p.add_argument("--workers", type=int, default=4, help="parallel requests")
            p.add_argument("--force", action="store_true", help="redo calls even when the saved result is current")
    args = parser.parse_args(argv)

    if args.command == "analyse":
        con = analyse()
        if con is not None:
            print(con.table("llm_rater_agreement").df().to_string(index=False))
            print(f"Exported {len(EXPORT_TABLES)} tables to {pipeline.EXPORTS}")
        return
    specs = load_specs()
    if args.command == "estimate":
        print(estimate(specs, args.model, args.effort).to_string(index=False))
    elif args.command == "generate":
        print(generate(make_client(), specs, args.model, args.limit, args.workers, args.force))
    elif args.command == "judge":
        print(judge(make_client(), specs, args.model, args.effort, args.limit, args.workers, args.force))


if __name__ == "__main__":
    main()
