"""Tests for the LLM evaluator, using a fake client: no API key, no network, no cost.

The fake writer puts the ticket ID in each transcript, and the fake "perfect" judge
looks the truth up by that ID. That gives an exactly known answer for every metric.
"""
import json
import re
import threading
from types import SimpleNamespace

import pandas as pd
import pytest

import llm_evaluator as le


class FakeClient:
    """Stands in for anthropic.Anthropic(): records every request and answers it with `answer`."""

    def __init__(self, answer):
        self.requests = []
        self._answer = answer
        self._lock = threading.Lock()
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        with self._lock:
            self.requests.append(kwargs)
        return self._answer(kwargs)


def response(payload, stop_reason="end_turn", model="claude-opus-5-5"):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=json.dumps(payload))],
        stop_reason=stop_reason,
        model=model,
        usage=SimpleNamespace(input_tokens=1_000, output_tokens=500,
                              cache_read_input_tokens=0, cache_creation_input_tokens=0),
    )


def is_judge(request):
    return request["output_config"]["format"]["schema"] is le.JUDGE_SCHEMA


def fake_answer(truth, flip=None, refuse=None):
    """Writer: a minimal transcript carrying the ticket ID. Judge: the truth, with optional changes."""
    def answer(request):
        # Writer prompts end with "Ticket ref: <id>"; transcripts mention the ID in the first message.
        tid = re.search(r"T\d{6}", request["messages"][0]["content"]).group(0)
        if refuse and tid == refuse:
            return response({}, stop_reason="refusal")
        if not is_judge(request):
            return response({"transcript": f"Channel: chat\n---\nCustomer: Hi, my order {tid} is late.\n"
                                           "Agent: Hello, I'm Sam. Let me help.\n---\nInternal note: late order."})
        results = {c: {"evidence": "quoted line", "result": truth[tid][c]} for c in le.CRITERIA}
        if flip and tid == flip[0]:
            c = flip[1]
            results[c]["result"] = "pass" if results[c]["result"] == "fail" else "fail"
        return response(results)
    return answer


@pytest.fixture(scope="module")
def specs(raw_dir):
    return le.load_specs(raw_dir)


@pytest.fixture(scope="module")
def truth(raw_dir):
    t = pd.read_csv(raw_dir / "calibration_truth.csv").set_index("ticket_id")
    return t[le.CRITERIA].to_dict(orient="index")


REAL_WRITER_PROMPT = le.writer_prompt


def writer_prompt_with_id(spec):
    # The real writer prompt doesn't contain the ticket ID; the fake needs it to answer.
    return REAL_WRITER_PROMPT(spec) + f"\nTicket ref: {spec['ticket_id']}"


@pytest.fixture
def run(tmp_path, specs, truth, monkeypatch):
    """Write and judge every calibration ticket with the fakes; returns (writer client, judge client, paths)."""
    monkeypatch.setattr(le, "writer_prompt", writer_prompt_with_id)
    paths = SimpleNamespace(t=tmp_path / "transcripts.jsonl", j=tmp_path / "judgments.jsonl")

    def go(flip=None, refuse=None):
        writer = FakeClient(fake_answer(truth))
        le.generate(writer, specs, path=paths.t)
        judge = FakeClient(fake_answer(truth, flip=flip, refuse=refuse))
        summary = le.judge(judge, specs, transcripts_path=paths.t, path=paths.j)
        return writer, judge, summary

    return go, paths


# --------------------------------------------------------------------------- data and prompts

def test_truth_covers_exactly_the_calibration_tickets(raw_dir):
    truth = pd.read_csv(raw_dir / "calibration_truth.csv")
    calibration = pd.read_csv(raw_dir / "calibration.csv")
    assert len(truth) == truth.ticket_id.nunique() == 120
    assert set(truth.ticket_id) == set(calibration.ticket_id)


def test_truth_file_is_reproducible(raw_dir, tmp_path):
    import generate_data
    generate_data.main(tmp_path)
    assert (raw_dir / "calibration_truth.csv").read_bytes() == (tmp_path / "calibration_truth.csv").read_bytes()


def test_specs_list_each_tickets_true_failures(specs, truth):
    assert len(specs) == 120
    for s in specs:
        assert s["failures"] == [c for c in le.CRITERIA if truth[s["ticket_id"]][c] == "fail"]
        assert s["channel"] in le.CHANNELS


def test_writer_prompt_states_what_must_fail_and_pass():
    spec = {"ticket_id": "T000001", "channel": "email", "category": "Refund request", "priority": "P2",
            "failures": ["empathy", "documentation"]}
    prompt = le.writer_prompt(spec)
    assert "FAIL exactly these criteria: empathy, documentation." in prompt
    assert "PASS every other criterion: greeting, verification, accuracy, policy_compliance." in prompt
    assert "PASS every criterion." in le.writer_prompt({**spec, "failures": []})


# --------------------------------------------------------------------------- requests

def test_judge_never_sees_the_answer(run, specs):
    go, _ = run
    _, judge, _ = go()
    assert len(judge.requests) == 120
    systems = {r["system"] for r in judge.requests}
    assert systems == {le.JUDGE_SYSTEM}   # identical on every call, so it can be cached
    for r in judge.requests:
        user = r["messages"][0]["content"]
        assert user.startswith("<transcript>") and user.endswith("</transcript>")
        assert "FAIL" not in user and "must PASS" not in user


def test_requests_use_structured_output_and_refusal_fallback(run):
    go, _ = run
    writer, judge, _ = go()
    w, j = writer.requests[0], judge.requests[0]
    assert w["model"] == j["model"] == "claude-opus-5-5"
    assert w["output_config"] == {"effort": "low", "format": {"type": "json_schema", "schema": le.WRITER_SCHEMA}}
    assert j["output_config"]["effort"] == "medium"
    for r in (w, j):
        assert r["betas"] == ["server-side-fallback-2026-07-01"] and r["fallbacks"] == "default"
        assert "temperature" not in r


def test_haiku_requests_carry_no_fallback(tmp_path, specs, truth, monkeypatch):
    monkeypatch.setattr(le, "writer_prompt", writer_prompt_with_id)
    client = FakeClient(fake_answer(truth))
    le.generate(client, specs, model="claude-haiku-5-5", limit=2, path=tmp_path / "t.jsonl")
    assert all("fallbacks" not in r and "betas" not in r for r in client.requests)


# --------------------------------------------------------------------------- cache

def test_saved_results_are_reused(run, specs, truth):
    go, paths = run
    go()
    writer, judge = FakeClient(fake_answer(truth)), FakeClient(fake_answer(truth))
    le.generate(writer, specs, path=paths.t)
    le.judge(judge, specs, transcripts_path=paths.t, path=paths.j)
    assert writer.requests == [] and judge.requests == []


def test_a_changed_transcript_is_judged_again(run, specs, truth):
    go, paths = run
    go()
    records = le.read_jsonl(paths.t)
    records[0]["transcript"] += "\nCustomer: Thanks."
    le.write_jsonl(paths.t, records, lambda r: r["ticket_id"])
    judge = FakeClient(fake_answer(truth))
    le.judge(judge, specs, transcripts_path=paths.t, path=paths.j)
    assert len(judge.requests) == 1


def test_generate_with_another_model_rewrites_the_transcripts(run, specs, truth, capsys):
    go, paths = run
    go()   # written with Opus
    capsys.readouterr()
    haiku = FakeClient(fake_answer(truth))
    le.generate(haiku, specs, model="claude-haiku-5-5", path=paths.t)
    assert len(haiku.requests) == 120
    assert "120 of them replace transcripts written by another model" in capsys.readouterr().out
    assert {r["model"] for r in le.read_jsonl(paths.t)} == {"claude-haiku-5-5"}
    again = FakeClient(fake_answer(truth))
    le.generate(again, specs, model="claude-haiku-5-5", path=paths.t)
    assert again.requests == []


def test_estimate_counts_a_model_switch_as_pending(run, specs):
    go, paths = run
    go()
    kw = dict(transcripts_path=paths.t, judgments_path=paths.j)
    assert le.estimate(specs, model="claude-opus-5-5", **kw).pending_calls.tolist() == [0, 0]
    # generate would rewrite all 120 transcripts with Haiku, and then each needs a Haiku judgment.
    assert le.estimate(specs, model="claude-haiku-5-5", **kw).pending_calls.tolist() == [120, 120]
    assert le.estimate(specs, model="claude-opus-5-5", effort="high", **kw).pending_calls.tolist() == [0, 120]


def test_judgments_from_each_model_are_kept_side_by_side(run, specs, truth, raw_dir):
    go, paths = run
    go()   # judged by Opus
    haiku = FakeClient(fake_answer(truth))
    le.judge(haiku, specs, model="claude-haiku-5-5", transcripts_path=paths.t, path=paths.j)
    assert len(haiku.requests) == 120 and len(le.read_jsonl(paths.j)) == 240
    con = le.analyse(raw_dir, paths.t, paths.j, exports=None)
    raters = con.sql("SELECT DISTINCT rater FROM llm_rater_agreement WHERE rater_type = 'LLM'").fetchall()
    assert sorted(r for (r,) in raters) == ["claude-haiku-5-5 @ medium", "claude-opus-5-5 @ medium"]


def test_cache_files_are_sorted_and_stable(run):
    go, paths = run
    go()
    first = paths.j.read_text()
    ids = [r["ticket_id"] for r in le.read_jsonl(paths.j)]
    assert ids == sorted(ids)
    go()   # nothing pending: the file must not change
    assert paths.j.read_text() == first


def test_unusable_output_is_skipped_not_saved(run, specs):
    go, paths = run
    refused = specs[5]["ticket_id"]
    _, _, summary = go(refuse=refused)
    assert summary["failed"] == [refused]
    saved = {r["ticket_id"] for r in le.read_jsonl(paths.j)}
    assert refused not in saved and len(saved) == 119


def test_transcripts_that_mention_the_rubric_are_rejected():
    ok = "Customer: Hi\nAgent: Hello, I'm Sam."
    assert le.check_transcript(ok) == ok
    for leak in ("rubric", "policy_compliance", "the criteria"):
        with pytest.raises(le.BadOutput):
            le.check_transcript(ok + f"\nAgent: ({leak})")
    with pytest.raises(le.BadOutput):
        le.check_transcript("Just some text")


def test_a_judgment_must_score_every_criterion():
    good = {c: {"evidence": "x", "result": "pass"} for c in le.CRITERIA}
    assert le.check_judgment(good)["greeting"] == {"result": "pass", "evidence": "x"}
    with pytest.raises(le.BadOutput):
        le.check_judgment({c: v for c, v in good.items() if c != "empathy"})
    with pytest.raises(le.BadOutput):
        le.check_judgment({**good, "accuracy": {"evidence": "x", "result": "maybe"}})


# --------------------------------------------------------------------------- analysis

def llm_row(con, criteria_type="All criteria"):
    df = con.sql("SELECT * FROM llm_rater_agreement WHERE rater_type = 'LLM'").df()
    return df[df.criteria_type == criteria_type].iloc[0]


def test_a_perfect_judge_matches_the_truth_exactly(run, raw_dir):
    go, paths = run
    go()
    con = le.analyse(raw_dir, paths.t, paths.j, exports=None)
    row = llm_row(con)
    assert row.rater == "claude-opus-5-5 @ medium"
    assert (row.judgements, row.true_fails, row.caught, row.false_flags) == (720, 64, 64, 0)
    assert row.agree_truth_pct == 100.0 and row.errors_caught_pct == 100.0 and row.kappa_vs_truth == 1.0
    assert con.sql("SELECT count(*) FROM llm_disagreements").fetchone()[0] == 0


def test_one_missed_error_is_counted_once(run, raw_dir, specs):
    go, paths = run
    tid = next(s["ticket_id"] for s in specs if "accuracy" in s["failures"])
    go(flip=(tid, "accuracy"))
    con = le.analyse(raw_dir, paths.t, paths.j, exports=None)
    row = llm_row(con, "Critical criteria")
    assert (row.caught, row.true_fails, row.false_flags) == (row.true_fails - 1, row.true_fails, 0)
    d = con.sql("SELECT ticket_id, criterion, truth, llm_result, kind FROM llm_disagreements").fetchall()
    assert d == [(tid, "accuracy", "fail", "pass", "missed error")]


def test_metrics_find_the_harsh_and_lenient_evaluators(run, raw_dir):
    # The generator makes QE04 flag soft-skill passes as fails and QE06 miss soft-skill errors.
    go, paths = run
    go()
    con = le.analyse(raw_dir, paths.t, paths.j, exports=None)
    soft = con.sql("SELECT * FROM llm_rater_agreement WHERE rater_type = 'Human' "
                   "AND criteria_type = 'Soft-skill criteria'").df().set_index("rater")
    assert soft.false_flag_pct.idxmax() == "QE04"
    assert soft.errors_caught_pct.idxmin() == "QE06"


def test_run_cost_adds_up(run, raw_dir):
    go, paths = run
    go()
    con = le.analyse(raw_dir, paths.t, paths.j, exports=None)
    cost = con.sql("SELECT stage, calls, cost_usd FROM llm_run_cost").fetchall()
    # 1,000 input + 500 output tokens per call at $4 / $20 per million = $0.014 a call
    assert cost == [("write transcripts", 120, 1.68), ("judge", 120, 1.68)]
    assert le.cost_usd("claude-opus-5-5", {"input_tokens": 1_000_000, "output_tokens": 0}) == 4.0
    assert le.cost_usd("some-unknown-model", {"input_tokens": 1, "output_tokens": 1}) is None


def test_analyse_without_judgments_does_nothing(raw_dir, tmp_path):
    assert le.analyse(raw_dir, tmp_path / "none.jsonl", tmp_path / "none.jsonl", exports=tmp_path) is None
    assert list(tmp_path.iterdir()) == []


def test_ci_replay_fails_without_current_judgments(raw_dir, tmp_path):
    with pytest.raises(SystemExit, match="No current judgments"):
        le.analyse(raw_dir, tmp_path / "none.jsonl", tmp_path / "none.jsonl", exports=tmp_path, require_results=True)


def test_ci_replay_fails_when_saved_judgments_are_out_of_date(run, raw_dir, monkeypatch, capsys):
    go, paths = run
    go()
    assert le.analyse(raw_dir, paths.t, paths.j, exports=None, require_results=True) is not None
    monkeypatch.setattr(le, "JUDGE_VERSION", "judge-v2")   # e.g. the judge prompt changed
    with pytest.raises(SystemExit):
        le.analyse(raw_dir, paths.t, paths.j, exports=None, require_results=True)
    assert "Ignoring 120 saved judgment(s) that are out of date" in capsys.readouterr().out


def test_cli_passes_require_results(monkeypatch):
    seen = {}
    monkeypatch.setattr(le, "analyse", lambda **kw: seen.update(kw))
    le.main(["analyse", "--require-results"])
    assert seen == {"require_results": True}
    le.main(["analyse"])
    assert seen == {"require_results": False}


def test_estimate_counts_pending_calls(specs, tmp_path):
    est = le.estimate(specs, transcripts_path=tmp_path / "t.jsonl", judgments_path=tmp_path / "j.jsonl")
    assert est.pending_calls.tolist() == [120, 120]
    assert (est.est_cost_usd > 0).all()
