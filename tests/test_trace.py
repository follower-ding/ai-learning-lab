"""第 14 课：一张追踪要能分开两种拒绝，并且不记下密钥。"""

import json
from pathlib import Path

import pytest

from ailab.agent import run_agent
from ailab.client import MockChatClient
from ailab.evals import load_cases
from ailab.messages import Completion
from ailab.rag import DEFAULT_MIN_SCORE, REFUSAL, build_lab_rag
from ailab.redact import redact
from ailab.tools import hotel_tools
from ailab.trace import TraceRecord, emit_trace, format_trace, trace_agent, trace_rag

UNKNOWN_DRAFT = "游泳池全天开放。来源：[faq-pool]"
MISSING_DRAFT = "下午 14:00 可以入住，不用看来源。"
SECRET = "sk-lesson14secret"
HEADER = f"Authorization: Bearer {SECRET}"


class StubClient:
    def __init__(self, text: str) -> None:
        self.text = text

    def complete(self, messages, *, temperature: float = 0.0) -> Completion:
        del messages, temperature
        return Completion(text=self.text, model="stub", prompt_tokens=1, completion_tokens=1)


def _cases() -> dict:
    return {case.id: case for case in load_cases()}


def _reason_line(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("拒绝原因："):
            return line
    raise AssertionError("追踪里没有拒绝原因")


def test_lesson08_refusal_reasons_differ_in_the_trace() -> None:
    """客人看到同一句拒绝。追踪里要能看出是没过阈值，还是引用检查把草稿扔了。"""

    cases = _cases()
    pool = cases["refuse-pool"]
    airport = cases["refuse-airport"]
    pool_answer = build_lab_rag().answer(pool.question)
    airport_answer = build_lab_rag().answer(airport.question)
    pool_trace = trace_rag(pool.question, pool_answer)
    airport_trace = trace_rag(airport.question, airport_answer)

    assert pool_answer.refused and airport_answer.refused
    assert pool_answer.answer == airport_answer.answer == REFUSAL
    assert pool_trace.refusal_reason == airport_trace.refusal_reason == "low_score"
    assert pool_trace.top_score == 0.0
    assert airport_trace.top_score == 0.0
    assert pool_trace.top_score < DEFAULT_MIN_SCORE
    assert pool_trace.question == pool.question
    assert pool_trace.tool_name == ""
    assert pool_trace.step_count == 0
    assert pool_trace.stop_reason == "low_score"

    checkin = cases["cite-checkin"]
    unknown = build_lab_rag(StubClient(UNKNOWN_DRAFT)).answer(checkin.question)
    missing = build_lab_rag(StubClient(MISSING_DRAFT)).answer(checkin.question)
    unknown_trace = trace_rag(checkin.question, unknown)
    missing_trace = trace_rag(checkin.question, missing)

    assert unknown.answer == missing.answer == pool_answer.answer == REFUSAL
    assert unknown_trace.refusal_reason == "unknown_citation"
    assert missing_trace.refusal_reason == "missing_citation"
    assert unknown_trace.refusal_reason != pool_trace.refusal_reason
    assert missing_trace.refusal_reason != pool_trace.refusal_reason
    assert unknown_trace.top_score is not None
    assert missing_trace.top_score is not None
    assert unknown_trace.top_score >= DEFAULT_MIN_SCORE
    assert missing_trace.top_score >= DEFAULT_MIN_SCORE
    assert unknown_trace.tool_name == ""
    assert unknown_trace.step_count == 0
    assert unknown_trace.stop_reason == "unknown_citation"
    assert missing_trace.stop_reason == "missing_citation"

    pool_text = format_trace(pool_trace)
    unknown_text = format_trace(unknown_trace)
    missing_text = format_trace(missing_trace)
    assert _reason_line(pool_text) == "拒绝原因：low_score"
    assert _reason_line(unknown_text) == "拒绝原因：unknown_citation"
    assert _reason_line(missing_text) == "拒绝原因：missing_citation"
    assert _reason_line(pool_text) != _reason_line(unknown_text)
    assert _reason_line(pool_text) != _reason_line(missing_text)
    assert REFUSAL not in pool_text
    assert REFUSAL not in unknown_text
    assert REFUSAL not in missing_text
    assert "low_score" not in unknown_text
    assert "unknown_citation" not in pool_text
    assert "最高分：0.000" in pool_text
    assert f"最高分：{unknown_trace.top_score:.3f}" in unknown_text
    for label in ("问题：", "最高分：", "拒绝原因：", "工具名：", "步数：", "停止原因："):
        assert label in pool_text
        assert label in unknown_text


def test_lesson08_agent_cases_record_tool_steps_and_stop() -> None:
    cases = _cases()
    priced = cases["agent-late-three-hours"]
    result = run_agent(MockChatClient(), hotel_tools(), priced.question, max_steps=4)
    record = trace_agent(priced.question, result)
    assert record.tool_name == "quote_late_checkout"
    assert record.step_count == 2
    assert record.stop_reason == "final"
    assert record.refusal_reason == ""
    assert record.top_score is None
    text = format_trace(record)
    assert "工具名：quote_late_checkout" in text
    assert "步数：2" in text
    assert "停止原因：final" in text
    assert "拒绝原因：（无）" in text
    assert "最高分：（无）" in text

    guess = cases["agent-do-not-guess-hours"]
    asked = run_agent(MockChatClient(), hotel_tools(), guess.question, max_steps=4)
    bare = trace_agent(guess.question, asked)
    assert bare.tool_name == ""
    assert bare.step_count == 1
    assert bare.stop_reason == "final"
    assert "工具名：（无）" in format_trace(bare)
    assert "步数：1" in format_trace(bare)
    assert "30" not in format_trace(bare)

    cancel = cases["agent-cancel-policy"]
    looked = run_agent(MockChatClient(), hotel_tools(), cancel.question, max_steps=4)
    policy = trace_agent(cancel.question, looked)
    assert policy.tool_name == "lookup_policy"
    assert policy.step_count == 2
    assert policy.stop_reason == "final"
    assert list(hotel_tools()) == ["lookup_policy", "quote_late_checkout"]


def test_trace_redacts_secrets_on_stdout_before_optional_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    question = f"请不要记下 {HEADER}"
    answer = build_lab_rag().answer("游泳池几点开放？")
    record = trace_rag(question, answer)
    assert SECRET not in record.question
    assert "Bearer" not in record.question
    assert HEADER not in record.question
    assert record.question == redact(question)

    text = emit_trace(record)
    printed = capsys.readouterr().out
    assert printed == text + "\n"
    assert SECRET not in printed
    assert "Bearer" not in printed
    assert HEADER not in printed
    assert "[redacted]" in printed

    blocked = tmp_path / "missing-dir" / "trace.txt"
    with pytest.raises(FileNotFoundError):
        emit_trace(record, blocked)
    already = capsys.readouterr().out
    assert already == text + "\n"
    assert SECRET not in already
    assert not blocked.exists()

    path = tmp_path / "trace.txt"
    again = emit_trace(record, path)
    assert capsys.readouterr().out == again + "\n"
    assert path.read_text(encoding="utf-8") == again + "\n"
    assert SECRET not in path.read_text(encoding="utf-8")
    assert "Bearer" not in path.read_text(encoding="utf-8")


def test_format_redacts_a_hand_built_record() -> None:
    record = TraceRecord(
        question=HEADER,
        top_score=None,
        refusal_reason="",
        tool_name=SECRET,
        step_count=1,
        stop_reason="unknown_tool",
    )
    text = format_trace(record)
    assert SECRET not in text
    assert "Bearer" not in text
    assert HEADER not in text
    assert "[redacted]" in text
    assert "停止原因：unknown_tool" in text


def test_unknown_tool_name_is_redacted_and_not_executed() -> None:
    raw = json.dumps(
        {"type": "tool_call", "name": SECRET, "arguments": {}},
        ensure_ascii=False,
    )
    result = run_agent(MockChatClient(script=[raw]), hotel_tools(), "列目录", max_steps=2)
    record = trace_agent("列目录", result)
    assert result.stopped_reason == "unknown_tool"
    assert all(step.tool_result is None for step in result.steps)
    assert record.stop_reason == "unknown_tool"
    assert record.tool_name == "[redacted]"
    assert SECRET not in format_trace(record)
    assert list(hotel_tools()) == ["lookup_policy", "quote_late_checkout"]
