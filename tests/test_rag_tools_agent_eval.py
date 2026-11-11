"""第 05–08 课：引用或拒绝、工具、循环、评测。"""

import json

import pytest

from ailab.agent import run_agent
from ailab.client import MockChatClient
from ailab.errors import ToolArgumentError
from ailab.evals import EvalCase, format_report, load_cases, run_eval
from ailab.messages import Completion
from ailab.rag import REFUSAL, build_lab_rag
from ailab.tools import hotel_tools, quote_late_checkout


class StubClient:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    def complete(self, messages, *, temperature: float = 0.0) -> Completion:
        del messages, temperature
        self.calls += 1
        return Completion(text=self.text, model="stub", prompt_tokens=1, completion_tokens=1)


def test_rag_cites_checkin_and_refuses_pool_without_calling_model() -> None:
    counter = StubClient("不应该被调用")
    refused = build_lab_rag(counter).answer("游泳池几点开放？")
    assert refused.refused
    assert refused.reason == "low_score"
    assert refused.answer == REFUSAL
    assert counter.calls == 0

    cited = build_lab_rag().answer("入住时间是什么时候？")
    assert cited.reason == "ok"
    assert cited.citations == ["faq-checkin"]
    assert "14:00" in cited.answer
    assert not cited.refused


def test_rag_quotes_checkout_but_does_not_invent_a_price() -> None:
    result = build_lab_rag().answer("延迟退房怎么收费？")
    assert result.reason == "ok"
    assert "faq-checkout" in result.citations
    assert "30" not in result.answer
    assert "90" not in result.answer


def test_missing_or_unknown_citation_drops_the_draft() -> None:
    missing = build_lab_rag(StubClient("下午 14:00 可以入住，不用看来源。")).answer("入住时间是什么时候？")
    assert missing.reason == "missing_citation"
    assert missing.answer == REFUSAL
    assert "14:00" not in missing.answer

    unknown = build_lab_rag(StubClient("游泳池全天开放。来源：[faq-pool]")).answer("入住时间是什么时候？")
    assert unknown.reason == "unknown_citation"
    assert unknown.answer == REFUSAL
    assert "全天开放" not in unknown.answer
    assert unknown.citations == []


def test_quote_validates_hours_and_policy_lookup_is_allowlisted() -> None:
    assert "合计 60 元" in quote_late_checkout({"hours": 2})
    with pytest.raises(ToolArgumentError):
        quote_late_checkout({"hours": True})
    with pytest.raises(ToolArgumentError):
        quote_late_checkout({"hours": 13})
    with pytest.raises(ToolArgumentError):
        hotel_tools()["lookup_policy"].run({"topic": "pool"})


def test_agent_answers_late_checkout_from_the_tool() -> None:
    result = run_agent(MockChatClient(), hotel_tools(), "延迟退房 3 小时要多少钱？", max_steps=4)
    assert result.stopped_reason == "final"
    assert [step.name for step in result.steps if step.name] == ["quote_late_checkout"]
    assert "90" in result.answer
    assert "30" in result.steps[0].tool_result


def test_agent_asks_instead_of_guessing_hours() -> None:
    result = run_agent(MockChatClient(), hotel_tools(), "延迟退房怎么收费？", max_steps=4)
    assert result.stopped_reason == "final"
    assert [step.name for step in result.steps if step.name] == []
    assert "30" not in result.answer


def test_repeated_tool_stops_before_a_second_execution() -> None:
    raw = json.dumps(
        {"type": "tool_call", "name": "quote_late_checkout", "arguments": {"hours": 2}},
        ensure_ascii=False,
    )
    result = run_agent(MockChatClient(script=[raw, raw]), hotel_tools(), "再算一次", max_steps=4)
    assert result.stopped_reason == "repeated_tool"
    assert result.answer == "重复调用同一工具，已停止。"
    assert result.steps[0].tool_result is not None
    assert result.steps[1].tool_result is None


def test_unknown_tool_is_not_executed() -> None:
    calls = {"n": 0}

    def run(arguments):
        del arguments
        calls["n"] += 1
        return "不应该执行"

    from ailab.tools import Tool

    tools = {"lookup_policy": Tool("lookup_policy", "读取规章", run)}
    raw = json.dumps({"type": "tool_call", "name": "shell", "arguments": {"cmd": "ls"}}, ensure_ascii=False)
    result = run_agent(MockChatClient(script=[raw]), tools, "列目录", max_steps=3)
    assert result.stopped_reason == "unknown_tool"
    assert "shell" in result.answer
    assert calls["n"] == 0


def test_invalid_output_consumes_a_step_and_cap_stops_the_loop() -> None:
    repaired = run_agent(
        MockChatClient(
            script=[
                "我去查一下",
                json.dumps({"type": "final", "answer": "请说明主题。"}, ensure_ascii=False),
            ]
        ),
        hotel_tools(),
        "你好",
        max_steps=4,
    )
    assert [step.kind for step in repaired.steps] == ["invalid", "final"]
    assert repaired.stopped_reason == "final"

    script = [
        json.dumps(
            {"type": "tool_call", "name": "quote_late_checkout", "arguments": {"hours": hour}},
            ensure_ascii=False,
        )
        for hour in (1, 2, 3, 4)
    ]
    capped = run_agent(MockChatClient(script=script), hotel_tools(), "一直换小时数", max_steps=3)
    assert capped.stopped_reason == "max_steps"
    assert len(capped.steps) == 3
    assert capped.answer.startswith("达到步数上限")
    assert "30" not in capped.answer


def test_shipped_eval_set_passes_and_a_wrong_expect_fails() -> None:
    report = run_eval()
    assert report.passed == report.total
    assert report.total == len(load_cases())
    assert "合计" in format_report(report)

    broken = run_eval([EvalCase("bad", "rag", "游泳池几点开放？", "cite", source_id="faq-checkin")])
    assert broken.passed == 0
    assert broken.checks[0].passed is False


def test_observe_is_optional_and_sees_the_graded_outcome(monkeypatch: pytest.MonkeyPatch) -> None:
    import ailab.evals as evals

    baseline = run_eval()
    untouched = run_eval(observe=lambda _case, _outcome: None)
    assert [check.detail for check in untouched.checks] == [check.detail for check in baseline.checks]
    assert [check.passed for check in untouched.checks] == [check.passed for check in baseline.checks]

    cases = load_cases()
    assert {case.kind for case in cases} == {"rag", "agent"}
    graded: list[tuple[EvalCase, object]] = []
    real_rag = evals._grade_rag
    real_agent = evals._grade_agent

    def wrap_rag(case, result):
        graded.append((case, result))
        return real_rag(case, result)

    def wrap_agent(case, result):
        graded.append((case, result))
        return real_agent(case, result)

    monkeypatch.setattr(evals, "_grade_rag", wrap_rag)
    monkeypatch.setattr(evals, "_grade_agent", wrap_agent)
    seen: list[tuple[EvalCase, object]] = []
    report = run_eval(cases, observe=lambda case, outcome: seen.append((case, outcome)))

    assert len(seen) == len(graded) == len(cases) == len(report.checks)
    for case, (seen_case, outcome), (graded_case, graded_result), check in zip(
        cases, seen, graded, report.checks, strict=True
    ):
        assert seen_case is case
        assert graded_case is case
        assert outcome is graded_result
        if case.kind == "rag":
            assert f"reason={outcome.reason}" in check.detail
            cited = ",".join(outcome.citations) or "-"
            assert f"citations={cited}" in check.detail
        else:
            names = [step.name for step in outcome.steps if step.name]
            assert f"stop={outcome.stopped_reason}" in check.detail
            assert f"tools={','.join(names) or '-'}" in check.detail
