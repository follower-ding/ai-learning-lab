"""第 11 课：缺小时数时用 clarify 先问，不调用计价，也不猜 1 小时。"""

import json

import pytest

from ailab.agent import Clarify, build_tool_system, parse_action, run_agent
from ailab.client import MockChatClient
from ailab.errors import JsonExtractError
from ailab.tools import Tool, hotel_tools


def test_parse_action_reads_clarify_question() -> None:
    action = parse_action('{"type":"clarify","question":"要延迟几小时？"}')
    assert isinstance(action, Clarify)
    assert action.question == "要延迟几小时？"

    with pytest.raises(JsonExtractError):
        parse_action('{"type":"clarify"}')
    with pytest.raises(JsonExtractError):
        parse_action('{"type":"clarify","question":"   "}')
    with pytest.raises(JsonExtractError):
        parse_action('{"type":"clarify","question":1}')


def test_clarify_stops_the_loop_without_calling_a_tool() -> None:
    raw = json.dumps({"type": "clarify", "question": "要延迟几小时？"}, ensure_ascii=False)
    result = run_agent(
        MockChatClient(script=[raw]),
        hotel_tools(),
        "延迟退房怎么收费？",
        max_steps=4,
    )
    assert result.stopped_reason == "clarify"
    assert [step.kind for step in result.steps] == ["clarify"]
    assert result.answer == "要延迟几小时？"
    assert all(step.name is None and step.tool_result is None for step in result.steps)


def test_missing_hours_clarifies_and_does_not_quote_or_guess() -> None:
    """缺小时数时必须是 clarify。用 final 反问、或猜成 1 小时去计价，都算失败。"""

    calls: list[dict] = []
    tools = hotel_tools()
    quote = tools["quote_late_checkout"]

    def counting(arguments):
        calls.append(dict(arguments))
        return quote.run(arguments)

    tools["quote_late_checkout"] = Tool(quote.name, quote.description, counting)
    result = run_agent(
        MockChatClient(),
        tools,
        "延迟退房怎么收费？",
        max_steps=4,
        ask_first=True,
    )

    assert calls == []
    assert result.stopped_reason == "clarify"
    assert [step.kind for step in result.steps] == ["clarify"]
    assert all(step.name is None for step in result.steps)
    payload = json.loads(result.steps[0].raw)
    assert payload["type"] == "clarify"
    assert payload["question"] == "要延迟几小时？"
    assert "hours" not in payload
    assert result.answer == "要延迟几小时？"
    visible = "\n".join(
        [result.answer, result.steps[0].raw, *(step.tool_result or "" for step in result.steps)]
    )
    for piece in ("30", "60", "90"):
        assert piece not in visible
    assert "1" not in result.answer


def test_stated_hours_still_quote_when_asking_first() -> None:
    result = run_agent(
        MockChatClient(),
        hotel_tools(),
        "延迟退房 2 小时要多少钱？",
        max_steps=4,
        ask_first=True,
    )
    assert result.stopped_reason == "final"
    assert [step.name for step in result.steps if step.name] == ["quote_late_checkout"]
    assert "合计 60 元" in result.answer


def test_default_protocol_still_uses_final_and_keeps_the_allowlist() -> None:
    tools = hotel_tools()
    assert list(tools) == ["lookup_policy", "quote_late_checkout"]
    assert "【缺槽先问】" not in build_tool_system(tools)
    assert "【缺槽先问】" in build_tool_system(tools, ask_first=True)

    legacy = run_agent(MockChatClient(), tools, "延迟退房怎么收费？", max_steps=4)
    assert legacy.stopped_reason == "final"
    assert [step.name for step in legacy.steps if step.name] == []
    assert "30" not in legacy.answer
