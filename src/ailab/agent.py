"""有步数上限的工具循环。停下来时说明原因，不用猜测把答案补圆。"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ailab.client import ChatClient
from ailab.errors import JsonExtractError, ToolArgumentError
from ailab.json_prompt import extract_json_object
from ailab.messages import Message
from ailab.tools import Tool

DEFAULT_MAX_STEPS = 4


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class FinalAnswer:
    answer: str


@dataclass(frozen=True)
class AgentStep:
    index: int
    kind: str
    name: str | None
    arguments: dict[str, Any] | None
    tool_result: str | None
    raw: str


@dataclass(frozen=True)
class AgentResult:
    answer: str
    steps: list[AgentStep]
    stopped_reason: str


def build_tool_system(tools: Mapping[str, Tool]) -> str:
    lines = [
        "你是岸港酒店实验前台，按【工具协议】行动。",
        "每一轮只输出一个 JSON 对象，不要加解释，不要加 Markdown。",
        '调用工具：{"type":"tool_call","name":"工具名","arguments":{}}',
        '结束并回答：{"type":"final","answer":"给客人的话"}',
        "只能使用下面列出的工具。问延迟退房多少钱时，没有小时数就先问清楚，不要猜。",
        "看到工具结果后输出 final，不要用相同参数再调用一次。",
        "可用工具：",
    ]
    for tool in tools.values():
        lines.append(f"- {tool.name}：{tool.description}")
    return "\n".join(lines)


def parse_action(text: str) -> ToolCall | FinalAnswer:
    data = extract_json_object(text)
    kind = data.get("type")
    if kind == "final":
        answer = data.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise JsonExtractError("final 缺少 answer")
        return FinalAnswer(answer.strip())
    if kind == "tool_call":
        name = data.get("name")
        arguments = data.get("arguments", {})
        if not isinstance(name, str) or not name.strip():
            raise JsonExtractError("tool_call 缺少 name")
        if not isinstance(arguments, dict):
            raise JsonExtractError("arguments 必须是对象")
        return ToolCall(name.strip(), dict(arguments))
    raise JsonExtractError("type 必须是 tool_call 或 final")


def run_agent(
    client: ChatClient,
    tools: Mapping[str, Tool],
    question: str,
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> AgentResult:
    """最多走 ``max_steps`` 轮。相同工具和参数出现第二次就停。"""

    if max_steps <= 0:
        raise ValueError("max_steps 必须是正整数")
    messages = [Message("system", build_tool_system(tools)), Message("user", question)]
    steps: list[AgentStep] = []
    seen: set[str] = set()
    for index in range(max_steps):
        raw = client.complete(messages).text
        try:
            action = parse_action(raw)
        except JsonExtractError as exc:
            steps.append(AgentStep(index, "invalid", None, None, None, raw))
            messages.extend(
                [
                    Message("assistant", raw),
                    Message("user", f"无法解析：{exc}。只输出一个 JSON 对象。"),
                ]
            )
            continue
        if isinstance(action, FinalAnswer):
            steps.append(AgentStep(index, "final", None, None, None, raw))
            return AgentResult(action.answer, steps, "final")

        signature = _signature(action)
        if signature in seen:
            steps.append(AgentStep(index, "tool_call", action.name, action.arguments, None, raw))
            return AgentResult("重复调用同一工具，已停止。", steps, "repeated_tool")
        seen.add(signature)

        tool = tools.get(action.name)
        if tool is None:
            steps.append(AgentStep(index, "tool_call", action.name, action.arguments, None, raw))
            return AgentResult(f"不允许的工具：{action.name}。已停止。", steps, "unknown_tool")
        try:
            tool_result = tool.run(action.arguments)
        except ToolArgumentError as exc:
            tool_result = f"工具参数错误：{exc}"
        steps.append(AgentStep(index, "tool_call", action.name, action.arguments, tool_result, raw))
        messages.extend(
            [
                Message("assistant", raw),
                Message("tool", tool_result, name=action.name),
            ]
        )
    return AgentResult("达到步数上限，已停止。请缩小问题后再试。", steps, "max_steps")


def _signature(action: ToolCall) -> str:
    encoded = json.dumps(action.arguments, sort_keys=True, ensure_ascii=False, default=str)
    return f"{action.name}:{encoded}"
