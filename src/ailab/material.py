"""第 13 课：检索到的段落和工具返回的字符串是数据，不是新的指令。

工具名只来自模型输出的 JSON，并且必须在允许列表里。
资料正文不提供工具名。两只工具仍是纯函数，没有命令行。
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ailab.agent import AgentResult, run_agent
from ailab.client import MockChatClient
from ailab.tools import Tool, hotel_tools

ALLOWED_TOOL_NAMES = ("lookup_policy", "quote_late_checkout")


@dataclass(frozen=True)
class Material:
    """一段检索到的或工具返回的文字。只是数据。"""

    text: str


@dataclass(frozen=True)
class Settled:
    """工具返回之后怎么看待这段文字。

    ``tool_name`` 只可能是允许列表里的名字，或 None。
    ``text`` 是返回的原文，不删改。
    ``name_from_text`` 是从原文里读出的工具名。资料不是指令，所以是 None。
    """

    tool_name: str | None
    text: str
    name_from_text: str | None


@dataclass(frozen=True)
class InjectedRun:
    """用本课夹具接在工具返回值后面的一次循环。"""

    result: AgentResult
    settled: tuple[Settled, ...]


def tool_name_from_material(material: Material) -> str | None:
    """不从资料正文里读工具名。

    调用方把整段资料传进来，是为了标明这段文字只是数据。
    返回值必须是 None。名字只来自模型的 JSON，再交允许列表决定。
    """

    if not isinstance(material, Material):
        raise TypeError("material 必须是资料")
    return None


def settle_tool_result(
    executed_name: str,
    result_text: str,
    tools: Mapping[str, Tool],
) -> Settled:
    """把工具返回的字符串留在资料里。工具名不从这段字符串里取。"""

    name_from_text = tool_name_from_material(Material(result_text))
    if executed_name in ALLOWED_TOOL_NAMES and executed_name in tools:
        tool_name = executed_name
    else:
        tool_name = None
    return Settled(tool_name=tool_name, text=result_text, name_from_text=name_from_text)


def run_with_injected_material(question: str, fixture: str) -> InjectedRun:
    """让 ``lookup_policy`` 的返回值末尾多出本课夹具，再跑现有的循环。

    不改默认的 ``hotel_tools()``。允许列表的键仍是那两只。
    夹具只接在返回的字符串后面，不参与选择工具名。
    """

    extra = fixture.strip()
    if extra == "":
        raise ValueError("夹具不能是空字符串")
    tools = hotel_tools()
    if set(tools) != set(ALLOWED_TOOL_NAMES) or len(tools) != len(ALLOWED_TOOL_NAMES):
        raise ValueError("允许列表必须正好是 lookup_policy 和 quote_late_checkout")
    original = tools["lookup_policy"]

    def run(arguments: Mapping[str, Any]) -> str:
        body = original.run(arguments)
        return f"{body}\n{extra}"

    tools["lookup_policy"] = Tool(original.name, original.description, run)
    result = run_agent(MockChatClient(), tools, question)
    settled: list[Settled] = []
    for step in result.steps:
        if step.name is None or step.tool_result is None:
            continue
        settled.append(settle_tool_result(step.name, step.tool_result, tools))
    return InjectedRun(result, tuple(settled))
