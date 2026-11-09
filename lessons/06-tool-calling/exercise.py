"""第 06 课练习：模型提出调用，代码执行允许列表里的工具。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ailab.agent import ToolCall, build_tool_system, parse_action
from ailab.client import MockChatClient
from ailab.errors import ToolArgumentError
from ailab.messages import Message
from ailab.tools import hotel_tools, quote_late_checkout


def main() -> int:
    print("第 06 课 · 工具调用")
    tools = hotel_tools()
    system = Message("system", build_tool_system(tools))
    for question in ("延迟退房 2 小时要多少钱？", "延迟退房怎么收费？"):
        raw = MockChatClient().complete([system, Message("user", question)]).text
        action = parse_action(raw)
        print(f"问题：{question}")
        print(f"模型输出：{raw}")
        if isinstance(action, ToolCall):
            print(f"代码执行：{tools[action.name].run(action.arguments)}")
        else:
            print(f"直接回答：{action.answer}")

    shell_call = parse_action('{"type":"tool_call","name":"shell","arguments":{"cmd":"ls"}}')
    assert isinstance(shell_call, ToolCall)
    decision = "执行" if shell_call.name in tools else "拒绝"
    print(f"允许列表之外：{shell_call.name} -> {decision}")
    try:
        quote_late_checkout({"hours": True})
    except ToolArgumentError as exc:
        print(f"坏参数：{exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
