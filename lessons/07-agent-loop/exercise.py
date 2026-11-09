"""第 07 课练习：工具循环，以及步数用尽时停下来。"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ailab.agent import run_agent
from ailab.client import MockChatClient
from ailab.tools import hotel_tools


def main() -> int:
    print("第 07 课 · 有上限的循环")
    tools = hotel_tools()
    happy = run_agent(MockChatClient(), tools, "延迟退房 2 小时要多少钱？", max_steps=4)
    print(f"正常结束：{happy.stopped_reason}")
    for step in happy.steps:
        if step.kind == "tool_call":
            print(f"  第 {step.index + 1} 步 调用 {step.name} 参数 {step.arguments}")
            print(f"    结果：{step.tool_result}")
        else:
            print(f"  第 {step.index + 1} 步 {step.kind}")
    print(f"回答：{happy.answer}")

    script = [
        json.dumps(
            {"type": "tool_call", "name": "quote_late_checkout", "arguments": {"hours": hour}},
            ensure_ascii=False,
        )
        for hour in (1, 2, 3, 4)
    ]
    capped = run_agent(
        MockChatClient(script=script),
        tools,
        "这条脚本每步都换一个小时数，用来把上限走完。",
        max_steps=3,
    )
    print(f"步数用尽：{capped.stopped_reason}，实际步数 {len(capped.steps)}")
    print(f"回答：{capped.answer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
