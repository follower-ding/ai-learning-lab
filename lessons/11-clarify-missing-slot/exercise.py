"""第 11 课练习：缺了小时数就先问，不调用计价，也不猜 1 小时。"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ailab.agent import AgentResult, run_agent
from ailab.client import MockChatClient
from ailab.tools import hotel_tools

PRICES = ("30", "60", "90")


def main() -> int:
    print("第 11 课 · 缺槽时先问")
    tools = hotel_tools()
    print("没写小时数时输出 clarify，不调用 quote_late_checkout，也不猜 1 小时。")

    missing = run_agent(MockChatClient(), tools, "延迟退房怎么收费？", ask_first=True)
    print(f"缺小时数：停止原因 {missing.stopped_reason}")
    for step in missing.steps:
        print(f"  第 {step.index + 1} 步 {step.kind}")
        print(f"    {step.raw}")
    print(f"反问：{missing.answer}")
    if _reject_missing(missing) != 0:
        return 1

    quoted = run_agent(MockChatClient(), tools, "延迟退房 2 小时要多少钱？", ask_first=True)
    print(f"写了 2 小时：停止原因 {quoted.stopped_reason}")
    for step in quoted.steps:
        if step.kind == "tool_call":
            print(f"  第 {step.index + 1} 步 调用 {step.name} 参数 {step.arguments}")
        else:
            print(f"  第 {step.index + 1} 步 {step.kind}")
    print(f"回答：{quoted.answer}")
    names = [step.name for step in quoted.steps if step.name]
    if (
        quoted.stopped_reason != "final"
        or names != ["quote_late_checkout"]
        or "合计 60 元" not in quoted.answer
    ):
        print("写明小时数时仍应计价，并在 final 里给出合计。")
        return 1

    legacy = run_agent(MockChatClient(), tools, "延迟退房怎么收费？")
    print(f"第 06 课的协议不打开【缺槽先问】，同一句仍是 {legacy.stopped_reason}。")
    if legacy.stopped_reason != "final" or any(step.kind == "clarify" for step in legacy.steps):
        print("不传 ask_first 时不该改成 clarify。第 06 课到第 08 课仍用 final 反问。")
        return 1
    return 0


def _reject_missing(result: AgentResult) -> int:
    if result.stopped_reason != "clarify" or [step.kind for step in result.steps] != ["clarify"]:
        print("缺小时数时应只停在 clarify，而不是 final，也不要先去调用工具。")
        return 1
    if any(step.name or step.tool_result for step in result.steps):
        print("clarify 没有工具名，也不该有工具结果。")
        return 1
    payload = json.loads(result.steps[0].raw)
    if payload.get("type") != "clarify" or payload.get("question") != "要延迟几小时？":
        print('反问必须是 {"type":"clarify","question":"要延迟几小时？"}。')
        return 1
    if "hours" in payload or "1" in result.answer:
        print("不要把小时数猜成 1。")
        return 1
    visible = result.answer + result.steps[0].raw
    if any(piece in visible for piece in PRICES):
        print("反问里出现了 30、60 或 90。缺小时数时不报价。")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
