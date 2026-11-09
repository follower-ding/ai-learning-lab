"""第 08 课练习：用一小套用例检查引用、拒绝和工具。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ailab.agent import run_agent
from ailab.client import MockChatClient
from ailab.evals import format_report, run_eval
from ailab.rag import build_lab_rag
from ailab.tools import hotel_tools


def main() -> int:
    print("第 08 课 · 一小套行为评测")
    question = "延迟退房怎么收费？"
    rag_result = build_lab_rag().answer(question)
    agent_result = run_agent(MockChatClient(), hotel_tools(), question, max_steps=4)
    priced = run_agent(MockChatClient(), hotel_tools(), "延迟退房 3 小时要多少钱？", max_steps=4)
    print("同一句话，两条路径：")
    print(f"  检索：{rag_result.answer}")
    print(f"  工具：{agent_result.answer}")
    print(f"给了 3 小时：{priced.answer}")
    report = run_eval()
    print(format_report(report))
    return 0 if report.passed == report.total else 1


if __name__ == "__main__":
    raise SystemExit(main())
