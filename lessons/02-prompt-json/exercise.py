"""第 02 课练习：把客人的问题收成一个 JSON 意图。"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ailab.client import MockChatClient
from ailab.json_prompt import classify_intent


def main() -> int:
    print("第 02 课 · 把回答收成 JSON")
    client = MockChatClient()
    questions = (
        "入住时间是什么时候？",
        "可以带宠物吗？",
        "附近有没有地铁？",
    )
    for question in questions:
        intent = classify_intent(client, question)
        print(f"{question} -> {intent.intent}（第 {intent.attempts} 次解析成功）")

    print("下面先给一句人话，再给 JSON，看修复轮次。")
    scripted = MockChatClient(
        script=[
            "我觉得客人是在问入住。",
            json.dumps(
                {"intent": "check_in", "question": "入住时间是什么时候？"},
                ensure_ascii=False,
            ),
        ]
    )
    repaired = classify_intent(scripted, "入住时间是什么时候？")
    print(f"修复后：{repaired.intent}，用了 {repaired.attempts} 次调用")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
