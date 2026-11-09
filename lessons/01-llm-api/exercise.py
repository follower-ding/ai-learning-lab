"""第 01 课练习：发送一组消息，拿回一段文本。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ailab.client import MockChatClient
from ailab.messages import Message


def main() -> int:
    client = MockChatClient()
    messages = [
        Message("system", "你是岸港酒店实验前台。只回答课程资料里有的规章，不要编造价格。"),
        Message("user", "入住时间是什么时候？"),
    ]
    completion = client.complete(messages)
    print("第 01 课 · 一次调用")
    print(f"模型：{completion.model}")
    print(f"回答：{completion.text}")
    print(
        "估算 token："
        f"prompt={completion.prompt_tokens} completion={completion.completion_tokens}"
    )
    print("这是估算，不是账单。真实接口的计数在 usage 字段里。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
