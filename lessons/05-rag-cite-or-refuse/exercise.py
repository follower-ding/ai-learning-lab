"""第 05 课练习：分数不够就拒绝；引用了没检索到的文档也拒绝。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ailab.messages import Completion
from ailab.rag import REFUSAL, build_lab_rag


class MisleadingClient:
    """故意引用一份库里不存在的文档，用来检查草稿会不会漏出去。"""

    def complete(self, messages, *, temperature: float = 0.0) -> Completion:
        del messages, temperature
        return Completion(
            text="游泳池全天开放。来源：[faq-pool]",
            model="misleading-stub",
            prompt_tokens=1,
            completion_tokens=1,
        )


def main() -> int:
    print("第 05 课 · 引用或拒绝")
    pipeline = build_lab_rag()
    for question in ("入住时间是什么时候？", "游泳池几点开放？"):
        result = pipeline.answer(question)
        best = result.hits[0]
        print(f"问题：{question}")
        print(f"最高分：{best.score:.3f} {best.id}  原因：{result.reason}")
        print(f"回答：{result.answer}")

    leaked = build_lab_rag(MisleadingClient()).answer("入住时间是什么时候？")
    print("模型草稿引用了 faq-pool，这份 id 不在本次检索结果里。")
    print(f"原因：{leaked.reason}")
    print(f"回答：{leaked.answer}")
    if leaked.answer != REFUSAL or "全天开放" in leaked.answer:
        print("草稿漏出去了。")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
