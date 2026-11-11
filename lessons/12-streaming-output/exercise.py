"""第 12 课练习：先收齐再检查，以及边打印边检查。作废的半句不留在最终回答里。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ailab.messages import Completion
from ailab.rag import REFUSAL, build_lab_rag
from ailab.streaming import (
    BUFFER_THEN_CHECK,
    PRINT_WHILE_CHECKING,
    STRATEGY_LABEL,
    StreamingAnswer,
    answer_streaming,
)

HALF = "游泳池全天开放"
UNKNOWN_DRAFT = "游泳池全天开放。来源：[faq-pool] 而且免费。"
MISSING_DRAFT = "下午 14:00 可以入住，不用看来源。"
STRATEGIES = (BUFFER_THEN_CHECK, PRINT_WHILE_CHECKING)


class FixedClient:
    """每次都返回同一段草稿，用来对照两种办法的屏幕。"""

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    def complete(self, messages, *, temperature: float = 0.0) -> Completion:
        del messages, temperature
        self.calls += 1
        return Completion(text=self.text, model="fixed-stub", prompt_tokens=1, completion_tokens=1)


def main() -> int:
    print("第 12 课 · 流式输出")
    print("同一段草稿并排走两种办法：先收齐再检查，以及边打印边检查。")
    print("引用检查要看到完整的 [faq-...] 才决定留还是扔。最终回答不保留被作废的半句。")

    if _kept() != 0:
        return 1
    if _unknown() != 0:
        return 1
    if _missing() != 0:
        return 1
    if _low_score() != 0:
        return 1
    return 0


def _show(result: StreamingAnswer) -> None:
    print(STRATEGY_LABEL[result.strategy])
    if not result.streamed:
        print("  中途：（无）")
    for piece in result.streamed:
        print(f"  中途：{piece}")
    print(f"  最终回答：{result.final_answer}")
    print(f"  原因：{result.reason}")


def _both(pipeline, question: str) -> tuple[StreamingAnswer, StreamingAnswer]:
    buffered = answer_streaming(pipeline, question, BUFFER_THEN_CHECK)
    printed = answer_streaming(pipeline, question, PRINT_WHILE_CHECKING)
    for result in (buffered, printed):
        _show(result)
    return buffered, printed


def _kept() -> int:
    print("入住问题。Mock 会带上 [faq-checkin]，这半句该留下。")
    buffered, printed = _both(build_lab_rag(), "入住时间是什么时候？")
    if buffered.streamed:
        print("先收齐再检查时，中途不该有字。整段要等检查完再给。")
        return 1
    if (
        buffered.reason != "ok"
        or printed.reason != "ok"
        or buffered.citations != ["faq-checkin"]
        or printed.citations != ["faq-checkin"]
    ):
        print("引用成立时，两种办法的原因都应当是 ok，引用都是 faq-checkin。")
        return 1
    if "14:00" not in buffered.final_answer or "14:00" not in printed.final_answer:
        print("这半句没有被作废，最终回答里应当留下 14:00。")
        return 1
    if not printed.streamed or "[faq-" in printed.streamed[0]:
        print("边打印时，第一段应当是引用还没到的那半句。")
        return 1
    if "[faq-checkin]" not in printed.streamed:
        print("引用成立时不该提前停，[faq-checkin] 应当出现在中途。")
        return 1
    if "".join(printed.streamed).strip() != printed.final_answer:
        print("留下的草稿，最终回答应当就是中途这些段拼起来的结果。")
        return 1
    if buffered.final_answer != printed.final_answer:
        print("两种办法的最终回答应当相同。")
        return 1
    return 0


def _unknown() -> int:
    print("模型写了 [faq-pool]，后面还多了一句「而且免费」。第 05 课会整段作废。")
    print("下面只印每种办法真正推到屏幕上的字。")
    pipeline = build_lab_rag(FixedClient(UNKNOWN_DRAFT))
    buffered, printed = _both(pipeline, "入住时间是什么时候？")
    if buffered.streamed:
        print("先收齐再检查时，作废的半句不该出现在中途。")
        return 1
    if _final_keeps_discarded(buffered):
        print("先收齐再检查时，最终回答留下了被作废的半句。")
        return 1
    if not any(HALF in piece for piece in printed.streamed):
        print("边打印边检查时，作废的半句应当先出现在中途。")
        return 1
    if any("而且免费" in piece for piece in printed.streamed):
        print("未知引用的方括号已经闭合，后面的段不该再推到屏幕上。")
        return 1
    if _final_keeps_discarded(printed):
        print("边打印边检查时，最终回答必须清掉作废的半句。")
        return 1
    if buffered.final_answer != printed.final_answer:
        print("两种办法的最终回答应当相同。")
        return 1
    return 0


def _missing() -> int:
    print("模型没写引用。检查要等到流结束才知道该扔。")
    pipeline = build_lab_rag(FixedClient(MISSING_DRAFT))
    buffered, printed = _both(pipeline, "入住时间是什么时候？")
    if buffered.streamed or buffered.reason != "missing_citation":
        print("先收齐再检查：中途不该有字，原因应当是 missing_citation。")
        return 1
    if "14:00" in buffered.final_answer or "不用看来源" in buffered.final_answer:
        print("先收齐再检查时，缺少引用的半句不该留在最终回答里。")
        return 1
    joined = "".join(printed.streamed)
    if "14:00" not in joined or "不用看来源" not in joined:
        print("边打印边检查时，缺少引用的半句应当先出现在中途。")
        return 1
    if printed.reason != "missing_citation":
        print("没闭合或没写完的引用，不能提前当成 unknown_citation。")
        return 1
    if "14:00" in printed.final_answer or "不用看来源" in printed.final_answer:
        print("边打印边检查时，最终回答必须清掉这半句。")
        return 1
    if buffered.final_answer != REFUSAL or printed.final_answer != REFUSAL:
        print("最终回答应当是第 05 课那句拒绝。")
        return 1
    return 0


def _low_score() -> int:
    print("游泳池几点开放？分数不够，两种办法都不调用模型，中途也没有草稿。")
    client = FixedClient(UNKNOWN_DRAFT)
    pipeline = build_lab_rag(client)
    for strategy in STRATEGIES:
        result = answer_streaming(pipeline, "游泳池几点开放？", strategy)
        _show(result)
        if result.streamed or result.reason != "low_score" or result.final_answer != REFUSAL:
            print("分数不够时不该流出草稿，最终回答仍是那句拒绝。")
            return 1
        if HALF in result.final_answer:
            print("模型没被调用，最终回答里不该出现那半句假话。")
            return 1
    if client.calls != 0:
        print("分数不够时不该调用模型。这和第 05 课一样。")
        return 1
    return 0


def _final_keeps_discarded(result: StreamingAnswer) -> bool:
    if result.final_answer != REFUSAL or result.reason != "unknown_citation":
        return True
    return HALF in result.final_answer or "而且免费" in result.final_answer


if __name__ == "__main__":
    raise SystemExit(main())
