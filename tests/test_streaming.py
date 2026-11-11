"""第 12 课：两种流式送到屏幕。作废的半句可以在中途出现，最终回答里必须清掉。"""

import pytest

from ailab.messages import Completion
from ailab.rag import REFUSAL, apply_citation_policy, build_lab_rag
from ailab.streaming import (
    BUFFER_THEN_CHECK,
    PRINT_WHILE_CHECKING,
    answer_streaming,
    split_draft,
)

HALF = "游泳池全天开放"
UNKNOWN_DRAFT = "游泳池全天开放。来源：[faq-pool] 而且免费。"
MISSING_DRAFT = "下午 14:00 可以入住，不用看来源。"
KEPT_DRAFT = "标准入住时间是下午 14:00。来源：[faq-checkin]"
UNCLOSED_DRAFT = "游泳池全天开放。来源：[faq-pool"
MIXED_DRAFT = "标准入住时间是下午 14:00。来源：[faq-checkin] 游泳池全天开放。[faq-pool]"
CHECKIN = "入住时间是什么时候？"


class StubClient:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    def complete(self, messages, *, temperature: float = 0.0) -> Completion:
        del messages, temperature
        self.calls += 1
        return Completion(text=self.text, model="stub", prompt_tokens=1, completion_tokens=1)


def _pair(draft: str, question: str = CHECKIN):
    client = StubClient(draft)
    pipeline = build_lab_rag(client)
    buffered = answer_streaming(pipeline, question, BUFFER_THEN_CHECK)
    printed = answer_streaming(pipeline, question, PRINT_WHILE_CHECKING)
    hits = pipeline.retrieve(question)
    strong = [hit for hit in hits if hit.score >= pipeline.min_score]
    policy = apply_citation_policy(draft, strong, hits)
    return client, buffered, printed, policy


def test_split_draft_lets_the_half_sentence_arrive_before_the_citation() -> None:
    """句号先断开，完整的 [faq-...] 单独成段。半句因此比引用先到。"""

    assert split_draft(UNKNOWN_DRAFT) == [
        "游泳池全天开放。",
        "来源：",
        "[faq-pool]",
        " 而且免费。",
    ]
    assert split_draft(KEPT_DRAFT)[0] == "标准入住时间是下午 14:00。"
    assert "[faq-checkin]" not in split_draft(KEPT_DRAFT)[0]
    assert split_draft(KEPT_DRAFT)[-1] == "[faq-checkin]"


def test_discarded_half_is_visible_only_while_printing_and_not_in_the_final_answer() -> None:
    """合约：先收齐时客人看不到作废半句；边打印时中途看得到，最终回答里不留。"""

    _client, buffered, printed, policy = _pair(UNKNOWN_DRAFT)

    assert policy.reason == "unknown_citation"
    assert policy.answer == REFUSAL
    assert HALF not in policy.answer

    assert buffered.strategy == BUFFER_THEN_CHECK
    assert buffered.streamed == []
    assert buffered.final_answer == REFUSAL
    assert buffered.reason == "unknown_citation"
    assert buffered.refused
    assert buffered.citations == []
    assert HALF not in buffered.final_answer
    assert "而且免费" not in buffered.final_answer
    assert all(HALF not in piece for piece in buffered.streamed)

    assert printed.strategy == PRINT_WHILE_CHECKING
    assert printed.streamed[0] == "游泳池全天开放。"
    assert HALF in "".join(printed.streamed)
    assert "[faq-pool]" in printed.streamed
    assert "而且免费" not in "".join(printed.streamed)
    assert printed.final_answer == REFUSAL
    assert printed.reason == "unknown_citation"
    assert printed.refused
    assert printed.citations == []
    assert HALF not in printed.final_answer
    assert "而且免费" not in printed.final_answer
    assert "faq-pool" not in printed.final_answer
    assert printed.final_answer == buffered.final_answer
    assert printed.draft == UNKNOWN_DRAFT
    assert buffered.draft == UNKNOWN_DRAFT


def test_missing_citation_flashes_while_printing_then_the_final_answer_is_cleaned() -> None:
    """没有完整引用时，要等到流结束才作废。边打印会先把半句推上屏幕。"""

    _client, buffered, printed, policy = _pair(MISSING_DRAFT)
    assert policy.reason == "missing_citation"

    assert buffered.streamed == []
    assert buffered.final_answer == REFUSAL
    assert buffered.reason == "missing_citation"
    assert "14:00" not in buffered.final_answer
    assert "不用看来源" not in buffered.final_answer

    assert "14:00" in "".join(printed.streamed)
    assert "不用看来源" in "".join(printed.streamed)
    assert "".join(printed.streamed) == MISSING_DRAFT
    assert printed.final_answer == REFUSAL
    assert printed.reason == "missing_citation"
    assert "14:00" not in printed.final_answer
    assert "不用看来源" not in printed.final_answer


def test_unclosed_bracket_is_not_a_citation_yet() -> None:
    """方括号没闭合时不能当成未知引用，也不能提前停。流结束按缺少引用作废。"""

    _client, buffered, printed, policy = _pair(UNCLOSED_DRAFT)
    assert policy.reason == "missing_citation"
    assert printed.reason == "missing_citation"
    assert buffered.reason == "missing_citation"
    assert "".join(printed.streamed) == UNCLOSED_DRAFT
    assert HALF in "".join(printed.streamed)
    assert printed.final_answer == REFUSAL
    assert HALF not in printed.final_answer
    assert buffered.streamed == []
    assert HALF not in buffered.final_answer


def test_one_unknown_citation_also_clears_the_true_half_from_the_final_answer() -> None:
    """整段作废。边打印中途可以看见 14:00 和假话，最终回答里两句都不留。"""

    _client, buffered, printed, policy = _pair(MIXED_DRAFT)
    assert policy.reason == "unknown_citation"
    assert "14:00" not in policy.answer

    assert "14:00" in "".join(printed.streamed)
    assert HALF in "".join(printed.streamed)
    assert "[faq-checkin]" in printed.streamed
    assert printed.final_answer == REFUSAL
    assert "14:00" not in printed.final_answer
    assert HALF not in printed.final_answer

    assert buffered.streamed == []
    assert buffered.final_answer == REFUSAL
    assert "14:00" not in buffered.final_answer
    assert HALF not in buffered.final_answer


def test_kept_draft_stays_in_the_final_answer_of_both_strategies() -> None:
    """引用成立时不清洗。先收齐只在检查后给出整段；边打印会先给出半句。"""

    _client, buffered, printed, policy = _pair(KEPT_DRAFT)
    assert policy.reason == "ok"
    assert policy.answer == KEPT_DRAFT

    assert buffered.streamed == []
    assert buffered.reason == "ok"
    assert buffered.refused is False
    assert buffered.citations == ["faq-checkin"]
    assert buffered.final_answer == KEPT_DRAFT
    assert "14:00" in buffered.final_answer

    assert printed.reason == "ok"
    assert printed.citations == ["faq-checkin"]
    assert printed.final_answer == KEPT_DRAFT
    assert printed.streamed[0] == "标准入住时间是下午 14:00。"
    assert "[faq-checkin]" not in printed.streamed[0]
    assert printed.streamed[-1] == "[faq-checkin]"
    assert "".join(printed.streamed) == KEPT_DRAFT
    assert "14:00" in printed.final_answer


def test_low_score_does_not_stream_or_call_the_model() -> None:
    client = StubClient(UNKNOWN_DRAFT)
    pipeline = build_lab_rag(client)
    for strategy in (BUFFER_THEN_CHECK, PRINT_WHILE_CHECKING):
        result = answer_streaming(pipeline, "游泳池几点开放？", strategy)
        assert result.reason == "low_score"
        assert result.refused
        assert result.final_answer == REFUSAL
        assert result.streamed == []
        assert result.draft == ""
        assert result.citations == []
        assert HALF not in result.final_answer
    assert client.calls == 0


def test_lesson_05_answer_still_drops_the_whole_draft() -> None:
    """第 05 课不流式的默认路径仍是收齐后整段作废，不把半句留在回答里。"""

    result = build_lab_rag(StubClient(UNKNOWN_DRAFT)).answer(CHECKIN)
    assert result.reason == "unknown_citation"
    assert result.answer == REFUSAL
    assert HALF not in result.answer
    assert "而且免费" not in result.answer


def test_unknown_strategy_is_rejected() -> None:
    pipeline = build_lab_rag(StubClient(KEPT_DRAFT))
    with pytest.raises(ValueError):
        answer_streaming(pipeline, CHECKIN, "print-everything")
