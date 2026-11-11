"""第 10 课：宽取 20 条之后只重排这一批，留下 4 条，不能造出新的 id。"""

import inspect

import pytest

from ailab.indexing import build_faq_index
from ailab.messages import Completion
from ailab.rag import DEFAULT_MIN_SCORE, REFUSAL, RagPipeline
from ailab.rerank import (
    KEEP_K,
    WIDE_K,
    answer_with_rerank,
    build_cutoff_shelf,
    passage_precision,
    rerank,
    retrieve_wide,
)
from ailab.vectorstore import Hit


class StubClient:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    def complete(self, messages, *, temperature: float = 0.0) -> Completion:
        del messages, temperature
        self.calls += 1
        return Completion(text=self.text, model="stub", prompt_tokens=1, completion_tokens=1)


class SpyStore:
    """记下 search 的 k。重排要是回头把整库再搜一遍，这里的次数或 k 会对不上。"""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.ks: list[int] = []

    def search(self, vector, k: int = 3):
        self.ks.append(k)
        return self.inner.search(vector, k)


def test_rerank_changes_order_and_rejects_ids_absent_from_the_batch() -> None:
    """手写的一批候选。占比更高、但没放进这批的正文，不能出现在结果里。"""

    question = "入住时间是什么时候？"
    absent = "入住时间是什么时候"
    candidates = [
        Hit(
            id="draft-long",
            text="标准入住时间是下午。如果早到，前台可以免费寄存行李，房间需等到打扫完成才能进入，还要再写一句与问句无关的停车和网络说明。",
            score=0.80,
            metadata={},
        ),
        Hit(
            id="draft-pets",
            text="除导盲犬外，酒店不接受宠物入住。导盲犬需要在入住当天向前台出示证明。",
            score=0.50,
            metadata={},
        ),
        Hit(
            id="draft-cancel",
            text="入住日前可以免费取消。",
            score=0.40,
            metadata={},
        ),
        Hit(
            id="draft-parking",
            text="北侧地下停车场先到先得。",
            score=0.20,
            metadata={},
        ),
        Hit(
            id="draft-wifi",
            text="客房网络密码印在房卡套内侧。",
            score=0.10,
            metadata={},
        ),
    ]
    original_ids = [hit.id for hit in candidates]
    kept = rerank(question, candidates, k=KEEP_K)

    assert KEEP_K == 4
    assert len(kept) == 4
    assert [hit.id for hit in kept] == [
        "draft-cancel",
        "draft-long",
        "draft-pets",
        "draft-parking",
    ]
    assert [hit.id for hit in kept] != original_ids[:KEEP_K]
    assert {hit.id for hit in kept} <= set(original_ids)
    assert [hit.id for hit in candidates] == original_ids
    assert passage_precision(question, absent) > max(
        passage_precision(question, hit.text) for hit in kept
    )
    assert passage_precision(question, absent) == pytest.approx(1.0)


def test_faq_wide_retrieve_keeps_four_from_the_lesson_04_corpus() -> None:
    """第 04 课那 7 条：向库要 20 条，实际拿回全部，再留下 4 条。顺序要变，id 不能新造。"""

    embedder, store = build_faq_index()
    question = "入住时间是什么时候？"
    wide = retrieve_wide(store, embedder, question, k=WIDE_K)

    assert WIDE_K == 20
    assert len(store) == 7
    assert len(wide) == 7
    assert [hit.id for hit in wide[:KEEP_K]] == [
        "faq-checkin",
        "faq-checkout",
        "faq-breakfast",
        "faq-pets",
    ]

    kept = rerank(question, wide, k=KEEP_K)
    assert [hit.id for hit in kept] == [
        "faq-checkin",
        "faq-checkout",
        "faq-breakfast",
        "faq-cancel",
    ]
    assert "faq-pets" not in {hit.id for hit in kept}
    assert {hit.id for hit in kept} <= {hit.id for hit in wide}
    pets = next(hit for hit in wide if hit.id == "faq-pets")
    cancel = next(hit for hit in kept if hit.id == "faq-cancel")
    assert passage_precision(question, cancel.text) > passage_precision(question, pets.text)
    assert pets.score > cancel.score


def test_second_pass_does_not_read_past_the_wide_batch() -> None:
    """占比最高的那条余弦排在第 21。只重排前 20 时它不能出现；把整库交给同一个函数时它会排第一。"""

    shelf = build_cutoff_shelf()
    spy = SpyStore(shelf.store)
    result = answer_with_rerank(
        shelf.question,
        store=spy,
        embedder=shelf.embedder,
        client=StubClient("不应该因为读了整库就被调用"),
    )

    assert spy.ks == [WIDE_K]
    assert len(result.wide) == WIDE_K
    assert len(result.kept) == KEEP_K
    assert shelf.trap_id not in {hit.id for hit in result.wide}
    assert shelf.trap_id not in {hit.id for hit in result.kept}
    assert {hit.id for hit in result.kept} <= {hit.id for hit in result.wide}

    best_in_batch = max(passage_precision(shelf.question, hit.text) for hit in result.wide)
    assert passage_precision(shelf.question, shelf.trap_text) > best_in_batch

    everyone = shelf.store.search(shelf.embedder.embed([shelf.question])[0], k=len(shelf.store))
    outside = {hit.id for hit in everyone} - {hit.id for hit in result.wide}
    assert shelf.trap_id in outside
    assert len(everyone) == len(result.wide) + len(outside)
    assert rerank(shelf.question, everyone, k=KEEP_K)[0].id == shelf.trap_id


def test_kept_four_go_through_lesson_05_threshold_and_citation_check() -> None:
    cited = answer_with_rerank("入住时间是什么时候？")
    assert [hit.id for hit in cited.kept][0] == "faq-checkin"
    assert cited.answer.reason == "ok"
    assert cited.answer.citations == ["faq-checkin"]
    assert "14:00" in cited.answer.answer
    assert not cited.answer.refused

    quiet = StubClient("不应该被调用")
    refused = answer_with_rerank("游泳池几点开放？", client=quiet)
    assert refused.answer.reason == "low_score"
    assert refused.answer.answer == REFUSAL
    assert refused.answer.citations == []
    assert quiet.calls == 0
    assert len(refused.kept) == KEEP_K
    assert all(hit.score < DEFAULT_MIN_SCORE for hit in refused.kept)

    leaked = answer_with_rerank(
        "入住时间是什么时候？",
        client=StubClient("宠物可以入住。来源：[faq-pets]"),
    )
    assert "faq-pets" in {hit.id for hit in leaked.wide}
    assert "faq-pets" not in {hit.id for hit in leaked.kept}
    assert leaked.answer.reason == "unknown_citation"
    assert leaked.answer.answer == REFUSAL
    assert "宠物可以入住" not in leaked.answer.answer

    cancel_hit = next(hit for hit in leaked.kept if hit.id == "faq-cancel")
    assert cancel_hit.score < DEFAULT_MIN_SCORE
    below = answer_with_rerank(
        "入住时间是什么时候？",
        client=StubClient("24 小时以内取消要收费。来源：[faq-cancel]"),
    )
    assert below.answer.reason == "unknown_citation"
    assert below.answer.answer == REFUSAL
    assert "24 小时" not in below.answer.answer


def test_rerank_rejects_non_positive_k_and_does_not_require_the_store() -> None:
    with pytest.raises(ValueError):
        rerank("入住", [], k=0)
    with pytest.raises(ValueError):
        retrieve_wide(*build_faq_index(), "入住", k=0)

    embedder, store = build_faq_index()
    with pytest.raises(ValueError):
        answer_with_rerank("入住时间是什么时候？", store=store)

    punctuation = Hit(id="only", text="入住", score=0.2, metadata={})
    assert rerank("？？？", [punctuation], k=4) == [punctuation]
    assert passage_precision("？？？", "入住") == 0.0
    assert passage_precision("入住", "") == 0.0


def test_lesson_05_default_retrieve_width_stays_three() -> None:
    """第 10 课走旁边的管道。第 05 课自己的检索宽度仍是 3。"""

    assert inspect.signature(RagPipeline.retrieve).parameters["k"].default == 3
    assert inspect.signature(RagPipeline.answer).parameters["k"].default == 3
