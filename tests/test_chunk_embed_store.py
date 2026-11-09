"""第 03、04 课：切块、字面向量、内存库。"""

import pytest

from ailab.chunking import chunk_text
from ailab.corpus import load_policy
from ailab.embeddings import TfidfEmbedder, cosine
from ailab.errors import VectorDimensionError
from ailab.indexing import build_faq_index
from ailab.rag import DEFAULT_MIN_SCORE
from ailab.vectorstore import MemoryVectorStore


def test_short_paragraphs_stay_whole() -> None:
    chunks = chunk_text("第一段。\n\n第二段。", size=80, overlap=10)
    assert [chunk.text for chunk in chunks] == ["第一段。", "第二段。"]


def test_overlap_keeps_the_boundary_in_both_windows() -> None:
    text = "甲" * 100
    chunks = chunk_text(text, size=30, overlap=10)
    assert chunks[0].text == "甲" * 30
    assert chunks[1].text.startswith("甲" * 10)
    assert chunks[0].text[-10:] == chunks[1].text[:10]


def test_snaps_to_punctuation_near_the_window_end() -> None:
    text = ("甲" * 20) + "。" + ("乙" * 40)
    chunks = chunk_text(text, size=30, overlap=5)
    assert chunks[0].text.endswith("。")
    assert "乙" in chunks[1].text


def test_rejects_overlap_that_would_stall() -> None:
    with pytest.raises(ValueError):
        chunk_text("甲" * 50, size=20, overlap=20)


def test_policy_chunks_respect_the_window() -> None:
    chunks = chunk_text(load_policy(), size=80, overlap=20)
    assert len(chunks) >= 7
    assert all(len(chunk.text) <= 80 for chunk in chunks)


def test_embedder_requires_fit_and_identical_text_has_cosine_one() -> None:
    embedder = TfidfEmbedder()
    with pytest.raises(RuntimeError):
        embedder.embed(["入住"])
    embedder.fit(["入住时间是下午"])
    vector = embedder.embed(["入住时间是下午"])[0]
    assert cosine(vector, vector) == pytest.approx(1.0)
    assert embedder.dimension == len(vector)


def test_literal_overlap_beats_an_unrelated_question() -> None:
    embedder, store = build_faq_index()
    checkin = store.search(embedder.embed(["入住时间是什么时候？"])[0], k=3)
    parking = store.search(embedder.embed(["停车在哪里？"])[0], k=1)
    breakfast = store.search(embedder.embed(["早餐时间是什么时候？"])[0], k=1)
    band = store.search(embedder.embed(["今晚有乐队演出吗"])[0], k=1)
    pool = store.search(embedder.embed(["游泳池几点开放？"])[0], k=1)

    assert checkin[0].id == "faq-checkin"
    assert parking[0].id == "faq-parking"
    assert breakfast[0].id == "faq-breakfast"
    assert checkin[0].score >= DEFAULT_MIN_SCORE
    assert band[0].score < DEFAULT_MIN_SCORE
    assert pool[0].score < DEFAULT_MIN_SCORE
    assert band[0].score < checkin[0].score


def test_store_rejects_duplicate_ids_and_mixed_dimensions() -> None:
    store = MemoryVectorStore()
    assert store.search([1.0, 0.0], k=1) == []
    store.add("faq-checkin", "下午入住", [1.0, 0.0], {"source_id": "faq-checkin"})
    with pytest.raises(ValueError):
        store.add("faq-checkin", "另一条", [0.0, 1.0])
    with pytest.raises(VectorDimensionError):
        store.add("faq-wifi", "网络", [1.0, 0.0, 0.0])
    with pytest.raises(VectorDimensionError):
        store.search([1.0], k=1)


def test_equal_scores_tie_break_by_id() -> None:
    store = MemoryVectorStore()
    store.add("b", "乙", [1.0, 0.0])
    store.add("a", "甲", [1.0, 0.0])
    hits = store.search([1.0, 0.0], k=2)
    assert [hit.id for hit in hits] == ["a", "b"]
