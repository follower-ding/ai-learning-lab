"""第 09 课：关键词与向量加权。无关问法不能因为多了一路就被抬过阈值。"""

import pytest

from ailab.hybrid import (
    BM25_SCALE,
    DEFAULT_HYBRID_MIN_SCORE,
    HybridQuery,
    build_hybrid_retriever,
    format_hybrid_report,
    load_hybrid_queries,
    run_hybrid_eval,
    squash_bm25,
)
from ailab.indexing import build_faq_index
from ailab.rag import DEFAULT_MIN_SCORE
from ailab.sparse import Bm25Index

LESSON_03_PROBES = (
    "入住时间是什么时候？",
    "我什么时候能进房间",
    "今晚有乐队演出吗",
)


def test_bm25_prefers_the_document_that_actually_contains_the_term() -> None:
    index = Bm25Index().fit(["a", "b"], ["入住时间", "早餐时间"])
    scores = index.raw_scores("入住")
    assert scores["a"] > scores["b"]
    assert scores["b"] == 0.0
    assert index.raw_scores("乐队") == {"a": 0.0, "b": 0.0}


def test_bm25_requires_fit_and_rejects_duplicate_ids() -> None:
    index = Bm25Index()
    with pytest.raises(RuntimeError):
        index.raw_scores("入住")
    with pytest.raises(ValueError):
        Bm25Index().fit(["faq-checkin", "faq-checkin"], ["入住", "再次入住"])
    with pytest.raises(ValueError):
        Bm25Index().fit([], [])


def test_squash_keeps_zero_and_never_reaches_one() -> None:
    assert squash_bm25(0.0) == 0.0
    assert squash_bm25(-1.0) == 0.0
    assert squash_bm25(BM25_SCALE) == pytest.approx(0.5)
    assert squash_bm25(100.0) < 1.0
    with pytest.raises(ValueError):
        squash_bm25(1.0, scale=0.0)


def test_weights_must_sum_to_one_and_k_must_be_positive() -> None:
    with pytest.raises(ValueError):
        build_hybrid_retriever(vector_weight=1.0, sparse_weight=1.0)
    with pytest.raises(ValueError):
        build_hybrid_retriever(vector_weight=-0.1, sparse_weight=1.1)
    retriever = build_hybrid_retriever()
    with pytest.raises(ValueError):
        retriever.search("入住", k=0)


def test_fixed_query_set_covers_lesson_03_and_passes() -> None:
    cases = load_hybrid_queries()
    questions = [case.question for case in cases]
    for probe in LESSON_03_PROBES:
        assert probe in questions
    assert any(case.expect == "hit" for case in cases)
    assert sum(case.expect == "refuse" for case in cases) >= 4

    report = run_hybrid_eval()
    assert report.passed == report.total == len(cases)
    assert "合计" in format_hybrid_report(report)

    broken = run_hybrid_eval(
        [HybridQuery("bad", "今晚有乐队演出吗", "hit", source_id="faq-checkin")]
    )
    assert broken.passed == 0
    assert broken.checks[0].passed is False


def test_paraphrase_sits_closer_to_the_line_on_vectors_than_after_fusion() -> None:
    retriever = build_hybrid_retriever()
    hit = retriever.search("我什么时候能进房间", k=1)[0]
    assert hit.id == "faq-checkin"
    assert DEFAULT_MIN_SCORE <= hit.vector_score < DEFAULT_HYBRID_MIN_SCORE
    assert hit.score >= retriever.min_score
    vector_margin = hit.vector_score - DEFAULT_MIN_SCORE
    fused_margin = hit.score - retriever.min_score
    assert fused_margin > vector_margin


def test_afternoon_wording_fails_the_vector_threshold_and_passes_hybrid() -> None:
    retriever = build_hybrid_retriever()
    hit = retriever.search("下午可以住吗", k=1)[0]
    assert hit.id == "faq-checkin"
    assert hit.vector_score < DEFAULT_MIN_SCORE
    assert hit.score >= retriever.min_score
    assert retriever.accepted("下午可以住吗")[0].id == "faq-checkin"


def test_band_and_pool_stay_at_zero_so_fusion_cannot_lift_them() -> None:
    retriever = build_hybrid_retriever()
    for question in ("今晚有乐队演出吗", "游泳池几点开放？"):
        hits = retriever.search(question, k=7)
        assert hits
        assert all(hit.vector_score == 0.0 for hit in hits)
        assert all(hit.sparse_raw == 0.0 for hit in hits)
        assert all(hit.score == 0.0 for hit in hits)
        assert retriever.accepted(question) == []


def test_generic_time_clears_the_vector_threshold_but_weighted_fusion_refuses() -> None:
    retriever = build_hybrid_retriever()
    best = retriever.search("时间", k=1)[0]
    assert best.vector_score >= DEFAULT_MIN_SCORE
    assert best.score < retriever.min_score
    peaked = max(best.vector_score, best.sparse_score)
    assert peaked >= retriever.min_score
    assert retriever.accepted("时间") == []


def test_vector_alone_would_cite_the_pet_rule_for_a_checkin_paraphrase() -> None:
    embedder, store = build_faq_index()
    hits = store.search(embedder.embed(["什么时候能入住"])[0], k=1)
    assert hits[0].id == "faq-pets"
    assert hits[0].score >= DEFAULT_MIN_SCORE

    retriever = build_hybrid_retriever()
    best = retriever.search("什么时候能入住", k=1)[0]
    assert best.id == "faq-pets"
    assert best.score < retriever.min_score
    assert retriever.accepted("什么时候能入住") == []


def test_dog_is_refused_because_neither_channel_sees_a_synonym() -> None:
    retriever = build_hybrid_retriever()
    best = retriever.search("可以带狗吗", k=1)[0]
    assert best.id != "faq-pets"
    assert best.score < retriever.min_score
    assert best.vector_score < DEFAULT_MIN_SCORE


def test_thin_overlap_stays_under_the_threshold() -> None:
    retriever = build_hybrid_retriever()
    best = retriever.search("早到怎么办", k=1)[0]
    assert best.id == "faq-checkin"
    assert best.sparse_raw > 0.0
    assert best.score < retriever.min_score


def test_empty_and_punctuation_queries_refuse_without_error() -> None:
    retriever = build_hybrid_retriever()
    for question in ("", "？？？", "   "):
        best = retriever.search(question, k=1)[0]
        assert best.score == 0.0
        assert retriever.accepted(question) == []


def test_vector_only_weight_drops_the_paraphrase_under_the_hybrid_threshold() -> None:
    """练习建议改的那一行：只留向量时，换一种说法不再过线。"""

    vector_only = build_hybrid_retriever(vector_weight=1.0, sparse_weight=0.0)
    hit = vector_only.search("我什么时候能进房间", k=1)[0]
    assert hit.score == pytest.approx(hit.vector_score)
    assert hit.score < vector_only.min_score
    report = run_hybrid_eval(retriever=vector_only)
    failed = {check.case_id for check in report.checks if not check.passed}
    assert "paraphrase-room" in failed


def test_literal_checkin_vector_matches_the_store_used_in_earlier_lessons() -> None:
    embedder, store = build_faq_index()
    question = "入住时间是什么时候？"
    from_store = store.search(embedder.embed([question])[0], k=1)[0]
    hybrid = build_hybrid_retriever().search(question, k=1)[0]
    assert hybrid.id == from_store.id == "faq-checkin"
    assert hybrid.vector_score == pytest.approx(from_store.score)


def test_a_real_keyword_still_clears_the_bar_when_the_question_is_about_something_else() -> None:
    """问句里夹了库内词。混合消不掉「游泳池」，停车规章会过线。

    正文里没有游泳池。这一课只锁住检索行为，不把它当成已经答对客人。
    过线之后仍要走第 05 课，读正文，再决定留还是拒绝。
    """

    hit = build_hybrid_retriever().search("游泳池在停车场吗", k=1)[0]
    assert hit.id == "faq-parking"
    assert hit.score >= DEFAULT_HYBRID_MIN_SCORE
    assert "游泳池" not in hit.text
