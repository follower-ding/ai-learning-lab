"""把关键词分和向量分留在明处，再按固定权重合成一个检索分。

向量分是余弦，落在 0 到 1。BM25 的原始分可以到好几分。两路不在同一把尺子上，
不能直接相加，也不能除以「这一问的最高分」——全库都很弱时，第一名会被抬成 1。

压缩用 ``raw / (raw + BM25_SCALE)``。``BM25_SCALE`` 和阈值只对当前这份短问答、
当前这套字面向量有意义。换资料或换嵌入模型之后要重新看分数，再决定阈值。
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass

from ailab.corpus import FaqItem, load_faq, read_json
from ailab.embeddings import Embedder, cosine
from ailab.indexing import build_faq_index, document_text
from ailab.sparse import Bm25Index

# 两路等权。改成只留向量（1 和 0）时，换一种说法的问句会回到第 03 课那个擦线的余弦。
VECTOR_WEIGHT = 0.5
SPARSE_WEIGHT = 0.5

# 原始 BM25 等于这个数时，压缩后是 0.5。只用于把关键词分拉进 (0, 1)，不是相关阈值。
BM25_SCALE = 2.5

# 压在一道缝上。当前这套查询里，应当留下的最低融合分是 0.341（「下午可以住吗」），
# 应当拒绝、又不是全 0 的最高融合分是 0.274（「早到怎么办」）。
# 0.30 只对当前资料和这一套权重有意义。
DEFAULT_HYBRID_MIN_SCORE = 0.30


@dataclass(frozen=True)
class HybridHit:
    id: str
    text: str
    score: float
    vector_score: float
    sparse_score: float
    sparse_raw: float
    metadata: Mapping[str, str]


@dataclass(frozen=True)
class HybridQuery:
    id: str
    question: str
    expect: str
    source_id: str = ""


@dataclass(frozen=True)
class HybridCheck:
    case_id: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class HybridReport:
    checks: list[HybridCheck]

    @property
    def passed(self) -> int:
        return sum(1 for check in self.checks if check.passed)

    @property
    def total(self) -> int:
        return len(self.checks)


def squash_bm25(raw: float, scale: float = BM25_SCALE) -> float:
    """把非负的 BM25 原始分压进 ``[0, 1)``。0 还是 0，原始分再大也到不了 1。"""

    if scale <= 0:
        raise ValueError("scale 必须是正数")
    if raw <= 0:
        return 0.0
    return raw / (raw + scale)


def _check_weights(vector_weight: float, sparse_weight: float) -> None:
    if vector_weight < 0 or sparse_weight < 0:
        raise ValueError("权重不能是负数")
    if not math.isclose(vector_weight + sparse_weight, 1.0, abs_tol=1e-9):
        raise ValueError("向量权重和关键词权重要加起来等于 1")


class HybridRetriever:
    """对库里每一条同时打两种分，再加权。资料只有几条，不做「先各取 top-k 再融合」。

    查询向量必须和第 04 课一样，来自构建文档向量的那一个 embedder。
    """

    def __init__(
        self,
        embedder: Embedder,
        documents: list[tuple[str, str, tuple[float, ...], dict[str, str]]],
        bm25: Bm25Index,
        *,
        vector_weight: float = VECTOR_WEIGHT,
        sparse_weight: float = SPARSE_WEIGHT,
        min_score: float = DEFAULT_HYBRID_MIN_SCORE,
    ) -> None:
        _check_weights(vector_weight, sparse_weight)
        if min_score < 0:
            raise ValueError("min_score 不能是负数")
        doc_ids = [doc_id for doc_id, _, _, _ in documents]
        if doc_ids != bm25.ids:
            raise ValueError("向量这边的文档和关键词索引必须是同一批 id、同一个顺序")
        self.embedder = embedder
        self._docs = documents
        self._bm25 = bm25
        self.vector_weight = vector_weight
        self.sparse_weight = sparse_weight
        self.min_score = min_score

    def search(self, question: str, k: int = 3) -> list[HybridHit]:
        if k <= 0:
            raise ValueError("k 必须是正整数")
        if not self._docs:
            return []
        query_vector = self.embedder.embed([question])[0]
        raw_scores = self._bm25.raw_scores(question)
        ranked: list[HybridHit] = []
        for doc_id, text, vector, metadata in self._docs:
            vector_score = cosine(query_vector, vector)
            sparse_raw = raw_scores[doc_id]
            sparse_score = squash_bm25(sparse_raw)
            fused = self.vector_weight * vector_score + self.sparse_weight * sparse_score
            ranked.append(
                HybridHit(
                    id=doc_id,
                    text=text,
                    score=fused,
                    vector_score=vector_score,
                    sparse_score=sparse_score,
                    sparse_raw=sparse_raw,
                    metadata=metadata,
                )
            )
        ranked.sort(key=lambda hit: (-hit.score, hit.id))
        return ranked[:k]

    def accepted(self, question: str, k: int = 3) -> list[HybridHit]:
        """融合分达到阈值的命中。空列表表示应当拒绝，并且不要调用模型。"""

        return [hit for hit in self.search(question, k=k) if hit.score >= self.min_score]


def build_hybrid_retriever(
    items: list[FaqItem] | None = None,
    *,
    vector_weight: float = VECTOR_WEIGHT,
    sparse_weight: float = SPARSE_WEIGHT,
    min_score: float = DEFAULT_HYBRID_MIN_SCORE,
) -> HybridRetriever:
    """用岸港酒店短问答建混合检索。标题参加打分，存下来的原文仍是正文。"""

    items = load_faq() if items is None else items
    embedder, _store = build_faq_index(items)
    texts = [document_text(item) for item in items]
    vectors = embedder.embed(texts)
    documents = [
        (
            item.id,
            item.text,
            tuple(vector),
            {"source_id": item.id, "title": item.title},
        )
        for item, vector in zip(items, vectors, strict=True)
    ]
    bm25 = Bm25Index().fit([item.id for item in items], texts)
    return HybridRetriever(
        embedder,
        documents,
        bm25,
        vector_weight=vector_weight,
        sparse_weight=sparse_weight,
        min_score=min_score,
    )


def load_hybrid_queries() -> list[HybridQuery]:
    cases = []
    for row in read_json("hybrid_queries.json"):
        expect = row["expect"]
        if expect not in {"hit", "refuse"}:
            raise ValueError(f"expect 无法识别: {expect}")
        source_id = row.get("source_id", "")
        if expect == "hit" and not source_id:
            raise ValueError(f"{row['id']} 期望命中，但没有 source_id")
        cases.append(
            HybridQuery(
                id=row["id"],
                question=row["question"],
                expect=expect,
                source_id=source_id,
            )
        )
    return cases


def run_hybrid_eval(
    cases: list[HybridQuery] | None = None,
    retriever: HybridRetriever | None = None,
) -> HybridReport:
    cases = load_hybrid_queries() if cases is None else cases
    retriever = build_hybrid_retriever() if retriever is None else retriever
    checks = [_grade(case, retriever) for case in cases]
    return HybridReport(checks)


def format_hybrid_report(report: HybridReport) -> str:
    lines = []
    for check in report.checks:
        mark = "通过" if check.passed else "失败"
        lines.append(f"{check.case_id}  {mark}  {check.detail}")
    lines.append(f"合计 {report.passed}/{report.total}")
    return "\n".join(lines)


def _grade(case: HybridQuery, retriever: HybridRetriever) -> HybridCheck:
    hits = retriever.search(case.question, k=3)
    if not hits:
        return HybridCheck(case.id, False, "没有文档")
    best = hits[0]
    detail = (
        f"fused={best.score:.3f} vector={best.vector_score:.3f} "
        f"sparse={best.sparse_score:.3f} raw={best.sparse_raw:.3f} id={best.id}"
    )
    if case.expect == "hit":
        passed = best.score >= retriever.min_score and best.id == case.source_id
        return HybridCheck(case.id, passed, detail)
    if case.expect == "refuse":
        return HybridCheck(case.id, best.score < retriever.min_score, detail)
    return HybridCheck(case.id, False, detail + " expect 无法识别")
