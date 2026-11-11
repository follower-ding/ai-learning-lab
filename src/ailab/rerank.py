"""先宽取，再只对这一批候选做第二次打分。

第 04 课的库按余弦取前几条。资料变多以后，先取 20 条，再用占比留下 4 条，
交给第 05 课的阈值和引用检查。占比只根据这次拿回来的正文来算。

占比和余弦不是同一把尺子。``Hit.score`` 仍是第 04 课的余弦，第 05 课的
``DEFAULT_MIN_SCORE`` 比的是这个数，不要拿占比去比。
"""

from collections.abc import Sequence
from dataclasses import dataclass

from ailab.client import ChatClient, MockChatClient
from ailab.embeddings import Embedder, TfidfEmbedder, char_bigrams
from ailab.indexing import build_faq_index
from ailab.rag import (
    DEFAULT_MIN_SCORE,
    REFUSAL,
    RagAnswer,
    apply_citation_policy,
    build_rag_messages,
)
from ailab.vectorstore import Hit, MemoryVectorStore

# 宽取的条数，以及交给第 05 课的条数。库里不足 20 条时，第一次会少拿，不是失败。
WIDE_K = 20
KEEP_K = 4

# 21 个互不相同的字，连成 20 个只在这一问里出现的二元组，再接上「时间」。
# 短卡片各抄一个稀有二元组，余弦靠前；只写「时间」的那张占比最高，余弦却在前 20 之外。
_RARE_CHARS = "甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉戌"


@dataclass(frozen=True)
class CutoffShelf:
    """比 20 更大的内存库，用来看清第 21 名不会被第二次打分造出来。

    卡片不是岸港酒店的规章，也不替换第 04 课的短问答。
    """

    question: str
    trap_id: str
    trap_text: str
    embedder: TfidfEmbedder
    store: MemoryVectorStore


@dataclass(frozen=True)
class RerankResult:
    wide: list[Hit]
    kept: list[Hit]
    answer: RagAnswer


def passage_precision(question: str, text: str) -> float:
    """问句的二元组，占这段正文二元组的比例。只看这一段，不看别的文档。

    正文没有二元组，或问句没有二元组时，记为 0。比例到不了 1 以上。
    """

    query_grams = set(char_bigrams(question))
    passage_grams = set(char_bigrams(text))
    if not query_grams or not passage_grams:
        return 0.0
    return len(query_grams & passage_grams) / len(passage_grams)


def rerank(question: str, candidates: Sequence[Hit], k: int = KEEP_K) -> list[Hit]:
    """按占比重排 ``candidates``，留下 ``k`` 条。

    排序键是占比、第一次的余弦、id。函数只读这次传入的列表，没有向量库参数。
    """

    if k <= 0:
        raise ValueError("k 必须是正整数")
    ranked = sorted(
        candidates,
        key=lambda hit: (-passage_precision(question, hit.text), -hit.score, hit.id),
    )
    return ranked[:k]


def retrieve_wide(
    store: MemoryVectorStore,
    embedder: Embedder,
    question: str,
    k: int = WIDE_K,
) -> list[Hit]:
    """用建库的那个 embedder 做第 04 课的余弦检索，一次取 ``k`` 条。"""

    if k <= 0:
        raise ValueError("k 必须是正整数")
    vector = embedder.embed([question])[0]
    return store.search(vector, k=k)


def answer_with_rerank(
    question: str,
    *,
    store: MemoryVectorStore | None = None,
    embedder: Embedder | None = None,
    client: ChatClient | None = None,
    wide_k: int = WIDE_K,
    keep_k: int = KEEP_K,
    min_score: float = DEFAULT_MIN_SCORE,
) -> RerankResult:
    """宽取，重排，再把留下的几条交给第 05 课。

    不传库时用第 04 课的短问答。传了库就要带上建库的 embedder，两套向量不能混。
    第 05 课的默认 ``RagPipeline`` 仍是自己检索 top-3，这里不改那条路径。
    """

    if (store is None) != (embedder is None):
        raise ValueError("store 和 embedder 要一起提供。查询向量必须来自建库时的那一个 embedder。")
    if min_score < 0:
        raise ValueError("min_score 不能是负数")
    if store is None or embedder is None:
        embedder, store = build_faq_index()
    client = client or MockChatClient()
    wide = retrieve_wide(store, embedder, question, k=wide_k)
    kept = rerank(question, wide, k=keep_k)
    return RerankResult(
        wide=wide,
        kept=kept,
        answer=_hand_to_lesson_05(question, kept, client, min_score),
    )


def build_cutoff_shelf() -> CutoffShelf:
    """装一个超过 20 条的库。占比最高的卡片在余弦的前 20 名之外。"""

    rare_grams = char_bigrams(_RARE_CHARS)
    if len(rare_grams) != WIDE_K:
        raise RuntimeError("稀有字的二元组数量必须等于宽取条数。")
    question = _RARE_CHARS + "时间"
    trap_id = "shelf-time"
    trap_text = "时间"
    ids: list[str] = []
    texts: list[str] = []
    for index, gram in enumerate(rare_grams):
        ids.append(f"near-{index:02d}")
        texts.append(gram + "外")
    for index in range(12):
        ids.append(f"common-{index:02d}")
        texts.append(f"时间杂记{index:02d}岸港日常")
    ids.append(trap_id)
    texts.append(trap_text)
    for index in range(4):
        ids.append(f"zero-{index:02d}")
        texts.append(f"游泳池乐队{index}")

    embedder = TfidfEmbedder().fit(texts)
    store = MemoryVectorStore()
    for doc_id, text, vector in zip(ids, texts, embedder.embed(texts), strict=True):
        store.add(doc_id, text, vector)
    shelf = CutoffShelf(
        question=question,
        trap_id=trap_id,
        trap_text=trap_text,
        embedder=embedder,
        store=store,
    )
    _require_trap_outside_wide_batch(shelf)
    return shelf


def _hand_to_lesson_05(
    question: str,
    kept: Sequence[Hit],
    client: ChatClient,
    min_score: float,
) -> RagAnswer:
    """和第 05 课同一条规则：余弦不到阈值就不调用模型；引用必须落在过线的 id 上。"""

    strong = [hit for hit in kept if hit.score >= min_score]
    if not strong:
        return RagAnswer(
            answer=REFUSAL,
            citations=[],
            refused=True,
            reason="low_score",
            hits=list(kept),
        )
    draft = client.complete(build_rag_messages(question, strong)).text
    return apply_citation_policy(draft, strong, list(kept))


def _require_trap_outside_wide_batch(shelf: CutoffShelf) -> None:
    wide = shelf.store.search(shelf.embedder.embed([shelf.question])[0], k=WIDE_K)
    if len(wide) != WIDE_K or any(hit.id == shelf.trap_id for hit in wide):
        raise RuntimeError("演示货架没有把占比最高的卡片留在前 20 名之外。")
    best = max(passage_precision(shelf.question, hit.text) for hit in wide)
    if passage_precision(shelf.question, shelf.trap_text) <= best:
        raise RuntimeError("演示货架里，前 20 名已经有不低于陷阱的占比。")
