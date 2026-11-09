"""检索之后：分数不够就拒绝；模型引用了没检索到的文档，也拒绝。"""

import re
from dataclasses import dataclass

from ailab.client import ChatClient, MockChatClient
from ailab.embeddings import Embedder
from ailab.indexing import build_faq_index
from ailab.messages import Message
from ailab.vectorstore import Hit, MemoryVectorStore

# 这份字面向量上，相关问题的最高分和无关问题之间有一道缝。
# 缝的位置只对当前资料和 TfidfEmbedder 有意义。换嵌入模型后要重新看分数。
DEFAULT_MIN_SCORE = 0.2

REFUSAL = "知识库里没有足够依据，不能回答。"
_CITATION = re.compile(r"\[(faq-[a-z0-9-]+)\]")


@dataclass(frozen=True)
class RagAnswer:
    answer: str
    citations: list[str]
    refused: bool
    reason: str
    hits: list[Hit]


class RagPipeline:
    """先检索，再决定要不要把资料交给模型。"""

    def __init__(
        self,
        store: MemoryVectorStore,
        embedder: Embedder,
        client: ChatClient,
        *,
        min_score: float = DEFAULT_MIN_SCORE,
    ) -> None:
        if min_score < 0:
            raise ValueError("min_score 不能是负数")
        self.store = store
        self.embedder = embedder
        self.client = client
        self.min_score = min_score

    def retrieve(self, question: str, k: int = 3) -> list[Hit]:
        vector = self.embedder.embed([question])[0]
        return self.store.search(vector, k=k)

    def answer(self, question: str, k: int = 3) -> RagAnswer:
        hits = self.retrieve(question, k=k)
        strong = [hit for hit in hits if hit.score >= self.min_score]
        if not strong:
            return RagAnswer(
                answer=REFUSAL,
                citations=[],
                refused=True,
                reason="low_score",
                hits=hits,
            )
        draft = self.client.complete(build_rag_messages(question, strong)).text
        return apply_citation_policy(draft, strong, hits)


def build_lab_rag(
    client: ChatClient | None = None,
    *,
    min_score: float = DEFAULT_MIN_SCORE,
) -> RagPipeline:
    embedder, store = build_faq_index()
    return RagPipeline(store, embedder, client or MockChatClient(), min_score=min_score)


def build_rag_messages(question: str, hits: list[Hit]) -> list[Message]:
    lines = []
    for hit in hits:
        source_id = hit.metadata.get("source_id", hit.id)
        title = hit.metadata.get("title", "")
        lines.append(f"- [{source_id}] {title}：{hit.text}")
    system = "\n".join(
        [
            "你是岸港酒店实验前台。只依据【检索资料】回答。",
            "回答的最后必须带上来源，格式是 [文档id]。",
            "资料里没有的数字、时间、价格，不要补充。",
            "【检索资料】",
            *lines,
        ]
    )
    return [Message("system", system), Message("user", question)]


def apply_citation_policy(draft: str, strong: list[Hit], hits: list[Hit]) -> RagAnswer:
    """引用必须落在这一次检索到、且过了阈值的文档里。否则整段草稿作废。"""

    allowed: list[str] = []
    for hit in strong:
        source_id = str(hit.metadata.get("source_id", hit.id))
        if source_id not in allowed:
            allowed.append(source_id)
    found: list[str] = []
    for source_id in _CITATION.findall(draft):
        if source_id not in found:
            found.append(source_id)
    if not found:
        return RagAnswer(REFUSAL, [], True, "missing_citation", hits)
    if any(source_id not in allowed for source_id in found):
        return RagAnswer(REFUSAL, [], True, "unknown_citation", hits)
    return RagAnswer(draft.strip(), found, False, "ok", hits)
