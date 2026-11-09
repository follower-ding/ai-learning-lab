"""把岸港酒店短问答装进内存向量库。短问答整段入库，长规章才切块。"""

from ailab.corpus import FaqItem, load_faq
from ailab.embeddings import TfidfEmbedder
from ailab.vectorstore import MemoryVectorStore


def document_text(item: FaqItem) -> str:
    """标题和正文一起参与向量，存进库里的原文仍然只保留正文。"""

    return f"{item.title}\n{item.text}"


def build_faq_index(
    items: list[FaqItem] | None = None,
) -> tuple[TfidfEmbedder, MemoryVectorStore]:
    items = load_faq() if items is None else items
    embedder = TfidfEmbedder()
    texts = [document_text(item) for item in items]
    embedder.fit(texts)
    vectors = embedder.embed(texts)
    store = MemoryVectorStore()
    for item, vector in zip(items, vectors, strict=True):
        store.add(
            item.id,
            item.text,
            vector,
            {"source_id": item.id, "title": item.title},
        )
    return embedder, store
