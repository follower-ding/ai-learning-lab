"""第 03 课练习：切块，并用同一套向量比较三句问法。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ailab.chunking import chunk_text
from ailab.corpus import load_faq, load_policy
from ailab.embeddings import TfidfEmbedder, cosine
from ailab.indexing import document_text


def main() -> int:
    print("第 03 课 · 切块与向量")
    chunks = chunk_text(load_policy(), size=80, overlap=20)
    print(f"规章被切成 {len(chunks)} 块（size=80，overlap=20）")
    for chunk in chunks[:3]:
        preview = chunk.text.replace("\n", "")[:24]
        print(f"  #{chunk.index} 长度 {len(chunk.text)}  {preview}…")

    items = load_faq()
    embedder = TfidfEmbedder()
    embedder.fit([document_text(item) for item in items])
    by_id = {
        item.id: vector
        for item, vector in zip(items, embedder.embed([document_text(item) for item in items]), strict=True)
    }
    checkin = by_id["faq-checkin"]
    probes = (
        "入住时间是什么时候？",
        "我什么时候能进房间",
        "今晚有乐队演出吗",
    )
    print(f"向量维度：{embedder.dimension}")
    for question in probes:
        score = cosine(embedder.embed([question])[0], checkin)
        print(f"  与入住规章的余弦 {score:.3f}  {question}")
    print("共用字面的句子分数最高。换一种说法会掉一截。完全无关的句子是 0。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
