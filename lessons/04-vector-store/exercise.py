"""第 04 课练习：把短问答放进内存向量库，取出分数最高的几条。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ailab.indexing import build_faq_index


def main() -> int:
    print("第 04 课 · 向量库")
    embedder, store = build_faq_index()
    print(f"库里有 {len(store)} 条，维度 {store.dimension}")
    questions = (
        "入住时间是什么时候？",
        "早餐时间是什么时候？",
        "游泳池几点开放？",
    )
    for question in questions:
        print(question)
        hits = store.search(embedder.embed([question])[0], k=3)
        for hit in hits:
            title = hit.metadata.get("title", "")
            print(f"  {hit.score:.3f}  {hit.id}  {title}")
    print("低分仍然留在 top-k 里。下一课再用阈值决定拒不拒绝。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
