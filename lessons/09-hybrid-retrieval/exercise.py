"""第 09 课练习：关键词分和向量分加权，无关问法仍然拒绝。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ailab.hybrid import build_hybrid_retriever, format_hybrid_report, run_hybrid_eval
from ailab.rag import DEFAULT_MIN_SCORE, REFUSAL


def main() -> int:
    print("第 09 课 · 混合检索")
    retriever = build_hybrid_retriever()
    print(
        f"关键词是 BM25，向量是 TF-IDF 余弦。"
        f"融合 = {retriever.vector_weight:.1f}×向量 + {retriever.sparse_weight:.1f}×压缩后的关键词分。"
        f"阈值 {retriever.min_score:.2f}。"
    )
    print("第 03 课的三句问法，两路都留下：")
    probes = (
        "入住时间是什么时候？",
        "我什么时候能进房间",
        "今晚有乐队演出吗",
    )
    for question in probes:
        hit = retriever.search(question, k=1)[0]
        print(
            f"  向量 {hit.vector_score:.3f}  关键词原始 {hit.sparse_raw:.3f}"
            f"  关键词 {hit.sparse_score:.3f}  融合 {hit.score:.3f}  {hit.id}"
        )
        print(f"    {question}")
    band = retriever.search("今晚有乐队演出吗", k=1)[0]
    if band.vector_score == 0.0 and band.sparse_raw == 0.0 and band.score == 0.0:
        print("乐队这一问两路都是 0。全 0 时第一名只是 id 的字母序，融合没有把它抬过线。")
        print(f"不过线就不调用模型，回答用第 05 课的拒绝句：{REFUSAL}")

    generic = retriever.search("时间", k=1)[0]
    peaked = max(generic.vector_score, generic.sparse_score)
    print(
        f"「时间」向量 {generic.vector_score:.3f}（第 05 课的阈值是 {DEFAULT_MIN_SCORE:.1f}），"
        f"关键词 {generic.sparse_score:.3f}，加权 {generic.score:.3f}，取最大则是 {peaked:.3f}。"
    )
    if generic.vector_score >= DEFAULT_MIN_SCORE and generic.score < retriever.min_score <= peaked:
        print("只看向量，这一个字已经过线。取两路的最大值也会过线。加权把它留在阈值下面。")

    report = run_hybrid_eval(retriever=retriever)
    print(format_hybrid_report(report))
    return 0 if report.passed == report.total else 1


if __name__ == "__main__":
    raise SystemExit(main())
