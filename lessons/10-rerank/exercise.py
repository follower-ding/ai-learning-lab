"""第 10 课练习：先宽取 20 条，再只对这一批重打分，留下 4 条交给第 05 课。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ailab.client import MockChatClient
from ailab.indexing import build_faq_index
from ailab.messages import Completion
from ailab.rag import REFUSAL
from ailab.rerank import (
    KEEP_K,
    WIDE_K,
    answer_with_rerank,
    build_cutoff_shelf,
    passage_precision,
    rerank,
)


class CitesPets:
    """引用宽取里有、但第二次没有留下的 id。"""

    def complete(self, messages, *, temperature: float = 0.0) -> Completion:
        del messages, temperature
        return Completion(
            text="宠物可以入住。来源：[faq-pets]",
            model="misleading-stub",
            prompt_tokens=1,
            completion_tokens=1,
        )


def main() -> int:
    print("第 10 课 · 重排序")
    print(
        f"先按第 04 课的余弦宽取 {WIDE_K} 条，再只对这一批算占比，留下 {KEEP_K} 条交给第 05 课。"
        "占比是问句二元组占正文二元组的比例。"
    )
    if _faq_demo() != 0:
        return 1
    if _shelf_demo() != 0:
        return 1
    return 0


def _faq_demo() -> int:
    embedder, store = build_faq_index()
    question = "入住时间是什么时候？"
    result = answer_with_rerank(question, store=store, embedder=embedder)
    print(f"短问答有 {len(store)} 条。向它要 {WIDE_K} 条，拿回 {len(result.wide)} 条。")
    print("第一次，按余弦：")
    for hit in result.wide:
        title = hit.metadata.get("title", "")
        print(f"  余弦 {hit.score:.3f}  占比 {passage_precision(question, hit.text):.3f}  {hit.id}  {title}")
    print(f"第二次，只留下 {len(result.kept)} 条：")
    for hit in result.kept:
        print(f"  占比 {passage_precision(question, hit.text):.3f}  余弦 {hit.score:.3f}  {hit.id}")

    wide_ids = [hit.id for hit in result.wide]
    kept_ids = [hit.id for hit in result.kept]
    if kept_ids == wide_ids[:KEEP_K]:
        print("顺序没有变。第二次打分没有生效。")
        return 1
    if not set(kept_ids) <= set(wide_ids):
        print("留下的 id 里有宽取阶段没有的文档。")
        return 1
    if "faq-pets" in kept_ids or "faq-cancel" not in kept_ids:
        print("这一问上，宠物规章应当让出前四，取消规章应当进来。")
        return 1
    print("顺序变了。留下的 id 都来自第一次的候选。")

    print(f"交给第 05 课之后：原因 {result.answer.reason}  回答：{result.answer.answer}")
    if result.answer.reason != "ok" or result.answer.citations != ["faq-checkin"]:
        print("过线的应当仍是 faq-checkin。")
        return 1

    quiet = MockChatClient()
    refused = answer_with_rerank("游泳池几点开放？", client=quiet)
    print(f"游泳池这一问，留下的 {len(refused.kept)} 条余弦都不够，原因：{refused.answer.reason}")
    if (
        refused.answer.reason != "low_score"
        or refused.answer.answer != REFUSAL
        or quiet.calls
        or len(refused.kept) != KEEP_K
    ):
        print("低分仍调用了模型，或没有使用第 05 课的拒绝句。")
        return 1

    leaked = answer_with_rerank(question, client=CitesPets())
    print(f"草稿引用了 faq-pets。它在宽取结果里，不在留下的 {KEEP_K} 条里。原因：{leaked.answer.reason}")
    print(f"回答：{leaked.answer.answer}")
    if (
        leaked.answer.reason != "unknown_citation"
        or leaked.answer.answer != REFUSAL
        or "宠物可以入住" in leaked.answer.answer
    ):
        print("宽取里有、但没留下的 id 漏进了最终回答。")
        return 1
    return 0


def _shelf_demo() -> int:
    shelf = build_cutoff_shelf()
    result = answer_with_rerank(shelf.question, store=shelf.store, embedder=shelf.embedder)
    ranked = shelf.store.search(shelf.embedder.embed([shelf.question])[0], k=len(shelf.store))
    rank = [hit.id for hit in ranked].index(shelf.trap_id) + 1
    trap_precision = passage_precision(shelf.question, shelf.trap_text)
    best_wide = max(passage_precision(shelf.question, hit.text) for hit in result.wide)
    print(
        f"另有一个 {len(shelf.store)} 条的货架，不是酒店规章。"
        f"只写「{shelf.trap_text}」的 {shelf.trap_id} 占比 {trap_precision:.3f}，"
        f"余弦排在第 {rank} 名。"
    )
    print(
        f"宽取 {len(result.wide)} 条，里面最高占比是 {best_wide:.3f}。"
        f"留下：{' '.join(hit.id for hit in result.kept)}。"
    )
    if shelf.trap_id in {hit.id for hit in result.wide} or shelf.trap_id in {hit.id for hit in result.kept}:
        print("第二次打分把宽取之外的 id 造出来了。")
        return 1
    if not {hit.id for hit in result.kept} <= {hit.id for hit in result.wide}:
        print("留下的 id 不全在这次宽取的 20 条里。")
        return 1
    if trap_precision <= best_wide:
        print("货架没有把更高的占比留在前 20 名之外。")
        return 1
    promoted = rerank(shelf.question, ranked, k=KEEP_K)
    print(f"同一个函数如果拿到整库，第一条会变成 {promoted[0].id}。宽取挡住了它。")
    if promoted[0].id != shelf.trap_id:
        print("整库重排没有把占比最高的卡片放在第一，演示看不出差别。")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
