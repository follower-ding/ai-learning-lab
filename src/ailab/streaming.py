"""第 12 课：同一段草稿，两种送到屏幕上的办法。

第 05 课的 ``apply_citation_policy`` 要看到完整的 ``[faq-...]`` 才决定留还是扔。
这里不改 ``RagPipeline.answer``，也不改第 06 课到第 08 课的工具协议。

离线不打开流式 HTTP。``split_draft`` 把 ``complete`` 的整段按句号和方括号切开，
当作逐段到达。两种办法的最终回答都等于引用检查的结果，作废的半句不留下：

- ``buffer_then_check``：检查完成前不往屏幕上推。半句不会出现。
- ``print_while_checking``：每一段先推上屏幕。半句比完整引用先到，中途看得到。
  未知引用一旦闭合，后面的段不再推。最终回答仍换成拒绝语。
"""

from dataclasses import dataclass

from ailab.rag import REFUSAL, RagPipeline, apply_citation_policy, build_rag_messages
from ailab.vectorstore import Hit

BUFFER_THEN_CHECK = "buffer_then_check"
PRINT_WHILE_CHECKING = "print_while_checking"

STRATEGY_LABEL = {
    BUFFER_THEN_CHECK: "先收齐再检查",
    PRINT_WHILE_CHECKING: "边打印边检查",
}


@dataclass(frozen=True)
class StreamingAnswer:
    """一次流式作答。

    ``streamed`` 是检查完成前已经推到屏幕上的段，按到达顺序。
    ``final_answer`` 是留给客人的最终回答。它和第 05 课的检查结果相同。
    ``draft`` 是模型返回的整段，只给检查用。分数不够时没有草稿，这里是空字符串。
    """

    strategy: str
    streamed: list[str]
    final_answer: str
    refused: bool
    reason: str
    citations: list[str]
    hits: list[Hit]
    draft: str


def split_draft(text: str) -> list[str]:
    """把整段草稿切成离线的逐段。不联网。

    句号（。）结束一段。左方括号另起一段，直到右方括号，引用因此单独成段。
    没有右方括号时，从左方括号起剩下的字单独成段，不把它当成一条已经写完的引用。
    """

    if text == "":
        return []
    parts: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        if buf:
            parts.append("".join(buf))
            buf.clear()

    index = 0
    while index < len(text):
        char = text[index]
        if char == "[":
            flush()
            end = text.find("]", index)
            if end == -1:
                parts.append(text[index:])
                return parts
            parts.append(text[index : end + 1])
            index = end + 1
            continue
        buf.append(char)
        index += 1
        if char == "。":
            flush()
    flush()
    return parts


def answer_streaming(
    pipeline: RagPipeline,
    question: str,
    strategy: str,
    *,
    k: int = 3,
) -> StreamingAnswer:
    """用第 05 课的检索和引用检查，按一种办法把回答送到屏幕上。

    ``strategy`` 只能是 ``buffer_then_check`` 或 ``print_while_checking``。
    分数低于阈值时不调用模型，两种办法的中途都是空的。
    """

    if strategy not in STRATEGY_LABEL:
        raise ValueError("strategy 只能是 buffer_then_check 或 print_while_checking")
    hits = pipeline.retrieve(question, k=k)
    strong = [hit for hit in hits if hit.score >= pipeline.min_score]
    if not strong:
        return StreamingAnswer(
            strategy=strategy,
            streamed=[],
            final_answer=REFUSAL,
            refused=True,
            reason="low_score",
            citations=[],
            hits=hits,
            draft="",
        )
    draft = pipeline.client.complete(build_rag_messages(question, strong)).text
    policy = apply_citation_policy(draft, strong, hits)
    streamed: list[str] = []
    if strategy == PRINT_WHILE_CHECKING:
        streamed = _stream_until_unknown(draft, strong, hits)
    return StreamingAnswer(
        strategy=strategy,
        streamed=streamed,
        final_answer=policy.answer,
        refused=policy.refused,
        reason=policy.reason,
        citations=list(policy.citations),
        hits=hits,
        draft=draft,
    )


def _stream_until_unknown(draft: str, strong: list[Hit], hits: list[Hit]) -> list[str]:
    """边推边看。未知引用的方括号一闭合，就不再推后面的段。

    前缀里还没有完整引用时不拒绝。缺引用要等流结束，由整段检查决定。
    方括号还没闭合时也不拒绝。最终回答不用这里的前缀，仍看整段草稿。
    """

    shown: list[str] = []
    buffer = ""
    for chunk in split_draft(draft):
        shown.append(chunk)
        buffer += chunk
        if _unknown_citation_closed(buffer, strong, hits):
            break
    return shown


def _unknown_citation_closed(prefix: str, strong: list[Hit], hits: list[Hit]) -> bool:
    if prefix.rfind("[") > prefix.rfind("]"):
        return False
    return apply_citation_policy(prefix, strong, hits).reason == "unknown_citation"
