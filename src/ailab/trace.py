"""第 14 课：一次作答留下一张最小追踪。

六个字段：问题、最高分、拒绝原因、工具名、步数、停止原因。
先打印到标准输出。要留档时，再把同一段已经脱敏的文字写入文件。

不改 ``RagPipeline.answer``，也不改第 06 课到第 13 课的默认路径。
密钥样式走 :func:`ailab.redact.redact`，和对话接口的报错是同一套。
"""

from dataclasses import dataclass
from pathlib import Path

from ailab.agent import AgentResult
from ailab.rag import RagAnswer
from ailab.redact import redact


@dataclass(frozen=True)
class TraceRecord:
    """一张追踪。六个字段都在。

    没有的检索分是 None，打印成「（无）」。
    没有的拒绝原因、工具名留空字符串，打印成「（无）」。
    检索路径不走工具循环，步数记 0。
    """

    question: str
    top_score: float | None
    refusal_reason: str
    tool_name: str
    step_count: int
    stop_reason: str


def trace_rag(question: str, answer: RagAnswer) -> TraceRecord:
    """从第 05 课的一次检索作答记一张追踪。不改那次作答。"""

    if answer.hits:
        top_score = max(hit.score for hit in answer.hits)
    else:
        top_score = None
    return TraceRecord(
        question=redact(question),
        top_score=top_score,
        refusal_reason=answer.reason if answer.refused else "",
        tool_name="",
        step_count=0,
        stop_reason=answer.reason,
    )


def trace_agent(question: str, result: AgentResult) -> TraceRecord:
    """从第 07 课的一次循环记一张追踪。不改那次循环。"""

    names: list[str] = []
    for step in result.steps:
        if not step.name:
            continue
        cleaned = redact(step.name)
        if cleaned not in names:
            names.append(cleaned)
    return TraceRecord(
        question=redact(question),
        top_score=None,
        refusal_reason="",
        tool_name="、".join(names),
        step_count=len(result.steps),
        stop_reason=result.stopped_reason,
    )


def format_trace(record: TraceRecord) -> str:
    """六个字段各一行。字符串再走一次 redact，手写的记录也不会留下密钥。"""

    if record.top_score is None:
        score = "（无）"
    else:
        score = f"{record.top_score:.3f}"
    lines = [
        f"问题：{redact(record.question)}",
        f"最高分：{score}",
        f"拒绝原因：{_shown(record.refusal_reason)}",
        f"工具名：{_shown(record.tool_name)}",
        f"步数：{record.step_count}",
        f"停止原因：{_shown(record.stop_reason)}",
    ]
    return "\n".join(lines)


def emit_trace(record: TraceRecord, path: str | Path | None = None) -> str:
    """先打印到标准输出。``path`` 给了，再把同一段写入文件。"""

    text = format_trace(record)
    print(text)
    if path is not None:
        Path(path).write_text(text + "\n", encoding="utf-8")
    return text


def _shown(value: str) -> str:
    cleaned = redact(value).strip()
    return cleaned or "（无）"
