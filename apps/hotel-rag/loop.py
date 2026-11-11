"""把现有的检索、工具循环、评测和追踪串成可重复的入口。

不改库的默认路径，不改语料，也不改课程。默认用 Mock。
``--live`` 只走 :func:`ailab.client.live_client_from_env`：没有密钥就返回 None，
不把 Mock 当成真实结果。
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ailab.agent import AgentResult, run_agent
from ailab.client import MockChatClient, OpenAICompatibleClient, live_client_from_env, read_chat_limits
from ailab.errors import ConfigError
from ailab.evals import EvalCase, format_report, load_cases, run_eval
from ailab.rag import REFUSAL, RagAnswer, RagPipeline, build_lab_rag
from ailab.redact import redact
from ailab.tools import hotel_tools
from ailab.trace import TraceRecord, emit_trace, trace_agent, trace_rag


@dataclass(frozen=True)
class DemoCase:
    id: str
    kind: str
    question: str


# 和第 08 课里这四条的问法、路径一致。工具循环不额外打开缺槽先问。
DEMO_CASES = (
    DemoCase("cite-checkin", "rag", "入住时间是什么时候？"),
    DemoCase("refuse-pool", "rag", "游泳池几点开放？"),
    DemoCase("agent-late-three-hours", "agent", "延迟退房 3 小时要多少钱？"),
    DemoCase("agent-cancel-policy", "agent", "取消政策是什么？"),
)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--live",
        action="store_true",
        help="有密钥时把 complete 换成真实接口。没有密钥则跳过真实接口，退出码为 0。",
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=None,
        help="可选。先把追踪打到标准输出，再把同一段已经脱敏的文字写入这个目录。",
    )


def run_live_or_mock(live: bool, body: Callable[[OpenAICompatibleClient | None], int]) -> int:
    """默认把 None 交给 body，由 body 走 Mock。``live`` 为真时才读环境。

    入口接住初始化、运行和关闭这三段里的异常，只打印脱敏文本，不把 traceback 打到标准错误。
    离线运行失败的退出码是 1。``--live`` 的失败退出码是 0，并写明没有补写结果。
    没有密钥仍是跳过，退出码 0。直接调用库函数时，异常照旧抛出。
    """

    if not live:
        try:
            return body(None)
        except Exception as exc:
            _print_failure("没有完成：", exc)
            return 1
    try:
        client = live_client_from_env()
    except Exception as exc:
        _print_failure("真实接口没有开始：", exc)
        _print_live_stopped()
        print("这次没有调用网络，也没有真实接口的结果。")
        return 0
    if client is None:
        _print_skip()
        return 0
    body_error: Exception | None = None
    close_error: Exception | None = None
    try:
        body(client)
    except Exception as exc:
        body_error = exc
    try:
        client.close()
    except Exception as exc:
        close_error = exc
    if body_error is None and close_error is None:
        return 0
    if body_error is not None:
        _print_failure("真实接口没有完成：", body_error)
    if close_error is not None:
        _print_failure("真实接口没有完成：", close_error)
    _print_live_stopped()
    return 0


def run_demo(client: OpenAICompatibleClient | None, trace_dir: Path | None) -> int:
    if client is None:
        print("岸港酒店演示")
        print("四条固定案例，一次跑完。默认离线 Mock，和第 08 课同一条路径。")
        rag = build_lab_rag()
    else:
        print("岸港酒店演示")
        print(f"真实接口。超时 {client.timeout:g} 秒，max_tokens={client.max_tokens}。")
        rag = build_lab_rag(client)
    tools = hotel_tools()
    problems: list[str] = []
    for case in DEMO_CASES:
        print(f"案例 {case.id}  {redact(case.question)}")
        if case.kind == "rag":
            answer = rag.answer(case.question)
            _print_retrieval(rag, answer)
            _print_guest(answer.answer)
            text = _emit(trace_rag(case.question, answer), trace_dir, case.id)
            problem = _demo_problem(case, answer=answer, trace_text=text)
        else:
            result = _run_tool(client, tools, case.question)
            _print_steps(result)
            _print_guest(result.answer)
            text = _emit(trace_agent(case.question, result), trace_dir, case.id)
            problem = _demo_problem(case, result=result, trace_text=text)
        if problem:
            problems.append(f"{case.id}：{problem}")
        print()
    if not problems:
        return 0
    for line in problems:
        print(line)
    if client is None:
        return 1
    print("真实接口的结果如上。这里不把离线约定写成已经通过。")
    return 0


def run_recorded_eval(client: OpenAICompatibleClient | None, trace_dir: Path | None) -> int:
    """用现有的 ``run_eval()`` 评分，并把同一次作答交给现有的追踪。不改用例集。"""

    captured: list[tuple[EvalCase, RagAnswer | AgentResult]] = []

    def observe(case: EvalCase, outcome: RagAnswer | AgentResult) -> None:
        captured.append((case, outcome))

    if client is None:
        print("岸港酒店评测")
        print("调用现有的 run_eval()，不传入客户端。这仍是第 08 课的 Mock。用例集不改。")
        report = run_eval(observe=observe)
    else:
        print("岸港酒店评测")
        print(f"真实接口。超时 {client.timeout:g} 秒，max_tokens={client.max_tokens}。")
        report = run_eval(client=client, observe=observe)
    print(redact(format_report(report)))
    print()
    # 当前 9 条用例的 id 都不重复，所以可以按 id 对上同一次作答。
    # 两条若共用一个 id，这里只会留下后写入的那条；这一层不支持重复 id。
    found = {case.id: (case, outcome) for case, outcome in captured}
    for case in load_cases():
        print(f"追踪 {case.id}")
        graded = found.get(case.id)
        if graded is None:
            print(f"未知 kind：{case.kind}")
            if client is None:
                return 1
            print()
            continue
        graded_case, outcome = graded
        if isinstance(outcome, RagAnswer):
            _emit(trace_rag(graded_case.question, outcome), trace_dir, case.id)
        elif isinstance(outcome, AgentResult):
            _emit(trace_agent(graded_case.question, outcome), trace_dir, case.id)
        else:
            print(f"未知 kind：{case.kind}")
            if client is None:
                return 1
        print()
    if client is not None:
        return 0
    cases = load_cases()
    if len(cases) != report.total or report.passed != report.total:
        print("离线 Mock 应当全部通过。")
        return 1
    return 0


def run_ask(
    question: str,
    kind: str,
    client: OpenAICompatibleClient | None,
    trace_dir: Path | None,
) -> int:
    """按调用方写明的路径走。不根据问法选择检索或工具。"""

    cleaned = question.strip()
    if not cleaned:
        print("请在命令后面写上客人的问题。")
        return 2
    if kind not in {"rag", "agent"}:
        print("请写 --rag 或 --agent。这里不按问题自动分流。")
        return 2
    if client is None:
        print("岸港酒店单问。默认离线 Mock。")
        rag = build_lab_rag()
    else:
        print(f"岸港酒店单问。真实接口。超时 {client.timeout:g} 秒，max_tokens={client.max_tokens}。")
        rag = build_lab_rag(client)
    print(f"路径：{kind}  {redact(cleaned)}")
    if kind == "rag":
        answer = rag.answer(cleaned)
        _print_retrieval(rag, answer)
        _print_guest(answer.answer)
        _emit(trace_rag(cleaned, answer), trace_dir, "ask")
    else:
        result = _run_tool(client, hotel_tools(), cleaned)
        _print_steps(result)
        _print_guest(result.answer)
        _emit(trace_agent(cleaned, result), trace_dir, "ask")
    print()
    return 0


def _run_tool(client: OpenAICompatibleClient | None, tools, question: str) -> AgentResult:
    chat = MockChatClient() if client is None else client
    return run_agent(chat, tools, question, max_steps=4)


def _print_retrieval(rag: RagPipeline, answer: RagAnswer) -> None:
    print("检索：")
    if not answer.hits:
        print("  （没有命中）")
        return
    for hit in answer.hits:
        source_id = str(hit.metadata.get("source_id", hit.id))
        mark = "过线" if hit.score >= rag.min_score else "未过线"
        print(f"  {source_id}  {hit.score:.3f}  {mark}")


def _print_steps(result: AgentResult) -> None:
    print("工具：")
    if not result.steps:
        print("  （没有步）")
        return
    for step in result.steps:
        if step.kind == "tool_call":
            # 工具名和参数来自模型输出，不能原样打出来。
            print(redact(f"  第 {step.index + 1} 步 调用 {step.name} 参数 {step.arguments}"))
            if step.tool_result is None:
                print("    结果：（无）")
            else:
                print(f"    结果：{redact(step.tool_result)}")
        else:
            print(f"  第 {step.index + 1} 步 {redact(step.kind)}")


def _print_failure(prefix: str, exc: BaseException) -> None:
    print(f"{prefix}{redact(_exception_text(exc))}")


def _print_live_stopped() -> None:
    print("没有补写结果。")


def _exception_text(exc: BaseException) -> str:
    """异常自己的文本、附注，以及 cause / context 链。交给 redact 之前先拼成一段。"""

    lines: list[str] = []
    seen: set[int] = set()

    def walk(current: BaseException | None) -> None:
        if current is None or id(current) in seen:
            return
        seen.add(id(current))
        lines.append(str(current))
        for note in getattr(current, "__notes__", ()) or ():
            lines.append(str(note))
        walk(current.__cause__)
        if not current.__suppress_context__:
            walk(current.__context__)

    walk(exc)
    return "\n".join(lines)


def _print_guest(text: str) -> None:
    print(f"回答：{redact(text)}")


def _emit(record: TraceRecord, trace_dir: Path | None, case_id: str) -> str:
    if trace_dir is None:
        return emit_trace(record)
    trace_dir.mkdir(parents=True, exist_ok=True)
    return emit_trace(record, trace_dir / f"{case_id}.txt")


_TRACE_FIELDS = ("问题：", "最高分：", "拒绝原因：", "工具名：", "步数：", "停止原因：")


def _demo_problem(
    case: DemoCase,
    *,
    answer: RagAnswer | None = None,
    result: AgentResult | None = None,
    trace_text: str,
) -> str:
    missing = [field for field in _TRACE_FIELDS if field not in trace_text]
    if missing:
        return "追踪缺少" + "、".join(field.rstrip("：") for field in missing)
    if case.id == "cite-checkin":
        if answer is None or answer.refused or "faq-checkin" not in answer.citations:
            return "引用里应当有 faq-checkin"
        return ""
    if case.id == "refuse-pool":
        if answer is None or answer.reason != "low_score" or answer.answer != REFUSAL or answer.citations:
            return "应当因 low_score 拒绝，客人看到的仍是固定拒绝"
        for line in trace_text.splitlines():
            if line.startswith("拒绝原因：") and REFUSAL in line:
                return "追踪的拒绝原因不该抄给客人的那句拒绝"
        if "拒绝原因：low_score" not in trace_text:
            return "追踪的拒绝原因应当是 low_score"
        return ""
    if case.id == "agent-late-three-hours":
        names = [] if result is None else [step.name for step in result.steps if step.name]
        if result is None or "quote_late_checkout" not in names or "90" not in result.answer:
            return "应当调用 quote_late_checkout，回答里有 90"
        return ""
    if case.id == "agent-cancel-policy":
        names = [] if result is None else [step.name for step in result.steps if step.name]
        if result is None or "lookup_policy" not in names or "24" not in result.answer:
            return "应当调用 lookup_policy，回答里有 24"
        return ""
    return "未知演示案例"


def _print_skip() -> None:
    try:
        timeout, max_tokens = read_chat_limits(os.environ)
    except ConfigError as exc:
        print(f"没有 OPENAI_API_KEY，跳过真实接口。时限没有读成：{redact(_exception_text(exc))}")
        print("这次没有调用网络，也没有真实接口的结果。")
        return
    print(
        "没有 OPENAI_API_KEY，跳过真实接口。"
        f"超时 {timeout:g} 秒，max_tokens={max_tokens}。"
        "这次没有调用网络，也没有真实接口的结果。"
    )
