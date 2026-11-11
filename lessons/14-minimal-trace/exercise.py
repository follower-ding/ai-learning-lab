"""第 14 课练习：六个字段先打到标准输出。两种拒绝在追踪里不能写成一样。"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ailab.agent import run_agent
from ailab.client import MockChatClient
from ailab.evals import load_cases
from ailab.messages import Completion
from ailab.rag import DEFAULT_MIN_SCORE, REFUSAL, build_lab_rag
from ailab.tools import hotel_tools
from ailab.trace import TraceRecord, emit_trace, trace_agent, trace_rag

UNKNOWN_DRAFT = "游泳池全天开放。来源：[faq-pool]"
MISSING_DRAFT = "下午 14:00 可以入住，不用看来源。"
SECRET = "sk-testsecret"


class FixedClient:
    def __init__(self, text: str) -> None:
        self.text = text

    def complete(self, messages, *, temperature: float = 0.0) -> Completion:
        del messages, temperature
        return Completion(text=self.text, model="fixed-stub", prompt_tokens=1, completion_tokens=1)


def main() -> int:
    print("第 14 课 · 最小追踪")
    print("一张追踪有六个字段：问题、最高分、拒绝原因、工具名、步数、停止原因。")
    print("先看标准输出。文件只是同一段文字的可选副本。")
    cases = {case.id: case for case in load_cases()}
    if _refusals(cases) != 0:
        return 1
    if _agent(cases) != 0:
        return 1
    if _redacted() != 0:
        return 1
    allowed = list(hotel_tools())
    print("允许列表：" + "、".join(allowed))
    if allowed != ["lookup_policy", "quote_late_checkout"]:
        print("允许列表必须正好是这两只工具。")
        return 1
    return 0


def _show(title: str, record: TraceRecord) -> str:
    print(title)
    return emit_trace(record)


def _reason(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("拒绝原因："):
            return line
    return ""


def _refusals(cases: dict) -> int:
    pool = cases["refuse-pool"]
    checkin = cases["cite-checkin"]
    pool_answer = build_lab_rag().answer(pool.question)
    unknown_answer = build_lab_rag(FixedClient(UNKNOWN_DRAFT)).answer(checkin.question)
    missing_answer = build_lab_rag(FixedClient(MISSING_DRAFT)).answer(checkin.question)
    print(f"客人看到的拒绝是同一句：{pool_answer.answer}")
    if pool_answer.answer != unknown_answer.answer or pool_answer.answer != missing_answer.answer:
        print("这三次给客人的句子应当相同。差别只留在追踪里。")
        return 1
    if pool_answer.answer != REFUSAL:
        print("拒绝语应当仍是第 05 课那一句。")
        return 1

    low = trace_rag(pool.question, pool_answer)
    unknown = trace_rag(checkin.question, unknown_answer)
    missing = trace_rag(checkin.question, missing_answer)
    low_text = _show("第 08 课 refuse-pool：检索没过阈值。", low)
    unknown_text = _show("同一句入住问题，草稿写了 [faq-pool]：引用检查把草稿扔了。", unknown)
    missing_text = _show("同一句入住问题，草稿没有引用：引用检查同样把草稿扔了。", missing)

    if low.refusal_reason != "low_score" or "拒绝原因：low_score" not in low_text:
        print("没过阈值时，拒绝原因应当是 low_score。")
        return 1
    if unknown.refusal_reason != "unknown_citation" or "拒绝原因：unknown_citation" not in unknown_text:
        print("引用了没检索到的文档时，拒绝原因应当是 unknown_citation。")
        return 1
    if missing.refusal_reason != "missing_citation" or "拒绝原因：missing_citation" not in missing_text:
        print("草稿没有引用时，拒绝原因应当是 missing_citation。")
        return 1
    if _reason(low_text) == _reason(unknown_text) or _reason(low_text) == _reason(missing_text):
        print("没过阈值，和引用检查扔掉草稿，在追踪里变成了同一个拒绝原因。")
        return 1
    if REFUSAL in low_text or REFUSAL in unknown_text or REFUSAL in missing_text:
        print("追踪里不该抄那句给客人的拒绝。看拒绝原因这个字段。")
        return 1
    if low.top_score is None or low.top_score >= DEFAULT_MIN_SCORE:
        print("没过阈值时，最高分应当低于 0.2。")
        return 1
    if unknown.top_score is None or unknown.top_score < DEFAULT_MIN_SCORE:
        print("引用检查扔掉草稿时，检索已经过线。")
        return 1
    if low.tool_name or low.step_count != 0 or unknown.step_count != 0:
        print("检索路径不走工具循环，工具名为空，步数为 0。")
        return 1
    return 0


def _agent(cases: dict) -> int:
    priced_case = cases["agent-late-three-hours"]
    guess_case = cases["agent-do-not-guess-hours"]
    cancel_case = cases["agent-cancel-policy"]
    tools = hotel_tools()
    priced = trace_agent(
        priced_case.question,
        run_agent(MockChatClient(), tools, priced_case.question, max_steps=4),
    )
    guess = trace_agent(
        guess_case.question,
        run_agent(MockChatClient(), tools, guess_case.question, max_steps=4),
    )
    cancel = trace_agent(
        cancel_case.question,
        run_agent(MockChatClient(), tools, cancel_case.question, max_steps=4),
    )
    priced_text = _show("第 08 课：写了 3 小时，计价工具算完再停。", priced)
    guess_text = _show("第 08 课：没写小时数，不调用计价。", guess)
    cancel_text = _show("第 08 课：取消政策走 lookup_policy。", cancel)
    if (
        priced.tool_name != "quote_late_checkout"
        or priced.step_count != 2
        or priced.stop_reason != "final"
        or "工具名：quote_late_checkout" not in priced_text
        or "步数：2" not in priced_text
        or "停止原因：final" not in priced_text
    ):
        print("写了小时数时，追踪里应当是 quote_late_checkout，步数 2，停止原因 final。")
        return 1
    if (
        guess.tool_name
        or guess.step_count != 1
        or guess.stop_reason != "final"
        or "工具名：（无）" not in guess_text
        or "30" in guess_text
    ):
        print("没写小时数时不调用工具，步数 1，停止原因仍是 final。追踪里不出现单价。")
        return 1
    if cancel.tool_name != "lookup_policy" or "工具名：lookup_policy" not in cancel_text:
        print("取消政策的工具名应当是 lookup_policy。")
        return 1
    if priced.top_score is not None or priced.refusal_reason:
        print("工具路径没有检索分，也不是第 05 课的拒绝。")
        return 1
    return 0


def _redacted() -> int:
    question = f"请不要记下 Authorization: Bearer {SECRET}"
    record = trace_rag(question, build_lab_rag().answer("游泳池几点开放？"))
    print("问题里夹了一份密钥。追踪先打到屏幕上，密钥不该出现。")
    text = emit_trace(record)
    if SECRET in text or "Bearer" in text or SECRET in record.question:
        print("追踪里还留着密钥或 Bearer。")
        return 1
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "trace.txt"
        again = emit_trace(record, path)
        stored = path.read_text(encoding="utf-8")
        if stored != again + "\n" or SECRET in stored or "Bearer" in stored:
            print("文件应当是刚才打到屏幕上的那一段，并且同样没有密钥。")
            return 1
    print("文件是可选项。上面这段在写入之前已经打到标准输出。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
