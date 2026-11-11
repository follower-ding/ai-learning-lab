"""很小的一组行为用例。评的是引用、拒绝和工具，不是模型排行榜。"""

from collections.abc import Callable
from dataclasses import dataclass

from ailab.agent import AgentResult, run_agent
from ailab.client import ChatClient, MockChatClient
from ailab.corpus import read_json
from ailab.rag import REFUSAL, RagAnswer, RagPipeline, build_lab_rag
from ailab.tools import hotel_tools


@dataclass(frozen=True)
class EvalCase:
    id: str
    kind: str
    question: str
    expect: str
    source_id: str = ""
    tool_name: str = ""
    forbid: tuple[str, ...] = ()
    require: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvalCheck:
    case_id: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class EvalReport:
    checks: list[EvalCheck]

    @property
    def passed(self) -> int:
        return sum(1 for check in self.checks if check.passed)

    @property
    def total(self) -> int:
        return len(self.checks)


def load_cases() -> list[EvalCase]:
    cases = []
    for row in read_json("eval_cases.json"):
        cases.append(
            EvalCase(
                id=row["id"],
                kind=row["kind"],
                question=row["question"],
                expect=row["expect"],
                source_id=row.get("source_id", ""),
                tool_name=row.get("tool_name", ""),
                forbid=tuple(row.get("forbid", [])),
                require=tuple(row.get("require", [])),
            )
        )
    return cases


def run_eval(
    cases: list[EvalCase] | None = None,
    *,
    client: ChatClient | None = None,
    observe: Callable[[EvalCase, RagAnswer | AgentResult], None] | None = None,
) -> EvalReport:
    """默认用 Mock，和第 08 课一样。传入 ``client`` 时，两路的 ``complete`` 都换成这一个。

    第 15 课的真实对照才传入真实客户端。不传时，检索和工具仍各用一个新的 Mock。

    ``observe`` 默认不传。传了以后，每条用例在评分算完时把刚刚用过的那个结果交回来，
    不为此再调用一次模型或工具循环。不传时，评分、返回值和调用次数都和原来一样。
    """

    cases = load_cases() if cases is None else cases
    tools = hotel_tools()
    rag = build_lab_rag() if client is None else build_lab_rag(client)

    def run_question(question: str) -> AgentResult:
        chat = MockChatClient() if client is None else client
        return run_agent(chat, tools, question, max_steps=4)

    checks = [_grade(case, rag, run_question, observe) for case in cases]
    return EvalReport(checks)


def format_report(report: EvalReport) -> str:
    lines = []
    for check in report.checks:
        mark = "通过" if check.passed else "失败"
        lines.append(f"{check.case_id}  {mark}  {check.detail}")
    lines.append(f"合计 {report.passed}/{report.total}")
    return "\n".join(lines)


def _grade(case: EvalCase, rag: RagPipeline, run_question, observe=None) -> EvalCheck:
    if case.kind == "rag":
        result = rag.answer(case.question)
        passed, detail = _grade_rag(case, result)
        text = result.answer
    elif case.kind == "agent":
        result = run_question(case.question)
        passed, detail = _grade_agent(case, result)
        text = result.answer
    else:
        return EvalCheck(case.id, False, f"未知 kind: {case.kind}")
    if any(piece in text for piece in case.forbid):
        passed = False
        detail += " 命中禁止片段"
    if any(piece not in text for piece in case.require):
        passed = False
        detail += " 缺少必要片段"
    if observe is not None:
        observe(case, result)
    return EvalCheck(case.id, passed, detail)


def _grade_rag(case: EvalCase, result: RagAnswer) -> tuple[bool, str]:
    cited = ",".join(result.citations) or "-"
    detail = f"reason={result.reason} citations={cited}"
    if case.expect == "cite":
        return (not result.refused) and case.source_id in result.citations, detail
    if case.expect == "refuse":
        return result.refused and result.answer == REFUSAL and not result.citations, detail
    return False, detail + " expect 无法识别"


def _grade_agent(case: EvalCase, result: AgentResult) -> tuple[bool, str]:
    names = [step.name for step in result.steps if step.name]
    detail = f"stop={result.stopped_reason} tools={','.join(names) or '-'}"
    if case.expect == "tool":
        return result.stopped_reason == "final" and case.tool_name in names, detail
    if case.expect == "no_tool":
        return result.stopped_reason == "final" and not names, detail
    return False, detail + " expect 无法识别"
