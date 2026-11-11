"""离线应用评测：用现有 run_eval 逐题评分，并记下同一次的耗时、调用数和 token。

不新增观察钩子，也不为了耗时或 token 再跑一遍。不打六字段追踪，也没有真实接口入口。
token 只来自这一次 Mock 的估算。现有 Completion 证明不了供应商 usage 或账单。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP_DIR = Path(__file__).resolve().parent
for _entry in (ROOT / "src", APP_DIR):
    _text = str(_entry)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from ailab.client import ChatClient, MockChatClient
from ailab.corpus import faq_index
from ailab.evals import EvalCase, run_eval
from ailab.messages import Completion
from ailab.redact import redact
from ailab.tools import hotel_tools

CASES_PATH = APP_DIR / "app_cases.json"
GROUPS = ("grounded", "ungrounded", "paraphrase", "tool")
_RAG_EXPECT = frozenset({"cite", "refuse"})
_AGENT_EXPECT = frozenset({"tool", "no_tool"})
_FIELDS = frozenset({"id", "group", "kind", "question", "expect", "source_id", "tool_name", "forbid", "require", "basis"})


class AppCaseError(Exception):
    """用例集本身不合法。此时还没有评分。"""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("；".join(problems))


@dataclass(frozen=True)
class AppCase:
    case: EvalCase
    group: str
    basis: str


@dataclass(frozen=True)
class MeasureRow:
    case_id: str
    group: str
    basis: str
    status: str
    calls: int
    elapsed_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None
    token_source: str
    detail: str


class TokenMeter:
    """包住一次 complete。签名不变，返回原来的 Completion，只累加这一次的 token 字段。"""

    def __init__(self, inner: ChatClient) -> None:
        self._inner = inner
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self._tokens_known = True
        self._is_mock = isinstance(inner, MockChatClient)

    def complete(self, messages: list, *, temperature: float = 0.0) -> Completion:
        completion = self._inner.complete(messages, temperature=temperature)
        self.calls += 1
        prompt = completion.prompt_tokens
        completion_tokens = completion.completion_tokens
        if _is_token_count(prompt) and _is_token_count(completion_tokens):
            self.prompt_tokens += prompt
            self.completion_tokens += completion_tokens
        else:
            self._tokens_known = False
        return completion

    @property
    def token_source(self) -> str:
        if not self._tokens_known:
            return "unknown"
        if self._is_mock:
            return "estimated"
        return "unknown"


def load_app_cases(path: Path | None = None) -> list[AppCase]:
    """读应用评测集。重复 id 或非法组合会整批拒绝。"""

    path = CASES_PATH if path is None else path
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise AppCaseError([f"读不到用例集：{path.name}：{exc.strerror or exc}"]) from exc
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise AppCaseError([f"用例集不是 JSON：{exc.msg}"]) from exc
    if not isinstance(payload, list):
        raise AppCaseError(["用例集必须是数组"])
    problems: list[str] = []
    cases: list[AppCase] = []
    seen: dict[str, int] = {}
    known_sources = set(faq_index())
    known_tools = set(hotel_tools())
    for index, row in enumerate(payload, start=1):
        label = f"第 {index} 条"
        if not isinstance(row, dict):
            problems.append(f"{label} 不是对象")
            continue
        extra = sorted(set(row) - _FIELDS)
        if extra:
            problems.append(f"{label} 有不能识别的字段：{', '.join(extra)}")
        case_id = _text(row.get("id"))
        if not case_id:
            problems.append(f"{label} 缺少 id")
            label_id = label
        else:
            label_id = case_id
            prior = seen.get(case_id)
            if prior is not None:
                problems.append(f"重复 id：{case_id}（第 {prior} 条和第 {index} 条）")
            else:
                seen[case_id] = index
        group = _text(row.get("group"))
        if group not in GROUPS:
            problems.append(f"{label_id} 的 group 必须是 {'、'.join(GROUPS)}")
        kind = _text(row.get("kind"))
        expect = _text(row.get("expect"))
        question = row.get("question")
        if not isinstance(question, str) or not question.strip():
            problems.append(f"{label_id} 的 question 必须是非空字符串")
            question = ""
        basis = row.get("basis")
        if not isinstance(basis, str) or not basis.strip():
            problems.append(f"{label_id} 缺少 basis。预期要写明来自哪条资料或哪条工具边界")
            basis = ""
        source_id = _optional_text(row.get("source_id"))
        tool_name = _optional_text(row.get("tool_name"))
        forbid, forbid_problem = _string_list(row.get("forbid", []), f"{label_id} 的 forbid")
        require, require_problem = _string_list(row.get("require", []), f"{label_id} 的 require")
        if forbid_problem:
            problems.append(forbid_problem)
        if require_problem:
            problems.append(require_problem)
        overlap = sorted(set(forbid) & set(require))
        if overlap:
            problems.append(f"{label_id} 的 require 和 forbid 重叠：{', '.join(overlap)}")
        problems.extend(
            _combination_problems(
                label_id,
                kind=kind,
                expect=expect,
                source_id=source_id,
                tool_name=tool_name,
                known_sources=known_sources,
                known_tools=known_tools,
            )
        )
        if problems and not case_id:
            continue
        if kind not in {"rag", "agent"} or not _expect_matches(kind, expect):
            continue
        cases.append(
            AppCase(
                case=EvalCase(
                    id=case_id,
                    kind=kind,
                    question=question.strip(),
                    expect=expect,
                    source_id=source_id,
                    tool_name=tool_name,
                    forbid=tuple(forbid),
                    require=tuple(require),
                ),
                group=group if group in GROUPS else "",
                basis=basis.strip(),
            )
        )
    if problems:
        raise AppCaseError(problems)
    if len(cases) < 12:
        raise AppCaseError([f"应用评测至少 12 条，现在是 {len(cases)} 条"])
    return cases


def measure_case(app_case: AppCase, client: ChatClient) -> MeasureRow:
    """只调用一次 run_eval。计时包住这一次，token 和调用数也只来自这一次。"""

    meter = TokenMeter(client)
    started = time.perf_counter()
    try:
        report = run_eval([app_case.case], client=meter)
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000
        return _row(app_case, meter, elapsed_ms, "error", redact(f"{type(exc).__name__}: {exc}"))
    elapsed_ms = (time.perf_counter() - started) * 1000
    if len(report.checks) != 1:
        return _row(app_case, meter, elapsed_ms, "error", f"评分条数是 {len(report.checks)}，不是 1")
    check = report.checks[0]
    if check.case_id != app_case.case.id:
        return _row(app_case, meter, elapsed_ms, "error", f"评分 id 是 {check.case_id}")
    status = "pass" if check.passed else "fail"
    return _row(app_case, meter, elapsed_ms, status, check.detail)


def format_report(rows: list[MeasureRow]) -> str:
    lines = [
        "岸港酒店应用评测",
        "离线 Mock。逐题调用现有的 run_eval，不重跑，也不打六字段追踪。",
        "token_source=estimated 是 Mock 的 estimate_tokens 估算，不是供应商 usage，也不是账单。",
        "token_source=unknown 表示现有 Completion 证明不了用量，这里不把它印成数字。",
        "退出码：0 全部通过；1 有失败或错误；2 用例集不合法，尚未评分。",
        "失败照实列出。预期按资料和工具边界写，不用 Mock 碰巧的结果改预期。",
    ]
    passed = sum(1 for row in rows if row.status == "pass")
    failed = sum(1 for row in rows if row.status == "fail")
    errors = sum(1 for row in rows if row.status == "error")
    for row in rows:
        mark = {"pass": "通过", "fail": "失败", "error": "错误"}[row.status]
        if row.token_source == "estimated":
            tokens = f"prompt_tokens={row.prompt_tokens}  completion_tokens={row.completion_tokens}"
        else:
            tokens = "prompt_tokens=unknown  completion_tokens=unknown"
        lines.append(
            f"{row.case_id}  {mark}  status={row.status}  calls={row.calls}  "
            f"elapsed_ms={row.elapsed_ms:.3f}  {tokens}  token_source={row.token_source}  "
            f"group={row.group}  {row.detail}"
        )
        lines.append(f"  依据：{row.basis}")
    lines.append(f"合计 通过 {passed}  失败 {failed}  错误 {errors}  / {len(rows)}")
    return "\n".join(lines)


def exit_code(rows: list[MeasureRow]) -> int:
    if any(row.status != "pass" for row in rows):
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="岸港酒店离线应用评测。没有真实接口。")
    parser.add_argument("--cases", type=Path, default=CASES_PATH, help="应用评测集 JSON。默认是本目录的 app_cases.json。")
    args = parser.parse_args(argv)
    try:
        cases = load_app_cases(args.cases)
    except AppCaseError as exc:
        print(f"用例集不能用：{exc}")
        return 2
    rows = [measure_case(app_case, MockChatClient()) for app_case in cases]
    print(format_report(rows))
    return exit_code(rows)


def _row(app_case: AppCase, meter: TokenMeter, elapsed_ms: float, status: str, detail: str) -> MeasureRow:
    known = meter.token_source == "estimated"
    return MeasureRow(
        case_id=app_case.case.id,
        group=app_case.group,
        basis=app_case.basis,
        status=status,
        calls=meter.calls,
        elapsed_ms=elapsed_ms,
        prompt_tokens=meter.prompt_tokens if known else None,
        completion_tokens=meter.completion_tokens if known else None,
        token_source=meter.token_source,
        detail=detail,
    )


def _combination_problems(
    label: str,
    *,
    kind: str,
    expect: str,
    source_id: str,
    tool_name: str,
    known_sources: set[str],
    known_tools: set[str],
) -> list[str]:
    problems: list[str] = []
    if kind == "rag":
        if expect not in _RAG_EXPECT:
            problems.append(f"{label} 的 kind=rag 时 expect 只能是 cite 或 refuse")
            return problems
        if tool_name:
            problems.append(f"{label} 是检索题，不能写 tool_name")
        if expect == "cite":
            if source_id not in known_sources:
                problems.append(f"{label} 的 cite 必须引用现有资料 id，现在是 {source_id or '（空）'}")
        elif source_id:
            problems.append(f"{label} 是拒绝题，不能再写 source_id")
        return problems
    if kind == "agent":
        if expect not in _AGENT_EXPECT:
            problems.append(f"{label} 的 kind=agent 时 expect 只能是 tool 或 no_tool")
            return problems
        if source_id:
            problems.append(f"{label} 是工具题，不能写 source_id")
        if expect == "tool":
            if tool_name not in known_tools:
                problems.append(f"{label} 的 tool 必须在允许列表里，现在是 {tool_name or '（空）'}")
        elif tool_name:
            problems.append(f"{label} 是 no_tool，不能再写 tool_name")
        return problems
    problems.append(f"{label} 的 kind 只能是 rag 或 agent")
    return problems


def _expect_matches(kind: str, expect: str) -> bool:
    if kind == "rag":
        return expect in _RAG_EXPECT
    if kind == "agent":
        return expect in _AGENT_EXPECT
    return False


def _string_list(value: object, label: str) -> tuple[list[str], str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        return [], f"{label} 必须是非空字符串组成的数组"
    return list(value), ""


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _optional_text(value: object) -> str:
    if value is None:
        return ""
    return _text(value)


def _is_token_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


if __name__ == "__main__":
    raise SystemExit(main())
