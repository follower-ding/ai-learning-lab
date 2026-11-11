"""应用评测集：离线、同一次评分的耗时和 token。不依赖六字段追踪。"""

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

from ailab.client import MockChatClient
from ailab.corpus import faq_index
from ailab.evals import load_cases
from ailab.messages import Completion, Message
from ailab.rag import DEFAULT_MIN_SCORE
from ailab.tools import hotel_tools

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "apps" / "hotel-rag"
ENTRY = APP / "app_eval.py"
CASES = APP / "app_cases.json"
COURSE_IDS = (
    "cite-checkin",
    "cite-breakfast",
    "cite-parking",
    "cite-late-no-price",
    "refuse-pool",
    "refuse-airport",
    "agent-do-not-guess-hours",
    "agent-late-three-hours",
    "agent-cancel-policy",
)


@pytest.fixture(autouse=True)
def _empty_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9")


@pytest.fixture
def app_eval(monkeypatch: pytest.MonkeyPatch):
    _forbid_http(monkeypatch)
    return _load("hotel_rag_app_eval", ENTRY)


def test_shipped_cases_are_unique_and_cover_the_four_boundaries(app_eval) -> None:
    cases = app_eval.load_app_cases(CASES)
    ids = [item.case.id for item in cases]
    assert len(cases) >= 12
    assert len(ids) == len(set(ids))
    assert set(item.group for item in cases) == set(app_eval.GROUPS)
    assert all(item.basis.strip() for item in cases)
    floor = next(item for item in cases if item.case.id == "app-paraphrase-floor")
    wifi_name = next(item for item in cases if item.case.id == "app-paraphrase-wifi-name")
    assert floor.case.expect == "cite" and floor.case.source_id == "faq-parking"
    assert "不证明车位在哪一层" in floor.basis
    assert wifi_name.case.expect == "cite" and wifi_name.case.source_id == "faq-wifi"
    course = load_cases()
    assert [item.id for item in course] == list(COURSE_IDS)
    assert DEFAULT_MIN_SCORE == 0.2
    assert set(faq_index()) == {
        "faq-checkin",
        "faq-checkout",
        "faq-breakfast",
        "faq-parking",
        "faq-pets",
        "faq-wifi",
        "faq-cancel",
    }
    assert set(hotel_tools()) == {"lookup_policy", "quote_late_checkout"}


def test_loader_rejects_duplicate_ids_and_illegal_combinations(app_eval, tmp_path: Path) -> None:
    with pytest.raises(app_eval.AppCaseError) as duplicate:
        app_eval.load_app_cases(_write(tmp_path / "dup.json", [_row("same"), _row("same", question="另一句")]))
    assert "重复 id" in str(duplicate.value)

    illegal = [
        ("bad-kind", _row("bad-kind", kind="search", expect="cite")),
        ("rag-tool", _row("rag-tool", kind="rag", expect="tool", tool_name="lookup_policy")),
        ("agent-cite", _row("agent-cite", kind="agent", expect="cite", source_id="faq-wifi")),
        ("cite-missing", _row("cite-missing", source_id="")),
        ("refuse-sourced", _row("refuse-sourced", expect="refuse", source_id="faq-wifi")),
        ("unknown-tool", _row("unknown-tool", kind="agent", expect="tool", tool_name="shell", source_id="")),
        ("no-tool-named", _row("no-tool-named", kind="agent", expect="no_tool", tool_name="lookup_policy", source_id="")),
        ("overlap", _row("overlap", forbid=["30"], require=["30"])),
        ("empty-piece", _row("empty-piece", forbid=[""])),
        ("no-basis", _row("no-basis", basis="")),
    ]
    for name, row in illegal:
        with pytest.raises(app_eval.AppCaseError) as caught:
            app_eval.load_app_cases(_write(tmp_path / f"{name}.json", [row, *_padding(name)]))
        assert caught.value.problems


def test_report_uses_one_run_and_lists_real_failures(app_eval, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    calls = {"n": 0}
    real = app_eval.run_eval

    def counting(cases, *, client=None):
        calls["n"] += 1
        assert len(cases) == 1
        assert isinstance(client, app_eval.TokenMeter)
        assert isinstance(client._inner, MockChatClient)
        return real(cases, client=client)

    monkeypatch.setattr(app_eval, "run_eval", counting)
    code = app_eval.main([])
    output = capsys.readouterr().out
    cases = app_eval.load_app_cases()
    assert calls["n"] == len(cases)
    assert code == 1
    assert "进入调用之前加一" in output
    assert "不是供应商 usage 的 0" in output
    assert "不是供应商 usage，也不是账单" in output
    assert "退出码：0 全部通过；1 有失败或错误；2 用例集不合法" in output
    assert "问题：" not in output
    assert "最高分：" not in output

    expected_fail = {"app-paraphrase-floor", "app-paraphrase-wifi-name"}
    for item in cases:
        line = _line(output, item.case.id)
        if item.case.id in expected_fail:
            assert "status=fail" in line and "calls=0" in line
        else:
            assert "status=pass" in line, line
    floor = _line(output, "app-paraphrase-floor")
    wifi_name = _line(output, "app-paraphrase-wifi-name")
    gym = _line(output, "app-ungrounded-gym")
    two_hours = _line(output, "app-tool-two-hours")
    assert "status=fail" in floor and "calls=0" in floor
    assert "prompt_tokens=0" in floor and "token_source=estimated" in floor
    assert "status=fail" in wifi_name and "calls=0" in wifi_name
    assert "status=pass" in gym and "calls=0" in gym and "token_source=estimated" in gym
    assert "status=pass" in two_hours and "calls=2" in two_hours and "token_source=estimated" in two_hours
    prompt, completion = _token_pair(two_hours)
    assert prompt > 0 and completion > 0
    assert "合计 通过 " in output
    assert "失败 " in output
    assert "错误 0" in output


def test_timer_wraps_only_that_run_eval(app_eval, monkeypatch: pytest.MonkeyPatch) -> None:
    case = next(item for item in app_eval.load_app_cases() if item.case.id == "app-ungrounded-gym")
    real = app_eval.run_eval
    seen = {"n": 0}

    def slow(cases, *, client=None):
        seen["n"] += 1
        time.sleep(0.05)
        return real(cases, client=client)

    monkeypatch.setattr(app_eval, "run_eval", slow)
    row = app_eval.measure_case(case, MockChatClient())
    assert seen["n"] == 1
    assert row.elapsed_ms >= 45
    assert row.calls == 0
    assert row.status == "pass"


def test_meter_returns_the_same_completion_and_sums_steps() -> None:
    app_eval = _load("hotel_rag_app_eval_meter", ENTRY)
    sentinel = Completion(text="一次", model="stub", prompt_tokens=2, completion_tokens=5)
    second = Completion(text="两次", model="stub", prompt_tokens=3, completion_tokens=7)

    class Inner:
        def __init__(self) -> None:
            self.pending = [sentinel, second]

        def complete(self, messages, *, temperature: float = 0.0) -> Completion:
            del messages, temperature
            return self.pending.pop(0)

    meter = app_eval.TokenMeter(Inner())
    assert meter.complete([Message("user", "一")]) is sentinel
    assert meter.complete([Message("user", "二")]) is second
    assert meter.calls == 2
    assert meter.prompt_tokens == 5
    assert meter.completion_tokens == 12
    assert meter.token_source == "unknown"

    odd = Completion(text="坏", model="stub", prompt_tokens="3", completion_tokens=4)
    odd_meter = app_eval.TokenMeter(_Fixed(odd))
    assert odd_meter.complete([Message("user", "三")]) is odd
    assert odd_meter.calls == 1
    assert odd_meter.token_source == "unknown"
    row = app_eval._row(
        app_eval.AppCase(case=_eval_case(), group="tool", basis="边界"),
        odd_meter,
        1.0,
        "pass",
        "detail",
    )
    text = app_eval.format_report([row])
    assert "prompt_tokens=unknown" in text
    assert "completion_tokens=unknown" in text
    assert "completion_tokens=4" not in text
    assert "token_source=unknown" in text

    mock_meter = app_eval.TokenMeter(MockChatClient())
    original = mock_meter.complete([Message("user", "入住时间是什么时候？")])
    assert mock_meter.calls == 1
    assert mock_meter.token_source == "estimated"
    assert mock_meter.prompt_tokens == original.prompt_tokens
    assert mock_meter.completion_tokens == original.completion_tokens


def test_exception_is_an_error_and_not_a_pass(app_eval, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def boom(self, messages, *, temperature: float = 0.0) -> Completion:
        del self, messages, temperature
        raise RuntimeError("应用评测爆炸")

    monkeypatch.setattr(MockChatClient, "complete", boom)
    assert app_eval.main([]) == 1
    output = capsys.readouterr()
    wifi = _line(output.out, "app-grounded-wifi-password")
    gym = _line(output.out, "app-ungrounded-gym")
    assert "status=error" in wifi and "calls=1" in wifi
    assert "token_source=unknown" in wifi
    assert "prompt_tokens=unknown" in wifi
    assert "prompt_tokens=0" not in wifi
    assert "通过" not in wifi.split("status=error", maxsplit=1)[0]
    assert "应用评测爆炸" in wifi
    assert "status=pass" in gym and "calls=0" in gym
    assert "token_source=estimated" in gym and "prompt_tokens=0" in gym
    assert "Traceback" not in output.err
    assert "合计" in output.out and "错误 " in output.out


def test_attempt_is_counted_before_complete_and_partial_tokens_stay_unknown(app_eval) -> None:
    import json

    from ailab.evals import EvalCase

    app_case = app_eval.AppCase(
        case=EvalCase(
            "late-step",
            "agent",
            "延迟退房 2 小时要多少钱？",
            "tool",
            tool_name="quote_late_checkout",
            require=("60",),
        ),
        group="tool",
        basis="两步里第二步失败。",
    )
    tool_call = json.dumps(
        {"type": "tool_call", "name": "quote_late_checkout", "arguments": {"hours": 2}},
        ensure_ascii=False,
    )

    class FirstFails:
        def complete(self, messages, *, temperature: float = 0.0) -> Completion:
            del messages, temperature
            raise RuntimeError("首步失败")

    first = app_eval.measure_case(app_case, FirstFails())
    assert first.calls == 1
    assert first.status == "error"
    assert first.token_source == "unknown"
    first_line = _line(app_eval.format_report([first]), "late-step")
    assert "calls=1" in first_line and "prompt_tokens=unknown" in first_line
    assert "prompt_tokens=0" not in first_line

    class SecondFails:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, messages, *, temperature: float = 0.0) -> Completion:
            del messages, temperature
            self.calls += 1
            if self.calls == 1:
                return Completion(text=tool_call, model="stub", prompt_tokens=11, completion_tokens=22)
            raise RuntimeError("第二步失败")

    second_client = SecondFails()
    second = app_eval.measure_case(app_case, second_client)
    assert second_client.calls == 2
    assert second.calls == 2
    assert second.status == "error"
    assert second.token_source == "unknown"
    second_line = _line(app_eval.format_report([second]), "late-step")
    assert "calls=2" in second_line
    assert "prompt_tokens=unknown" in second_line
    assert "completion_tokens=unknown" in second_line
    assert "prompt_tokens=11" not in second_line
    assert "completion_tokens=22" not in second_line


def test_report_redacts_sk_only_marker(app_eval, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    marker = "sk-" + "only-rule-one"
    rows = json.loads(CASES.read_text(encoding="utf-8"))
    assert len(rows) >= 12
    rows[0]["basis"] = rows[0]["basis"] + " " + marker
    rows[0]["id"] = marker
    scored = {"n": 0}
    real = app_eval.run_eval

    def counting(cases, *, client=None):
        scored["n"] += 1
        return real(cases, client=client)

    monkeypatch.setattr(app_eval, "run_eval", counting)
    code = app_eval.main(["--cases", str(_write(tmp_path / "report.json", rows))])
    printed = capsys.readouterr()
    assert code == 1
    assert scored["n"] == len(rows)
    assert marker not in printed.out
    assert marker not in printed.err
    assert "[redacted]" in printed.out
    assert "Bear" + "er " not in printed.out
    blocked = {"n": 0}

    def forbid(cases, *, client=None):
        del cases, client
        blocked["n"] += 1
        raise AssertionError("集合错误不应评分")

    monkeypatch.setattr(app_eval, "run_eval", forbid)
    dup_id = marker
    payload = rows[1:] + [_row("pad-a"), _row("pad-b")]
    payload.append(_row(dup_id))
    payload.append(_row(dup_id, question="另一句重复"))
    assert len([row for row in payload if row["id"] != dup_id]) >= 12
    assert app_eval.main(["--cases", str(_write(tmp_path / "dup.json", payload))]) == 2
    failed = capsys.readouterr()
    assert blocked["n"] == 0
    assert dup_id not in failed.out
    assert dup_id not in failed.err
    assert "[redacted]" in failed.out
    assert "重复 id" in failed.out
    assert "至少 12" not in failed.out
    assert "Traceback" not in failed.err


def test_report_redacts_bearer_only_marker(app_eval, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    marker = "Bear" + "er " + "only-rule-two"
    rows = json.loads(CASES.read_text(encoding="utf-8"))
    assert len(rows) >= 12
    rows[1]["basis"] = marker
    scored = {"n": 0}
    real = app_eval.run_eval

    def counting(cases, *, client=None):
        scored["n"] += 1
        return real(cases, client=client)

    monkeypatch.setattr(app_eval, "run_eval", counting)
    assert app_eval.main(["--cases", str(_write(tmp_path / "report.json", rows))]) == 1
    printed = capsys.readouterr()
    assert scored["n"] == len(rows)
    assert marker not in printed.out
    assert marker not in printed.err
    assert "only-rule-two" not in printed.out
    assert "[redacted]" in printed.out
    assert "sk-" not in printed.out
    blocked = {"n": 0}

    def forbid(cases, *, client=None):
        del cases, client
        blocked["n"] += 1
        raise AssertionError("集合错误不应评分")

    monkeypatch.setattr(app_eval, "run_eval", forbid)
    payload = list(rows)
    payload.append(_row(marker))
    payload.append(_row(marker, question="另一句重复"))
    assert app_eval.main(["--cases", str(_write(tmp_path / "dup.json", payload))]) == 2
    failed = capsys.readouterr()
    assert blocked["n"] == 0
    assert marker not in failed.out
    assert marker not in failed.err
    assert "only-rule-two" not in failed.out
    assert "[redacted]" in failed.out
    assert "至少 12" not in failed.out
    assert "Traceback" not in failed.err


def test_non_string_fields_and_bad_encoding_exit_2_without_scoring(
    app_eval, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    base = json.loads(CASES.read_text(encoding="utf-8"))
    assert len(base) >= 12
    scored = {"n": 0}

    def forbid(cases, *, client=None):
        del cases, client
        scored["n"] += 1
        raise AssertionError("类型错误不应评分")

    monkeypatch.setattr(app_eval, "run_eval", forbid)
    samples = [
        ("refuse-number", _row("refuse-number", expect="refuse", source_id=123)),
        ("cite-list", _row("cite-list", tool_name=[])),
        ("no-tool-object", _row("no-tool-object", kind="agent", expect="no_tool", source_id="", tool_name={})),
    ]
    for name, bad in samples:
        scored["n"] = 0
        path = _write(tmp_path / f"{name}.json", [*base, bad])
        assert app_eval.main(["--cases", str(path)]) == 2
        printed = capsys.readouterr()
        assert scored["n"] == 0
        assert "必须是字符串" in printed.out
        assert "至少 12" not in printed.out
        assert "status=" not in printed.out
        assert "Traceback" not in printed.err

    raw = tmp_path / "bad-encoding.json"
    raw.write_bytes(b"\xff\xfe not utf8")
    scored["n"] = 0
    assert app_eval.main(["--cases", str(raw)]) == 2
    printed = capsys.readouterr()
    assert scored["n"] == 0
    assert "不是 UTF-8" in printed.out
    assert "Traceback" not in printed.err
    assert printed.err == ""


def test_invalid_file_exits_2_before_scoring(app_eval, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    path = _write(tmp_path / "bad.json", [_row("only-one")])
    assert app_eval.main(["--cases", str(path)]) == 2
    output = capsys.readouterr().out
    assert output.startswith("用例集不能用：")
    assert "合计" not in output
    assert "status=" not in output


def test_entry_has_no_live_switch_and_stays_out_of_lessons() -> None:
    source = ENTRY.read_text(encoding="utf-8")
    assert "def _grade" not in source
    assert "observe" not in source
    assert "trace_rag" not in source
    assert "trace_agent" not in source
    assert "--live" not in source
    assert "live_client_from_env" not in source
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = ""
    dry = subprocess.run(
        ["make", "-n", "hotel-rag-app"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    lessons = subprocess.run(
        ["make", "-n", "lessons"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert dry.returncode == 0, dry.stderr
    assert "apps/hotel-rag/app_eval.py" in dry.stdout
    assert lessons.returncode == 0, lessons.stderr
    assert "app_eval.py" not in lessons.stdout
    assert "hotel-rag-app" not in lessons.stdout


def test_subprocess_empty_key_lists_failures_and_stays_offline() -> None:
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = ""
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = str(ROOT / "src")
    env["OPENAI_BASE_URL"] = "http://127.0.0.1:9"
    completed = subprocess.run(
        [sys.executable, str(ENTRY)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 1, completed.stderr
    assert "status=fail" in completed.stdout
    assert "app-paraphrase-floor" in completed.stdout
    assert "token_source=estimated" in completed.stdout
    assert "Traceback" not in completed.stderr
    assert "sk-" not in completed.stdout
    assert "Bearer" not in completed.stdout


class _Fixed:
    def __init__(self, completion: Completion) -> None:
        self.completion = completion

    def complete(self, messages, *, temperature: float = 0.0) -> Completion:
        del messages, temperature
        return self.completion


def _eval_case():
    from ailab.evals import EvalCase

    return EvalCase("x", "agent", "延迟退房 2 小时要多少钱？", "tool", tool_name="quote_late_checkout")


def _row(case_id: str, **overrides: object) -> dict:
    row = {
        "id": case_id,
        "group": "grounded",
        "kind": "rag",
        "question": "客房网络密码在哪里？",
        "expect": "cite",
        "source_id": "faq-wifi",
        "basis": "faq-wifi 写了密码的位置。",
    }
    row.update(overrides)
    return row


def _padding(prefix: str) -> list[dict]:
    rows = []
    for index in range(12):
        rows.append(_row(f"{prefix}-pad-{index}", question=f"填充问题 {index}"))
    return rows


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return path


def _line(output: str, case_id: str) -> str:
    for line in output.splitlines():
        if line.startswith(case_id + " "):
            return line
    raise AssertionError(case_id)


def _token_pair(line: str) -> tuple[int, int]:
    prompt = int(line.split("prompt_tokens=", maxsplit=1)[1].split()[0])
    completion = int(line.split("completion_tokens=", maxsplit=1)[1].split()[0])
    return prompt, completion


def _forbid_http(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(self: httpx.Client, *args: object, **kwargs: object) -> None:
        del self, args, kwargs
        raise AssertionError("应用评测不应构造 HTTP 客户端")

    monkeypatch.setattr(httpx.Client, "__init__", boom)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
