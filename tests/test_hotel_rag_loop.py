"""演示和评测入口：默认离线，没有密钥不构造 HTTP 客户端。"""

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from ailab.errors import LabError
from ailab.messages import Completion
from ailab.tools import Tool

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "apps" / "hotel-rag"
DEMO = APP / "demo.py"
EVAL = APP / "eval.py"
ASK = APP / "ask.py"
FIELDS = ("问题：", "最高分：", "拒绝原因：", "工具名：", "步数：", "停止原因：")
REFUSAL = "知识库里没有足够依据，不能回答。"


@pytest.fixture(autouse=True)
def _empty_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """空字符串会挡住 load_dotenv(override=False) 从本地 .env 把密钥填回来。"""

    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9")


def test_demo_offline_shows_citation_refusal_tools_and_trace(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _forbid_http(monkeypatch)
    demo = _load("hotel_rag_demo", DEMO)
    _forbid_live_lookup(monkeypatch)
    assert demo.main([]) == 0
    output = capsys.readouterr().out
    assert "跳过真实接口" not in output
    assert "第 16 课" not in output
    _assert_clean(output)
    for case_id in ("cite-checkin", "refuse-pool", "agent-late-three-hours", "agent-cancel-policy"):
        assert output.count(f"案例 {case_id}") == 1

    checkin = _block(output, "cite-checkin")
    assert checkin.index("检索：") < checkin.index("回答：") < checkin.index("问题：")
    assert "faq-checkin" in checkin
    assert "过线" in checkin
    for field in FIELDS:
        assert field in checkin

    pool = _block(output, "refuse-pool")
    assert pool.index("检索：") < pool.index("回答：") < pool.index("拒绝原因：low_score")
    assert f"回答：{REFUSAL}" in pool
    assert "未过线" in pool
    assert f"拒绝原因：{REFUSAL}" not in pool
    for field in FIELDS:
        assert field in pool

    late = _block(output, "agent-late-three-hours")
    assert late.index("工具：") < late.index("回答：") < late.index("工具名：quote_late_checkout")
    assert "90" in late
    assert "quote_late_checkout" in late

    cancel = _block(output, "agent-cancel-policy")
    assert cancel.index("工具：") < cancel.index("回答：") < cancel.index("工具名：lookup_policy")
    assert "24" in cancel
    assert "lookup_policy" in cancel

    for line in output.splitlines():
        if line.startswith("拒绝原因："):
            assert line in {"拒绝原因：low_score", "拒绝原因：（无）"}


def test_eval_offline_is_nine_of_nine_and_attaches_traces(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from ailab.evals import load_cases

    _forbid_http(monkeypatch)
    evaluate = _load("hotel_rag_eval", EVAL)
    _forbid_live_lookup(monkeypatch)
    assert evaluate.main([]) == 0
    output = capsys.readouterr().out
    assert "合计 9/9" in output
    assert "失败" not in output
    assert "跳过真实接口" not in output
    assert "第 16 课" not in output
    _assert_clean(output)
    cases = load_cases()
    assert len(cases) == 9
    parts = output.split("追踪 ")
    assert len(parts) == 10
    for case, part in zip(cases, parts[1:], strict=True):
        assert part.startswith(case.id)
        for field in FIELDS:
            assert field in part
        if case.id in {"refuse-pool", "refuse-airport"}:
            assert "拒绝原因：low_score" in part
            assert REFUSAL not in part
    for line in output.splitlines():
        if line.startswith("拒绝原因："):
            assert line in {"拒绝原因：low_score", "拒绝原因：（无）"}


def test_live_without_key_skips_and_does_not_construct_http(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _forbid_http(monkeypatch)
    demo = _load("hotel_rag_demo_live", DEMO)
    evaluate = _load("hotel_rag_eval_live", EVAL)
    ask = _load("hotel_rag_ask_live", ASK)

    monkeypatch.setenv("OPENAI_TIMEOUT", "9")
    monkeypatch.setenv("OPENAI_MAX_TOKENS", "11")
    assert demo.main(["--live"]) == 0
    output = capsys.readouterr().out
    assert "跳过真实接口" in output
    assert "没有调用网络" in output
    assert "没有真实接口的结果" in output
    assert "超时 9 秒" in output
    assert "max_tokens=11" in output
    assert "faq-checkin" not in output
    assert "合计" not in output
    _assert_clean(output)

    assert evaluate.main(["--live"]) == 0
    output = capsys.readouterr().out
    assert "跳过真实接口" in output
    assert "合计 9/9" not in output
    assert "没有真实接口的结果" in output
    _assert_clean(output)

    assert ask.main(["--live", "入住时间是什么时候？"]) == 0
    output = capsys.readouterr().out
    assert "跳过真实接口" in output
    assert "faq-checkin" not in output
    assert "没有真实接口的结果" in output
    _assert_clean(output)

    monkeypatch.delenv("OPENAI_TIMEOUT", raising=False)
    monkeypatch.delenv("OPENAI_MAX_TOKENS", raising=False)
    assert demo.main(["--live"]) == 0
    output = capsys.readouterr().out
    assert "超时 30 秒" in output
    assert "max_tokens=400" in output
    assert "跳过真实接口" in output

    monkeypatch.setenv("OPENAI_TIMEOUT", "很快")
    assert evaluate.main(["--live"]) == 0
    output = capsys.readouterr().out
    assert "跳过真实接口" in output
    assert "没有调用网络" in output
    assert "没有真实接口的结果" in output
    assert "合计 9/9" not in output
    _assert_clean(output)


def test_subprocess_sets_empty_api_key_and_stays_offline() -> None:
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = ""
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = str(ROOT / "src")
    env["OPENAI_BASE_URL"] = "http://127.0.0.1:9"
    probe = subprocess.run(
        [sys.executable, "-c", "import os; print(repr(os.environ.get('OPENAI_API_KEY')))"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert probe.stdout.strip() == "''"

    demo = _spawn(env, "demo.py")
    assert demo.returncode == 0, demo.stderr
    assert "faq-checkin" in demo.stdout
    assert "拒绝原因：low_score" in demo.stdout
    assert "quote_late_checkout" in demo.stdout
    assert "lookup_policy" in demo.stdout
    for field in FIELDS:
        assert field in demo.stdout
    _assert_clean(demo.stdout)
    _assert_clean(demo.stderr)

    evaluate = _spawn(env, "eval.py")
    assert evaluate.returncode == 0, evaluate.stderr
    assert "合计 9/9" in evaluate.stdout
    assert "失败" not in evaluate.stdout
    _assert_clean(evaluate.stdout)

    skipped = _spawn(env, "demo.py", "--live")
    assert skipped.returncode == 0, skipped.stderr
    assert "跳过真实接口" in skipped.stdout
    assert "没有真实接口的结果" in skipped.stdout
    assert "faq-checkin" not in skipped.stdout
    _assert_clean(skipped.stdout)

    skipped_eval = _spawn(env, "eval.py", "--live")
    assert skipped_eval.returncode == 0, skipped_eval.stderr
    assert "跳过真实接口" in skipped_eval.stdout
    assert "合计 9/9" not in skipped_eval.stdout
    _assert_clean(skipped_eval.stdout)


def test_trace_files_match_already_redacted_stdout(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    from ailab.evals import load_cases

    _forbid_http(monkeypatch)
    demo = _load("hotel_rag_demo_trace", DEMO)
    evaluate = _load("hotel_rag_eval_trace", EVAL)
    ask = _load("hotel_rag_ask_trace", ASK)
    demo_dir = tmp_path / "demo"
    assert demo.main(["--trace-dir", str(demo_dir)]) == 0
    demo_out = capsys.readouterr().out
    for case_id in ("cite-checkin", "refuse-pool", "agent-late-three-hours", "agent-cancel-policy"):
        stored = (demo_dir / f"{case_id}.txt").read_text(encoding="utf-8")
        assert stored.endswith("\n")
        assert stored[:-1] in demo_out
        _assert_clean(stored)
    pool = (demo_dir / "refuse-pool.txt").read_text(encoding="utf-8")
    assert "拒绝原因：low_score" in pool
    assert REFUSAL not in pool

    eval_dir = tmp_path / "eval"
    assert evaluate.main(["--trace-dir", str(eval_dir)]) == 0
    eval_out = capsys.readouterr().out
    assert "合计 9/9" in eval_out
    for case in load_cases():
        stored = (eval_dir / f"{case.id}.txt").read_text(encoding="utf-8")
        assert stored.endswith("\n")
        assert stored[:-1] in eval_out
        for field in FIELDS:
            assert field in stored
        _assert_clean(stored)
        if case.expect == "refuse":
            assert "拒绝原因：low_score" in stored
            assert REFUSAL not in stored

    token = "sk-" + "demo-placeholder"
    ask_dir = tmp_path / "ask"
    assert ask.main(["--trace-dir", str(ask_dir), f"入住时间是什么时候？ {token}"]) == 0
    ask_out = capsys.readouterr().out
    assert token not in ask_out
    assert "[redacted]" in ask_out
    stored = (ask_dir / "ask.txt").read_text(encoding="utf-8")
    assert token not in stored
    assert stored[:-1] in ask_out
    _assert_clean(ask_out)
    _assert_clean(stored)


def test_ask_uses_the_flag_and_does_not_route_by_wording(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _forbid_http(monkeypatch)
    ask = _load("hotel_rag_ask", ASK)
    _forbid_live_lookup(monkeypatch)

    assert ask.main(["取消政策是什么？"]) == 0
    default_out = capsys.readouterr().out
    assert "路径：rag" in default_out
    assert "检索：" in default_out
    assert "调用 lookup_policy" not in default_out
    assert "工具名：（无）" in default_out
    _assert_clean(default_out)

    assert ask.main(["--rag", "入住时间是什么时候？"]) == 0
    rag_out = capsys.readouterr().out
    assert "faq-checkin" in rag_out
    assert "调用 quote_late_checkout" not in rag_out

    assert ask.main(["--agent", "取消政策是什么？"]) == 0
    agent_out = capsys.readouterr().out
    assert "路径：agent" in agent_out
    assert agent_out.index("工具：") < agent_out.index("回答：") < agent_out.index("工具名：lookup_policy")
    assert "24" in agent_out

    assert ask.main(["--rag", "--agent", "取消政策是什么？"]) == 2
    assert "不按问题自动分流" in capsys.readouterr().out
    assert ask.main([]) == 2


def test_make_targets_stay_out_of_lessons() -> None:
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = ""
    demo = subprocess.run(
        ["make", "-n", "hotel-rag-demo"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    evaluate = subprocess.run(
        ["make", "-n", "hotel-rag-eval"],
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
    assert demo.returncode == 0, demo.stderr
    assert "apps/hotel-rag/demo.py" in demo.stdout
    assert evaluate.returncode == 0, evaluate.stderr
    assert "apps/hotel-rag/eval.py" in evaluate.stdout
    assert lessons.returncode == 0, lessons.stderr
    assert "hotel-rag" not in lessons.stdout
    assert "lessons/16-" not in lessons.stdout
    assert "lessons/15-real-api-compare/exercise.py" in lessons.stdout


def test_layer_does_not_retarget_library_defaults() -> None:
    from ailab.evals import load_cases

    sources = [(path, path.read_text(encoding="utf-8")) for path in sorted(APP.glob("*.py"))]
    joined = "\n".join(text for _, text in sources)
    assert re.search(r"(?<![\w])client_from_env\b", joined) is None
    for name in ("OPENAI_EMBEDDING_MODEL", "ask_first", "ailab.hybrid", "ailab.rerank", "ailab.streaming"):
        assert name not in joined
    loop_text = next(text for path, text in sources if path.name == "loop.py")
    assert "run_eval()" in loop_text
    assert "run_eval(observe=observe)" in loop_text
    assert "run_eval(client=client, observe=observe)" in loop_text
    assert "live_client_from_env(" in loop_text
    assert "build_lab_rag" in loop_text
    assert not list((ROOT / "lessons").glob("16-*"))

    demo = _load("hotel_rag_demo_cases", DEMO)
    del demo
    import inspect

    import loop

    recorded = inspect.getsource(loop.run_recorded_eval)
    assert "rag.answer" not in recorded
    assert "_run_tool" not in recorded
    assert "trace_rag" in recorded
    assert "trace_agent" in recorded

    cases = {case.id: case for case in load_cases()}
    assert len(cases) == 9
    for demo_case in loop.DEMO_CASES:
        assert cases[demo_case.id].question == demo_case.question
        assert cases[demo_case.id].kind == demo_case.kind


def test_eval_score_and_trace_come_from_one_nondeterministic_pass(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """第二次开口会换成另一种回答。评分和追踪若不是同一次，字段会对不上，调用次数也会变多。"""

    from ailab.evals import load_cases

    _forbid_http(monkeypatch)
    _load("hotel_rag_eval_once", EVAL)
    import loop

    client = _DriftingClient()
    assert loop.run_recorded_eval(client, None) == 0
    output = capsys.readouterr().out
    report, traces = _split_eval(output)
    cases = load_cases()
    rag_model = [case for case in cases if case.kind == "rag" and case.expect != "refuse"]
    agents = [case for case in cases if case.kind == "agent"]
    refused = [case for case in cases if case.expect == "refuse"]
    expected = {("rag", case.question) for case in rag_model} | {("agent", case.question) for case in agents}

    assert client.calls == len(expected)
    assert set(client.starts) == expected
    assert set(client.starts.values()) == {1}
    for case in rag_model:
        assert "reason=missing_citation" in report[case.id]
        assert "拒绝原因：missing_citation" in traces[case.id]
        assert "停止原因：missing_citation" in traces[case.id]
    for case in agents:
        assert "stop=final" in report[case.id]
        assert "tools=-" in report[case.id]
        assert "工具名：（无）" in traces[case.id]
        assert "停止原因：final" in traces[case.id]
        assert "lookup_policy" not in traces[case.id]
    for case in refused:
        assert "reason=low_score" in report[case.id]
        assert "拒绝原因：low_score" in traces[case.id]
        assert ("rag", case.question) not in client.starts
    assert "合计 " in output
    _assert_clean(output)


def test_malicious_arguments_are_redacted_on_stdout_trace_and_exception(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    marker = "sk-" + "hotelrag-fake"
    leaked = "Bear" + "er " + marker
    arguments = {"topic": marker, "note": leaked}

    _forbid_http(monkeypatch)
    _load("hotel_rag_ask_redact", ASK)
    import loop

    _forbid_live_lookup(monkeypatch)
    secret_call = _ScriptedClient(
        [
            {
                "type": "tool_call",
                "name": marker,
                "arguments": arguments,
            }
        ]
    )
    trace_dir = tmp_path / "ask"
    code = loop.run_live_or_mock(
        False,
        lambda _client: loop.run_ask("取消政策是什么？", "agent", secret_call, trace_dir),
    )
    assert code == 0
    assert secret_call.calls == 1
    printed = capsys.readouterr()
    stored = (trace_dir / "ask.txt").read_text(encoding="utf-8")
    assert marker not in printed.out
    assert marker not in printed.err
    assert marker not in stored
    assert "[redacted]" in printed.out
    assert "工具名：[redacted]" in stored
    assert "参数" in printed.out
    _assert_clean(printed.out)
    _assert_clean(printed.err)
    _assert_clean(stored)

    def explode(arguments: dict) -> str:
        raise LabError("参数错误 " + str(dict(arguments)))

    def fake_tools() -> dict[str, Tool]:
        return {
            "lookup_policy": Tool("lookup_policy", "读取规章", explode),
            "quote_late_checkout": Tool("quote_late_checkout", "报价", explode),
        }

    monkeypatch.setattr(loop, "hotel_tools", fake_tools)
    failing = _ScriptedClient(
        [
            {
                "type": "tool_call",
                "name": "lookup_policy",
                "arguments": arguments,
            }
        ]
    )
    failed_dir = tmp_path / "failed"
    code = loop.run_live_or_mock(
        False,
        lambda _client: loop.run_ask("取消政策是什么？", "agent", failing, failed_dir),
    )
    assert code == 1
    assert failing.calls == 1
    failed = capsys.readouterr()
    assert "没有完成：" in failed.out
    assert "[redacted]" in failed.out
    assert marker not in failed.out
    assert marker not in failed.err
    assert not failed_dir.exists()
    _assert_clean(failed.out)
    _assert_clean(failed.err)


class _DriftingClient:
    """第一次开口不引用、不调用工具。若再开口，检索会补上引用，工具循环会去查取消政策。"""

    def __init__(self) -> None:
        self.calls = 0
        self.starts: dict[tuple[str, str], int] = {}
        self.timeout = 30
        self.max_tokens = 400

    def complete(self, messages, *, temperature: float = 0.0) -> Completion:
        del temperature
        self.calls += 1
        system = "\n".join(message.content for message in messages if message.role == "system")
        if "【工具协议】" in system:
            channel = "agent"
        elif "【检索资料】" in system:
            channel = "rag"
        else:
            channel = "other"
        question = next((message.content for message in messages if message.role == "user"), "")
        opening = not any(message.role == "assistant" for message in messages)
        key = (channel, question)
        if opening:
            phase = self.starts.get(key, 0) + 1
            self.starts[key] = phase
        else:
            phase = self.starts.get(key, 1)
        if channel == "agent":
            if phase == 1 and opening:
                text = json.dumps({"type": "final", "answer": "第一次不调用工具"}, ensure_ascii=False)
            elif opening:
                text = json.dumps(
                    {
                        "type": "tool_call",
                        "name": "lookup_policy",
                        "arguments": {"topic": "cancel"},
                    },
                    ensure_ascii=False,
                )
            else:
                text = json.dumps({"type": "final", "answer": "第二次才调用工具"}, ensure_ascii=False)
        elif phase == 1:
            text = "第一次没有引用。"
        else:
            text = "第二次补上引用。[faq-checkin]"
        return Completion(text=text, model="drift", prompt_tokens=1, completion_tokens=1)


class _ScriptedClient:
    def __init__(self, payloads: list[dict]) -> None:
        self._payloads = list(payloads)
        self.calls = 0
        self.timeout = 30
        self.max_tokens = 400

    def complete(self, messages, *, temperature: float = 0.0) -> Completion:
        del messages, temperature
        self.calls += 1
        payload = self._payloads.pop(0) if self._payloads else {"type": "final", "answer": "没有更多脚本"}
        return Completion(
            text=json.dumps(payload, ensure_ascii=False),
            model="script",
            prompt_tokens=1,
            completion_tokens=1,
        )


def _split_eval(output: str) -> tuple[dict[str, str], dict[str, str]]:
    head, *parts = output.split("追踪 ")
    report: dict[str, str] = {}
    for line in head.splitlines():
        if "  通过  " not in line and "  失败  " not in line:
            continue
        case_id, rest = line.split("  ", maxsplit=1)
        report[case_id] = rest
    traces = {part.splitlines()[0]: part for part in parts}
    return report, traces


def test_new_files_have_no_secrets() -> None:
    pattern = re.compile(r"sk-[A-Za-z0-9_\-]+")
    roots = [APP, ROOT / "tests" / "test_hotel_rag_loop.py", ROOT / "Makefile"]
    for path in _files(roots):
        text = path.read_text(encoding="utf-8")
        found = pattern.findall(text)
        assert not found, path.relative_to(ROOT)
        assert ("Bearer" + " ") not in text
        assert ("Authorization" + ":") not in text


def _forbid_http(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(self: httpx.Client, *args: object, **kwargs: object) -> None:
        del self, args, kwargs
        raise AssertionError("没有密钥时不应构造 HTTP 客户端")

    monkeypatch.setattr(httpx.Client, "__init__", boom)


def _forbid_live_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    import loop

    def boom(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("默认路径不应调用 live_client_from_env")

    monkeypatch.setattr(loop, "live_client_from_env", boom)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _block(output: str, case_id: str) -> str:
    part = output.split(f"案例 {case_id}", maxsplit=1)[1]
    if "\n案例 " in part:
        part = part.split("\n案例 ", maxsplit=1)[0]
    return part


def _spawn(env: dict[str, str], script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(APP / script), *args],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def _assert_clean(text: str) -> None:
    assert "sk-" not in text
    assert "Bearer" not in text
    assert "Authorization" not in text


def _files(roots: list[Path]) -> list[Path]:
    found: list[Path] = []
    for root in roots:
        if root.is_file():
            found.append(root)
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.name == ".env" or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            found.append(path)
    return found
