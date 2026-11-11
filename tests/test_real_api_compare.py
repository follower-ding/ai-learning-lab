"""第 15 课能离线核对的部分：超时和 max_tokens、默认仍走 Mock、没有密钥就退出 0。"""

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from ailab.client import OpenAICompatibleClient
from ailab.errors import ConfigError
from ailab.evals import run_eval
from ailab.messages import Completion, Message

ROOT = Path(__file__).resolve().parents[1]
EXERCISE = ROOT / "lessons" / "15-real-api-compare" / "exercise.py"
ALLOWED_SECRET_TOKENS = {"sk-testsecret", "sk-lesson14secret"}


def _chat_response(text: str = "好") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "demo-model",
            "choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
    )


def test_constructor_puts_timeout_and_max_tokens_on_the_fake_transport() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["max_tokens"] = json.loads(request.content.decode())["max_tokens"]
        seen["host"] = request.url.host
        return _chat_response()

    with OpenAICompatibleClient(
        base_url="https://example.test/v1",
        api_key="sk-testsecret",
        model="demo-model",
        timeout=7.5,
        max_tokens=64,
        transport=httpx.MockTransport(handler),
    ) as client:
        assert client.timeout == 7.5
        assert client._client.timeout.read == 7.5
        client.complete([Message("user", "你好")])
    assert seen["max_tokens"] == 64
    assert seen["host"] == "example.test"


def test_read_chat_limits_defaults_and_rejects_bad_values() -> None:
    from ailab.client import read_chat_limits

    assert read_chat_limits({}) == (30.0, 400)
    with pytest.raises(ConfigError):
        read_chat_limits({"OPENAI_TIMEOUT": "0"})
    with pytest.raises(ConfigError):
        read_chat_limits({"OPENAI_MAX_TOKENS": "-3"})
    with pytest.raises(ConfigError):
        read_chat_limits({"OPENAI_TIMEOUT": "很快"})


def test_live_client_without_key_does_not_open_http_and_with_key_uses_fake_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ailab.client import MockChatClient, client_from_env, live_client_from_env

    def boom(self: httpx.Client, *args: object, **kwargs: object) -> None:
        del self, args, kwargs
        raise AssertionError("没有密钥时不应构造 HTTP 客户端")

    monkeypatch.setattr(httpx.Client, "__init__", boom)
    assert live_client_from_env(environ={}) is None
    assert live_client_from_env(environ={"OPENAI_API_KEY": "   "}) is None
    assert isinstance(client_from_env(environ={}), MockChatClient)
    monkeypatch.undo()

    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["max_tokens"] = json.loads(request.content.decode())["max_tokens"]
        seen["host"] = request.url.host
        return _chat_response()

    environ = {
        "OPENAI_API_KEY": "sk-testsecret",
        "OPENAI_BASE_URL": "https://example.test/v1",
        "OPENAI_MODEL": "demo-model",
        "OPENAI_TIMEOUT": "9",
        "OPENAI_MAX_TOKENS": "11",
    }
    client = live_client_from_env(environ=environ, transport=httpx.MockTransport(handler))
    assert client is not None
    try:
        assert client.timeout == 9.0
        assert client.max_tokens == 11
        assert client._client.timeout.read == 9.0
        client.complete([Message("user", "只回复一个字：好")])
    finally:
        client.close()
    assert seen == {"max_tokens": 11, "host": "example.test"}

    forwarded = client_from_env(environ=environ, transport=httpx.MockTransport(handler))
    try:
        assert forwarded.timeout == 9.0
        assert forwarded.max_tokens == 11
    finally:
        forwarded.close()


def test_default_eval_stays_on_mock_and_can_swap_a_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-testsecret")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9")

    def boom(self: httpx.Client, *args: object, **kwargs: object) -> None:
        del self, args, kwargs
        raise AssertionError("默认评测不应构造真实客户端")

    monkeypatch.setattr(httpx.Client, "__init__", boom)
    offline = run_eval()
    assert offline.passed == offline.total == 9

    class Misleading:
        def complete(self, messages: list[Message], *, temperature: float = 0.0) -> Completion:
            del messages, temperature
            return Completion(
                text="游泳池全天开放。来源：[faq-pool]",
                model="misleading-stub",
                prompt_tokens=1,
                completion_tokens=1,
            )

    swapped = run_eval(client=Misleading())
    by_id = {check.case_id: check for check in swapped.checks}
    assert by_id["cite-checkin"].passed is False
    assert "unknown_citation" in by_id["cite-checkin"].detail
    assert by_id["refuse-pool"].passed is True
    assert by_id["agent-cancel-policy"].passed is False


def test_optional_live_test_still_skips_without_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from test_openai_transport import test_live_chat_optional

    with pytest.raises(pytest.skip.Exception):
        test_live_chat_optional()


def _load_exercise():
    spec = importlib.util.spec_from_file_location("lesson15_exercise", EXERCISE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exercise_without_key_exits_0_and_does_not_construct_http(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_TIMEOUT", "9")
    monkeypatch.setenv("OPENAI_MAX_TOKENS", "11")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9")

    def boom(self: httpx.Client, *args: object, **kwargs: object) -> None:
        del self, args, kwargs
        raise AssertionError("没有密钥时不应构造 HTTP 客户端")

    monkeypatch.setattr(httpx.Client, "__init__", boom)
    exercise = _load_exercise()
    assert exercise.main() == 0
    output = capsys.readouterr().out
    assert "第 15 课" in output.splitlines()[0]
    assert "第 08 课" in output
    assert "跳过真实接口" in output
    assert "没有调用网络" in output
    assert "没有真实接口的结果" in output
    assert "超时 9 秒" in output
    assert "max_tokens=11" in output
    assert "离线：引用成立，拒绝成立，工具成立" in output
    assert "合计 9/9" in output
    assert "真实接口：" not in output
    assert "sk-" not in output
    assert "Bearer" not in output
    assert "Authorization" not in output


def test_exercise_subprocess_without_key_exits_0() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env["PYTHONIOENCODING"] = "utf-8"
    env["OPENAI_API_KEY"] = ""
    env["OPENAI_TIMEOUT"] = "9"
    env["OPENAI_MAX_TOKENS"] = "11"
    env["OPENAI_BASE_URL"] = "http://127.0.0.1:9"
    completed = subprocess.run(
        [sys.executable, str(EXERCISE)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr + "\n" + completed.stdout
    assert "第 15 课" in completed.stdout.splitlines()[0]
    assert "跳过真实接口" in completed.stdout
    assert "没有调用网络" in completed.stdout
    assert "没有真实接口的结果" in completed.stdout
    assert "sk-" not in completed.stdout
    assert "Bearer" not in completed.stdout


def test_tree_has_no_live_secrets_and_example_key_stays_empty() -> None:
    example = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "OPENAI_API_KEY=\n" in example or example.rstrip().endswith("OPENAI_API_KEY=")
    for line in example.splitlines():
        if line.startswith("OPENAI_API_KEY="):
            assert line.strip() == "OPENAI_API_KEY="
    pattern = re.compile(r"sk-[A-Za-z0-9_\-]+")
    roots = [ROOT / "src", ROOT / "tests", ROOT / "lessons", ROOT / "README.md", ROOT / "ROADMAP.md", ROOT / ".env.example"]
    for path in _files(roots):
        text = path.read_text(encoding="utf-8")
        for match in pattern.findall(text):
            assert match in ALLOWED_SECRET_TOKENS, f"{path.relative_to(ROOT)} 含有未允许的 sk- 样式"


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
