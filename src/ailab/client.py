"""对话客户端。练习默认用 Mock；真实 HTTP 客户端留着同一套方法签名。"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import httpx

from ailab.errors import ConfigError, LabError
from ailab.messages import Completion, Message
from ailab.mock_front_desk import respond
from ailab.redact import redact
from ailab.tokens import estimate_tokens


class ChatClient(Protocol):
    """Mock 和真实客户端都实现这一个方法。练习依赖它，不依赖某个厂商的 SDK。"""

    def complete(self, messages: list[Message], *, temperature: float = 0.0) -> Completion:
        """把一组消息发给模型，拿回一段文本。"""
        ...


class MockChatClient:
    """离线客户端。

    ``script`` 按调用顺序逐句返回，用完就报错，不会偷偷改走启发式。
    不传 ``script`` 时，由 :func:`ailab.mock_front_desk.respond` 根据提示里的标记作答。
    """

    model = "mock-hotel-front-desk"

    def __init__(self, script: Sequence[str] | None = None) -> None:
        self._script: list[str] | None = None if script is None else list(script)
        self.calls: list[list[Message]] = []

    def complete(self, messages: list[Message], *, temperature: float = 0.0) -> Completion:
        del temperature
        self.calls.append(list(messages))
        if self._script is not None:
            if not self._script:
                raise LabError("Mock 脚本已用完。请补一句回复，或改用不传 script 的启发式 Mock。")
            text = self._script.pop(0)
        else:
            text = respond(messages)
        prompt = sum(estimate_tokens(message.content) for message in messages)
        return Completion(
            text=text,
            model=self.model,
            prompt_tokens=prompt,
            completion_tokens=estimate_tokens(text),
        )


class OpenAICompatibleClient:
    """调用 OpenAI 兼容的 ``/chat/completions``。

    ``timeout`` 交给 HTTP 客户端，到点就不再等。``max_tokens`` 写进请求体，限制这一次生成的长度。
    默认测试传入假传输，不访问网络。

    本课约定模型只在 ``content`` 里输出文本或 JSON，不读取厂商的 ``tool_calls`` 字段。
    这样 Mock 和真实接口走同一套解析。工具结果会写成一条 user 消息，因为部分兼容接口不接受 tool 这个 role。
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 30.0,
        max_tokens: int = 400,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not api_key:
            raise ConfigError("缺少 API key")
        if max_tokens <= 0:
            raise ConfigError("max_tokens 必须是正整数")
        import httpx

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self._api_key = api_key
        self._client = httpx.Client(timeout=timeout, transport=transport, trust_env=False)

    def complete(self, messages: list[Message], *, temperature: float = 0.0) -> Completion:
        import httpx

        payload = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
            "messages": [_render_message(message) for message in messages],
        }
        try:
            response = self._client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=payload,
            )
        except httpx.HTTPError as exc:
            raise ConfigError("调用对话接口失败") from exc
        if response.status_code >= 400:
            preview = redact(response.text[:300])
            raise ConfigError(f"对话接口返回 {response.status_code}：{preview}")
        payload = response.json()
        try:
            text = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ConfigError("对话接口的返回结构无法识别") from exc
        if not isinstance(text, str) or not text.strip():
            raise ConfigError("对话接口的 content 不是可用字符串")
        usage = payload.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        return Completion(
            text=text,
            model=str(payload.get("model") or self.model),
            prompt_tokens=int(prompt_tokens) if isinstance(prompt_tokens, int) else _estimate_prompt(messages),
            completion_tokens=int(completion_tokens) if isinstance(completion_tokens, int) else estimate_tokens(text),
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "OpenAICompatibleClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_chat_limits(environ: Mapping[str, str]) -> tuple[float, int]:
    """真实对话客户端的超时（秒）和 ``max_tokens``。没写时用 30 和 400。"""

    return _read_timeout(environ), _read_max_tokens(environ)


def live_client_from_env(
    environ: Mapping[str, str] | None = None,
    *,
    transport: httpx.BaseTransport | None = None,
) -> OpenAICompatibleClient | None:
    """有密钥才返回真实客户端，并带上超时和 ``max_tokens``。没有密钥返回 None，不退回 Mock。"""

    environ = _runtime_environ(environ)
    api_key = environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    timeout, max_tokens = read_chat_limits(environ)
    base_url = environ.get("OPENAI_BASE_URL", "").strip() or "https://api.openai.com/v1"
    model = environ.get("OPENAI_MODEL", "").strip() or "gpt-4o-mini"
    return OpenAICompatibleClient(
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout=timeout,
        max_tokens=max_tokens,
        transport=transport,
    )


def client_from_env(
    environ: Mapping[str, str] | None = None,
    *,
    transport: httpx.BaseTransport | None = None,
) -> MockChatClient | OpenAICompatibleClient:
    """有密钥就构造真实客户端，没有就返回 Mock。练习默认不走这个开关。

    第 15 课不要把这里的 Mock 当成真实结果。没有密钥时应使用 :func:`live_client_from_env`，它返回 None。
    """

    live = live_client_from_env(environ, transport=transport)
    if live is None:
        return MockChatClient()
    return live


def _runtime_environ(environ: Mapping[str, str] | None) -> Mapping[str, str]:
    if environ is not None:
        return environ
    from dotenv import load_dotenv

    root = Path(__file__).resolve().parents[2]
    load_dotenv(root / ".env", override=False)
    return os.environ


def _read_timeout(environ: Mapping[str, str]) -> float:
    raw = environ.get("OPENAI_TIMEOUT", "").strip()
    if not raw:
        return 30.0
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError("OPENAI_TIMEOUT 必须是正数") from exc
    if not (value > 0) or value == float("inf"):
        raise ConfigError("OPENAI_TIMEOUT 必须是正数")
    return value


def _read_max_tokens(environ: Mapping[str, str]) -> int:
    raw = environ.get("OPENAI_MAX_TOKENS", "").strip()
    if not raw:
        return 400
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError("OPENAI_MAX_TOKENS 必须是正整数") from exc
    if value <= 0:
        raise ConfigError("OPENAI_MAX_TOKENS 必须是正整数")
    return value


def _render_message(message: Message) -> dict[str, str]:
    if message.role == "tool":
        label = message.name or "tool"
        return {"role": "user", "content": f"工具 {label} 的结果：{message.content}"}
    return {"role": message.role, "content": message.content}


def _estimate_prompt(messages: list[Message]) -> int:
    return sum(estimate_tokens(message.content) for message in messages)
