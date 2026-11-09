"""对话客户端。练习默认用 Mock；真实 HTTP 客户端留着同一套方法签名。"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
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


def client_from_env(environ: Mapping[str, str] | None = None) -> MockChatClient | OpenAICompatibleClient:
    """有密钥就构造真实客户端，没有就返回 Mock。练习默认不走这个开关。"""

    if environ is None:
        from dotenv import load_dotenv

        load_dotenv(override=False)
        environ = os.environ
    api_key = environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return MockChatClient()
    return OpenAICompatibleClient(
        base_url=environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").strip(),
        api_key=api_key,
        model=environ.get("OPENAI_MODEL", "gpt-4o-mini").strip(),
    )


def _render_message(message: Message) -> dict[str, str]:
    if message.role == "tool":
        label = message.name or "tool"
        return {"role": "user", "content": f"工具 {label} 的结果：{message.content}"}
    return {"role": message.role, "content": message.content}


def _estimate_prompt(messages: list[Message]) -> int:
    return sum(estimate_tokens(message.content) for message in messages)
