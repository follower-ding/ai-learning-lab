"""真实客户端只在测试里用假传输，不访问网络。"""

import json

import httpx
import pytest

from ailab.client import OpenAICompatibleClient
from ailab.embeddings import OpenAICompatibleEmbedder
from ailab.errors import ConfigError
from ailab.messages import Message


def test_chat_completion_parses_usage_and_hides_the_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        assert body["model"] == "demo-model"
        assert body["max_tokens"] == 400
        assert request.headers["Authorization"].startswith("Bearer ")
        assert body["messages"][1]["role"] == "user"
        assert "合计 60 元" in body["messages"][1]["content"]
        return httpx.Response(
            200,
            json={
                "model": "demo-model",
                "choices": [{"message": {"content": "下午 14:00 可以入住。"}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 8},
            },
        )

    with OpenAICompatibleClient(
        base_url="https://example.test/v1",
        api_key="sk-testsecret",
        model="demo-model",
        transport=httpx.MockTransport(handler),
    ) as client:
        completion = client.complete(
            [
                Message("system", "只回答规章。"),
                Message("tool", "合计 60 元", name="quote_late_checkout"),
            ]
        )
    assert completion.text == "下午 14:00 可以入住。"
    assert completion.prompt_tokens == 12
    assert completion.completion_tokens == 8


def test_http_error_redacts_secrets() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(403, text="bad key sk-testsecret and Bearer sk-testsecret")

    client = OpenAICompatibleClient(
        base_url="https://example.test/v1",
        api_key="sk-testsecret",
        model="demo-model",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(ConfigError) as caught:
        client.complete([Message("user", "你好")])
    client.close()
    message = str(caught.value)
    assert "403" in message
    assert "sk-testsecret" not in message


def test_embeddings_restore_index_order() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        assert body["input"] == ["入住", "早餐"]
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.0, 1.0]},
                    {"index": 0, "embedding": [1.0, 0.0]},
                ]
            },
        )

    embedder = OpenAICompatibleEmbedder(
        base_url="https://example.test/v1",
        api_key="sk-testsecret",
        model="embed-demo",
        transport=httpx.MockTransport(handler),
    )
    assert embedder.embed(["入住", "早餐"]) == [[1.0, 0.0], [0.0, 1.0]]
    embedder.close()


def test_embedding_error_redacts_secrets() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(401, text="unauthorized sk-testsecret")

    embedder = OpenAICompatibleEmbedder(
        base_url="https://example.test/v1",
        api_key="sk-testsecret",
        model="embed-demo",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(ConfigError) as caught:
        embedder.embed(["入住"])
    embedder.close()
    assert "sk-testsecret" not in str(caught.value)


@pytest.mark.integration
def test_live_chat_optional() -> None:
    import os

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        pytest.skip("未设置 OPENAI_API_KEY，跳过真实接口")
    client = OpenAICompatibleClient(
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        api_key=api_key,
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
    )
    try:
        completion = client.complete([Message("user", "只回复一个字：好")])
    finally:
        client.close()
    assert completion.text.strip()
