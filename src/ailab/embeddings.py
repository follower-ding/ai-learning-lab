"""把文本变成同一维度的向量。

默认实现是字符二元组上的 TF-IDF，离线、可重复。它看的是字面重叠，
不是“这句话在说同一件事”。换神经网络嵌入时实现 ``embed`` 即可，调用方不用改。
"""

import math
from collections.abc import Sequence
from typing import Protocol

from ailab.errors import ConfigError, VectorDimensionError
from ailab.redact import redact


class Embedder(Protocol):
    """把一批文本变成同一次调用里维度一致的向量。"""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """返回和 ``texts`` 等长的向量列表。"""
        ...


def char_bigrams(text: str) -> list[str]:
    """去掉空白和标点后取相邻两个字符。单个字的查询保留这个字本身。"""

    kept = "".join(char.lower() for char in text if char.isalnum())
    if not kept:
        return []
    if len(kept) == 1:
        return [kept]
    return [kept[index : index + 2] for index in range(len(kept) - 1)]


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """两个向量方向有多接近。长度为 0 的向量规定为 0，避免除零。"""

    if len(left) != len(right):
        raise VectorDimensionError(f"维度不一致：{len(left)} 与 {len(right)}")
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for x_value, y_value in zip(left, right, strict=True):
        dot += x_value * y_value
        left_norm += x_value * x_value
        right_norm += y_value * y_value
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / math.sqrt(left_norm * right_norm)


class TfidfEmbedder:
    """在已经 ``fit`` 过的文档集合上，把文本变成稠密向量。

    词频用对数，乘上 idf，再用余弦比较方向。常见二元组（比如很多规章里都有的“时间”）
    权重会低一些。查询里没在文档中出现过的二元组会被丢掉。
    """

    def __init__(self) -> None:
        self._df: dict[str, int] = {}
        self._vocab: dict[str, int] = {}
        self._n = 0

    @property
    def dimension(self) -> int:
        self._require_fit()
        return len(self._vocab)

    def fit(self, texts: list[str]) -> "TfidfEmbedder":
        if not texts:
            raise ValueError("fit 至少需要一条文档")
        df: dict[str, int] = {}
        for text in texts:
            for token in set(char_bigrams(text)):
                df[token] = df.get(token, 0) + 1
        if not df:
            raise ValueError("文档里没有可用的字符二元组")
        self._df = df
        self._vocab = {token: index for index, token in enumerate(sorted(df))}
        self._n = len(texts)
        return self

    def embed(self, texts: list[str]) -> list[list[float]]:
        self._require_fit()
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        counts: dict[str, int] = {}
        for token in char_bigrams(text):
            if token not in self._vocab:
                continue
            counts[token] = counts.get(token, 0) + 1
        vector = [0.0] * len(self._vocab)
        for token, count in counts.items():
            tf = 1.0 + math.log(count)
            idf = math.log((1 + self._n) / (1 + self._df[token])) + 1.0
            vector[self._vocab[token]] = tf * idf
        return vector

    def _require_fit(self) -> None:
        if not self._vocab:
            raise RuntimeError("先 fit 文档，再 embed。查询和文档必须共用同一套 idf。")


class OpenAICompatibleEmbedder:
    """调用 OpenAI 兼容的 ``/embeddings``。测试可以传入 ``transport``，不必访问网络。"""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 30.0,
        transport=None,
    ) -> None:
        if not api_key:
            raise ConfigError("缺少 API key")
        import httpx

        self.base_url = base_url.rstrip("/")
        self.model = model
        self._api_key = api_key
        self._client = httpx.Client(timeout=timeout, transport=transport, trust_env=False)

    def embed(self, texts: list[str]) -> list[list[float]]:
        import httpx

        try:
            response = self._client.post(
                f"{self.base_url}/embeddings",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={"model": self.model, "input": texts},
            )
        except httpx.HTTPError as exc:
            raise ConfigError("调用嵌入接口失败") from exc
        if response.status_code >= 400:
            raise ConfigError(f"嵌入接口返回 {response.status_code}：{redact(response.text[:300])}")
        payload = response.json()
        try:
            rows = sorted(payload["data"], key=lambda row: row["index"])
            vectors = [list(row["embedding"]) for row in rows]
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError("嵌入接口的返回结构无法识别") from exc
        if len(vectors) != len(texts):
            raise ConfigError("嵌入条数和输入条数不一致")
        return vectors

    def close(self) -> None:
        self._client.close()
