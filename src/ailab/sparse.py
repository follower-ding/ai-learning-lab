"""关键词检索。BM25 按词累加，离线、可重复。

分词和 `TfidfEmbedder` 一样，用字符二元组。两条路看见的是同一批字，
打分方式不同：这里对不上的词不加分，也不把整句方向拽偏。
"""

import math
from collections import Counter

from ailab.embeddings import char_bigrams


class Bm25Index:
    """Okapi BM25。`k1` 和 `b` 用常见默认值，不拿某一道题来调。

    问句里的二元组出现几次就加几次。文档里没出现的词贡献 0。
    没在任何文档里出现过的词直接丢掉，不会因为「库外的词」把分数拉成负数。
    """

    def __init__(self, *, k1: float = 1.2, b: float = 0.75) -> None:
        if k1 <= 0:
            raise ValueError("k1 必须是正数")
        if not 0.0 <= b <= 1.0:
            raise ValueError("b 必须在 0 和 1 之间")
        self.k1 = k1
        self.b = b
        self._ids: list[str] = []
        self._tf: list[dict[str, int]] = []
        self._dl: list[int] = []
        self._df: dict[str, int] = {}
        self._avgdl = 0.0
        self._n = 0

    def __len__(self) -> int:
        return self._n

    @property
    def ids(self) -> list[str]:
        return list(self._ids)

    def fit(self, ids: list[str], texts: list[str]) -> "Bm25Index":
        if len(ids) != len(texts):
            raise ValueError("id 和文本的条数要一致")
        if not ids:
            raise ValueError("fit 至少需要一条文档")
        seen: set[str] = set()
        for doc_id in ids:
            if doc_id in seen:
                raise ValueError(f"重复 id: {doc_id}")
            seen.add(doc_id)

        tf_rows: list[dict[str, int]] = []
        lengths: list[int] = []
        df: dict[str, int] = {}
        for text in texts:
            tokens = char_bigrams(text)
            counts = Counter(tokens)
            tf_rows.append(dict(counts))
            lengths.append(len(tokens))
            for token in counts:
                df[token] = df.get(token, 0) + 1
        if not df:
            raise ValueError("文档里没有可用的字符二元组")

        self._ids = list(ids)
        self._tf = tf_rows
        self._dl = lengths
        self._df = df
        self._avgdl = sum(lengths) / len(lengths)
        self._n = len(ids)
        return self

    def raw_scores(self, text: str) -> dict[str, float]:
        """每个文档一个未压缩的 BM25 分。键是 fit 时的 id。"""

        if self._n == 0:
            raise RuntimeError("先 fit 文档，再计算关键词分。查询和文档必须共用同一套词频。")
        tokens = char_bigrams(text)
        scores: dict[str, float] = {}
        for doc_id, tf, length in zip(self._ids, self._tf, self._dl, strict=True):
            score = 0.0
            for token in tokens:
                freq = tf.get(token)
                if not freq:
                    continue
                doc_freq = self._df[token]
                idf = math.log(1.0 + (self._n - doc_freq + 0.5) / (doc_freq + 0.5))
                normalizer = freq + self.k1 * (1.0 - self.b + self.b * length / self._avgdl)
                score += idf * (freq * (self.k1 + 1.0)) / normalizer
            scores[doc_id] = score
        return scores
