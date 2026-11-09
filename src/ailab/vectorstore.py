"""内存向量库。只保存课程规模的几十条记录，方便把检索过程看清楚。"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ailab.embeddings import cosine
from ailab.errors import VectorDimensionError


@dataclass(frozen=True)
class Hit:
    id: str
    text: str
    score: float
    metadata: Mapping[str, str]


class MemoryVectorStore:
    """同一张表里记下稳定 id、原文、向量和少量 metadata。"""

    def __init__(self) -> None:
        self._records: list[tuple[str, str, tuple[float, ...], dict[str, str]]] = []
        self.dimension: int | None = None

    def __len__(self) -> int:
        return len(self._records)

    def add(
        self,
        record_id: str,
        text: str,
        vector: Sequence[float],
        metadata: Mapping[str, str] | None = None,
    ) -> None:
        if any(existing[0] == record_id for existing in self._records):
            raise ValueError(f"重复 id: {record_id}")
        stored = tuple(float(value) for value in vector)
        if not stored:
            raise ValueError("向量不能为空")
        if self.dimension is None:
            self.dimension = len(stored)
        elif len(stored) != self.dimension:
            raise VectorDimensionError(
                f"维度不一致：库里是 {self.dimension}，这条是 {len(stored)}。换模型后要整库重算。"
            )
        self._records.append((record_id, text, stored, dict(metadata or {})))

    def search(self, vector: Sequence[float], k: int = 3) -> list[Hit]:
        if k <= 0:
            raise ValueError("k 必须是正整数")
        if self.dimension is None:
            return []
        query = tuple(float(value) for value in vector)
        if len(query) != self.dimension:
            raise VectorDimensionError(
                f"维度不一致：库里是 {self.dimension}，查询是 {len(query)}。"
            )
        ranked = [
            (cosine(query, stored), record_id, text, metadata)
            for record_id, text, stored, metadata in self._records
        ]
        ranked.sort(key=lambda row: (-row[0], row[1]))
        return [
            Hit(id=record_id, text=text, score=score, metadata=metadata)
            for score, record_id, text, metadata in ranked[:k]
        ]
