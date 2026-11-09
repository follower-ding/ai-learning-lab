"""读取课程共用的岸港酒店资料。练习和测试都用这一份，避免样例漂移。"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent / "data"


@dataclass(frozen=True)
class FaqItem:
    id: str
    title: str
    text: str


def read_text(name: str) -> str:
    return (DATA_DIR / name).read_text(encoding="utf-8")


def read_json(name: str) -> Any:
    return json.loads(read_text(name))


def load_faq() -> list[FaqItem]:
    return [FaqItem(id=row["id"], title=row["title"], text=row["text"]) for row in read_json("hotel_faq.json")]


def load_policy() -> str:
    return read_text("hotel_policy.md")


def faq_index() -> dict[str, FaqItem]:
    return {item.id: item for item in load_faq()}
