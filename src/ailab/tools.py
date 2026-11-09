"""两只纯函数工具。模型可以提出调用，不能自己填写结果。"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ailab.corpus import faq_index
from ailab.errors import ToolArgumentError

TOPICS = {
    "checkin": "faq-checkin",
    "checkout": "faq-checkout",
    "breakfast": "faq-breakfast",
    "parking": "faq-parking",
    "pets": "faq-pets",
    "wifi": "faq-wifi",
    "cancel": "faq-cancel",
}

HOURLY_RATE = 30


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    run: Callable[[Mapping[str, Any]], str]


def lookup_policy(arguments: Mapping[str, Any]) -> str:
    topic = arguments.get("topic")
    faq_id = TOPICS.get(topic) if isinstance(topic, str) else None
    if faq_id is None:
        allowed = "、".join(TOPICS)
        raise ToolArgumentError(f"topic 必须是 {allowed} 之一")
    item = faq_index()[faq_id]
    return f"{item.title}：{item.text}"


def quote_late_checkout(arguments: Mapping[str, Any]) -> str:
    hours = arguments.get("hours")
    if isinstance(hours, bool) or not isinstance(hours, int):
        raise ToolArgumentError("hours 需要 1 到 12 的整数")
    if hours < 1 or hours > 12:
        raise ToolArgumentError("hours 需要 1 到 12 的整数")
    total = hours * HOURLY_RATE
    return f"延迟退房 {hours} 小时，实验价每小时 {HOURLY_RATE} 元，合计 {total} 元。需前台确认还有空房。"


def hotel_tools() -> dict[str, Tool]:
    return {
        "lookup_policy": Tool(
            name="lookup_policy",
            description="按主题读取岸港酒店规章。topic 取 checkin、checkout、breakfast、parking、pets、wifi、cancel。",
            run=lookup_policy,
        ),
        "quote_late_checkout": Tool(
            name="quote_late_checkout",
            description="按小时计算延迟退房的实验价。hours 为 1 到 12 的整数。这个单价不在规章正文里。",
            run=quote_late_checkout,
        ),
    }
