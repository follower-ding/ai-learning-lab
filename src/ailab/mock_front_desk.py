"""启发式 Mock。它读提示里的标记决定走哪条脚本，不是模型，也不能当安全边界。"""

import json
import re

from ailab.corpus import faq_index
from ailab.messages import Message

# 顺序有意义：取消要排在入住前面，否则“入住日前取消”会被分成入住。
KEYWORDS: list[tuple[str, tuple[str, ...], str]] = [
    ("cancel", ("取消",), "faq-cancel"),
    ("check_out", ("退房",), "faq-checkout"),
    ("check_in", ("入住",), "faq-checkin"),
    ("breakfast", ("早餐",), "faq-breakfast"),
    ("parking", ("停车", "车位"), "faq-parking"),
    ("pets", ("宠物", "导盲犬"), "faq-pets"),
    ("wifi", ("wifi", "wi-fi", "网络"), "faq-wifi"),
]

PLAIN_FALLBACK = "我目前只准备了入住、退房、早餐、停车、宠物、网络和取消这几类实验问题。"


def respond(messages: list[Message]) -> str:
    system = "\n".join(message.content for message in messages if message.role == "system")
    if "【工具协议】" in system:
        return _tool_response(messages, system)
    if "【意图分类】" in system:
        return _intent_response(messages)
    if "【检索资料】" in system:
        return _rag_response(system)
    return plain_reply(_last_user(messages))


def plain_reply(question: str) -> str:
    match = _match_keyword(question)
    if match is None:
        return PLAIN_FALLBACK
    _intent, _words, faq_id = match
    return faq_index()[faq_id].text


def _intent_response(messages: list[Message]) -> str:
    question = _last_user(messages)
    match = _match_keyword(question)
    intent = "unknown" if match is None else match[0]
    return json.dumps({"intent": intent, "question": question}, ensure_ascii=False)


def _rag_response(system: str) -> str:
    matched = re.search(r"\[(faq-[a-z0-9-]+)\][^\n]*", system)
    if matched is None:
        return "知识库里没有足够依据，不能回答。"
    source_id = matched.group(1)
    body = re.sub(r"^\[faq-[a-z0-9-]+\]\s*", "", matched.group(0))
    return f"{body} 来源：[{source_id}]"


def _tool_response(messages: list[Message], system: str) -> str:
    tool_messages = [message for message in messages if message.role == "tool"]
    if tool_messages:
        last = tool_messages[-1]
        answer = f"根据工具 {last.name} 的结果：{last.content}"
        return json.dumps({"type": "final", "answer": answer}, ensure_ascii=False)

    question = _last_user(messages)
    hours = re.search(r"(\d+)\s*小时", question)
    asks_late_checkout = "延迟退房" in question or ("退房" in question and hours is not None)
    if asks_late_checkout and "取消" not in question:
        if hours is None:
            # 第 11 课才打开【缺槽先问】。没有这个标记时仍用 final，第 06 到 08 课不变。
            if "【缺槽先问】" in system:
                return json.dumps(
                    {"type": "clarify", "question": "要延迟几小时？"},
                    ensure_ascii=False,
                )
            answer = "请告诉我要延迟几小时。规章里没有默认时长，我不会猜一个数字。"
            return json.dumps({"type": "final", "answer": answer}, ensure_ascii=False)
        payload = {
            "type": "tool_call",
            "name": "quote_late_checkout",
            "arguments": {"hours": int(hours.group(1))},
        }
        return json.dumps(payload, ensure_ascii=False)

    match = _match_keyword(question)
    topic_by_intent = {
        "cancel": "cancel",
        "check_out": "checkout",
        "check_in": "checkin",
        "breakfast": "breakfast",
        "parking": "parking",
        "pets": "pets",
        "wifi": "wifi",
    }
    if match is not None and match[0] in topic_by_intent:
        payload = {
            "type": "tool_call",
            "name": "lookup_policy",
            "arguments": {"topic": topic_by_intent[match[0]]},
        }
        return json.dumps(payload, ensure_ascii=False)
    answer = "请说明要查的规章主题，或说明要延迟退房几小时。"
    return json.dumps({"type": "final", "answer": answer}, ensure_ascii=False)


def _match_keyword(question: str) -> tuple[str, tuple[str, ...], str] | None:
    folded = question.lower()
    for intent, words, faq_id in KEYWORDS:
        if any(word in folded for word in words):
            return intent, words, faq_id
    return None


def _last_user(messages: list[Message]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            return message.content
    return ""
