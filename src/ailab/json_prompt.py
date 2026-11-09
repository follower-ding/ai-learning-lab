"""从模型输出里取出一个 JSON 对象，并做一次意图分类。"""

import json
import re
from dataclasses import dataclass

from ailab.errors import JsonExtractError
from ailab.messages import Completion, Message

ALLOWED_INTENTS = (
    "check_in",
    "check_out",
    "breakfast",
    "parking",
    "pets",
    "wifi",
    "cancel",
    "unknown",
)

INTENT_SYSTEM = """【意图分类】
把客人的问题分到一个意图。只输出一个 JSON 对象，不要解释。
格式：{"intent":"check_in","question":"客人的原话"}
intent 只能是 check_in、check_out、breakfast、parking、pets、wifi、cancel、unknown 之一。
question 原样抄写客人的问题。
"""

REPAIR_USER = "你刚才的输出无法解析。只输出一个 JSON 对象，不要解释。"


@dataclass(frozen=True)
class Intent:
    intent: str
    question: str
    attempts: int


def extract_json_object(text: str) -> dict:
    """取出第一个 JSON 对象。允许外面有说明文字或 ```json 围栏。"""

    candidate = text.strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", candidate, flags=re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1).strip()
    start = candidate.find("{")
    if start < 0:
        raise JsonExtractError("没有找到 JSON 对象")
    blob = _matching_object(candidate, start)
    try:
        value = json.loads(blob)
    except json.JSONDecodeError as exc:
        raise JsonExtractError("JSON 无法解析") from exc
    if not isinstance(value, dict):
        raise JsonExtractError("顶层必须是 JSON 对象")
    return value


def _matching_object(text: str, start: int) -> str:
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise JsonExtractError("JSON 对象没有闭合")


def parse_intent_payload(data: dict) -> tuple[str, str]:
    intent = data.get("intent")
    question = data.get("question")
    if intent not in ALLOWED_INTENTS:
        raise JsonExtractError("intent 不在允许列表")
    if not isinstance(question, str) or not question.strip():
        raise JsonExtractError("question 必须是非空字符串")
    return str(intent), question.strip()


def classify_intent(client, question: str) -> Intent:
    """分类意图。第一次解析失败时，把错误发回去再要一次。"""

    messages = [Message("system", INTENT_SYSTEM), Message("user", question)]
    last_error = "无法解析"
    for attempt in (1, 2):
        completion: Completion = client.complete(messages)
        try:
            intent, echoed = parse_intent_payload(extract_json_object(completion.text))
        except JsonExtractError as exc:
            last_error = str(exc)
            if attempt == 2:
                raise
            messages = [
                *messages,
                Message("assistant", completion.text),
                Message("user", REPAIR_USER),
            ]
            continue
        return Intent(intent=intent, question=echoed, attempts=attempt)
    raise JsonExtractError(last_error)
