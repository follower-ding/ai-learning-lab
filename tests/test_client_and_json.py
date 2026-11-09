"""第 01、02 课：调用和 JSON。"""

import json

import pytest

from ailab.client import MockChatClient, client_from_env
from ailab.errors import JsonExtractError, LabError
from ailab.json_prompt import classify_intent, extract_json_object
from ailab.messages import Message
from ailab.mock_front_desk import PLAIN_FALLBACK


def test_mock_answers_checkin_without_network() -> None:
    client = MockChatClient()
    completion = client.complete(
        [
            Message("system", "你是岸港酒店实验前台。只回答课程资料里有的规章。"),
            Message("user", "入住时间是什么时候？"),
        ]
    )
    assert "14:00" in completion.text
    assert completion.model == "mock-hotel-front-desk"
    assert completion.prompt_tokens > 0
    assert completion.completion_tokens > 0


def test_mock_does_not_invent_unknown_topic() -> None:
    completion = MockChatClient().complete([Message("user", "游泳池几点开放？")])
    assert completion.text == PLAIN_FALLBACK
    assert "开放" not in completion.text


def test_script_does_not_silently_fall_back() -> None:
    client = MockChatClient(script=["只有一句"])
    client.complete([Message("user", "你好")])
    with pytest.raises(LabError):
        client.complete([Message("user", "还有吗")])


def test_extracts_fenced_json() -> None:
    payload = extract_json_object('说明如下：\n```json\n{"intent": "check_in", "question": "入住"}\n```')
    assert payload["intent"] == "check_in"


def test_extracts_object_with_brace_inside_string() -> None:
    payload = extract_json_object('前面 {"text": "看这个 } 符号", "n": 1} 后面')
    assert payload == {"text": "看这个 } 符号", "n": 1}


def test_rejects_unclosed_object_and_top_level_array() -> None:
    with pytest.raises(JsonExtractError):
        extract_json_object('{"a": 1')
    with pytest.raises(JsonExtractError):
        extract_json_object("[1, 2]")


def test_cancel_wins_over_checkin_keyword() -> None:
    intent = classify_intent(MockChatClient(), "入住日前可以取消吗？")
    assert intent.intent == "cancel"
    assert intent.attempts == 1


def test_unknown_topic_is_an_intent() -> None:
    intent = classify_intent(MockChatClient(), "附近有没有地铁？")
    assert intent.intent == "unknown"
    assert intent.question == "附近有没有地铁？"


def test_repairs_invalid_json_once() -> None:
    client = MockChatClient(
        script=[
            "我觉得是入住。",
            json.dumps({"intent": "check_in", "question": "入住时间是什么时候？"}, ensure_ascii=False),
        ]
    )
    intent = classify_intent(client, "入住时间是什么时候？")
    assert intent.intent == "check_in"
    assert intent.attempts == 2
    assert len(client.calls) == 2


def test_rejects_intent_outside_the_list_then_accepts_repair() -> None:
    client = MockChatClient(
        script=[
            '{"intent": "swim", "question": "游泳池"}',
            '{"intent": "unknown", "question": "游泳池"}',
        ]
    )
    intent = classify_intent(client, "游泳池几点开放？")
    assert intent.intent == "unknown"
    assert intent.attempts == 2


def test_second_bad_payload_raises() -> None:
    client = MockChatClient(script=["不是 json", "还是不是"])
    with pytest.raises(JsonExtractError):
        classify_intent(client, "入住时间是什么时候？")


def test_env_without_key_stays_on_mock() -> None:
    assert isinstance(client_from_env(environ={}), MockChatClient)
