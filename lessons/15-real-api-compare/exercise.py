"""第 15 课练习：用第 08 课的用例对照一次真实接口。没有密钥就跳过。"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ailab.client import live_client_from_env, read_chat_limits
from ailab.errors import ConfigError, LabError
from ailab.evals import EvalReport, format_report, load_cases, run_eval
from ailab.redact import redact
from ailab.tools import hotel_tools

_GROUPS = ("引用", "拒绝", "工具")


def main() -> int:
    print("第 15 课 · 换成真实接口的对照")
    print("对照第 08 课的 9 条用例。检索仍用字面向量，只把 complete 换成真实接口。")
    allowed = list(hotel_tools())
    print("允许列表：" + "、".join(allowed))
    if allowed != ["lookup_policy", "quote_late_checkout"]:
        print("允许列表必须正好是这两只工具。")
        return 1
    print("拒绝那两条在分数不够时不会调用模型。离线如此，换上真实客户端也如此。")

    cases = load_cases()
    offline = run_eval()
    print("离线 Mock：")
    print(format_report(offline))
    offline_buckets = _buckets(offline, cases)
    print("离线：" + _phrase(offline_buckets) + "。")
    if offline.passed != offline.total:
        print("离线 Mock 应当全部通过。真实接口对照还没开始。")
        return 1

    try:
        client = live_client_from_env()
    except ConfigError as exc:
        print(f"真实接口没有开始：{redact(str(exc))}")
        print("这次没有调用网络，也没有真实接口的结果。")
        return 0
    if client is None:
        _print_skip()
        return 0

    print(f"真实接口：超时 {client.timeout:g} 秒，max_tokens={client.max_tokens}。")
    try:
        live = run_eval(client=client)
    except LabError as exc:
        print(f"真实接口没有完成：{redact(str(exc))}")
        print("没有记下对照结果。")
        return 0
    finally:
        client.close()

    print(redact(format_report(live)))
    live_buckets = _buckets(live, cases)
    print("真实接口这一侧：" + _phrase(live_buckets) + "。")
    print("对照：" + _compare(offline_buckets, live_buckets) + "。")
    return 0


def _print_skip() -> None:
    try:
        timeout, max_tokens = read_chat_limits(os.environ)
    except ConfigError as exc:
        print(f"没有 OPENAI_API_KEY，跳过真实接口。时限没有读成：{redact(str(exc))}")
        print("这次没有调用网络，也没有真实接口的结果。")
        return
    print(
        "没有 OPENAI_API_KEY，跳过真实接口。"
        f"超时 {timeout:g} 秒，max_tokens={max_tokens}。"
        "这次没有调用网络，也没有真实接口的结果。"
    )
    print("把 .env.example 复制成 .env，在本地填上密钥后再运行本练习。第 01 课到第 14 课不需要密钥。")


def _buckets(report: EvalReport, cases) -> dict[str, list[bool]]:
    expect = {case.id: case.expect for case in cases}
    buckets: dict[str, list[bool]] = {name: [] for name in _GROUPS}
    for check in report.checks:
        kind = expect[check.case_id]
        if kind == "cite":
            buckets["引用"].append(check.passed)
        elif kind == "refuse":
            buckets["拒绝"].append(check.passed)
        else:
            buckets["工具"].append(check.passed)
    return buckets


def _phrase(buckets: dict[str, list[bool]]) -> str:
    parts = []
    for name in _GROUPS:
        flags = buckets[name]
        held = bool(flags) and all(flags)
        parts.append(f"{name}{'成立' if held else '不成立'}")
    return "，".join(parts)


def _compare(before: dict[str, list[bool]], after: dict[str, list[bool]]) -> str:
    parts = []
    for name in _GROUPS:
        was = bool(before[name]) and all(before[name])
        now = bool(after[name]) and all(after[name])
        if was and now:
            parts.append(f"{name}仍成立")
        elif was and not now:
            parts.append(f"{name}有变化")
        else:
            parts.append(f"{name}不成立")
    return "，".join(parts)


if __name__ == "__main__":
    raise SystemExit(main())
