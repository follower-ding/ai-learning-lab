"""问一句。用 --rag 或 --agent 选择路径，默认检索。不按问法自动分流。"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP_DIR = Path(__file__).resolve().parent
for _entry in (ROOT / "src", APP_DIR):
    _text = str(_entry)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from loop import add_common_args, run_ask, run_live_or_mock


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="问一句。默认走检索。要走工具请写 --agent。")
    parser.add_argument("question", nargs="?", default="", help="客人的问题")
    parser.add_argument("--rag", action="store_true", help="走检索。这是默认。")
    parser.add_argument("--agent", action="store_true", help="走工具循环。")
    add_common_args(parser)
    args = parser.parse_args(argv)
    if args.rag and args.agent:
        print("请只写 --rag 或 --agent。这里不按问题自动分流。")
        return 2
    if not args.question.strip():
        print("请在命令后面写上客人的问题。")
        return 2
    kind = "agent" if args.agent else "rag"
    return run_live_or_mock(
        args.live,
        lambda client: run_ask(args.question, kind, client, args.trace_dir),
    )


if __name__ == "__main__":
    raise SystemExit(main())
