"""四条固定案例，一次跑完。默认离线 Mock。"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP_DIR = Path(__file__).resolve().parent
for _entry in (ROOT / "src", APP_DIR):
    _text = str(_entry)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from loop import add_common_args, run_demo, run_live_or_mock


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="岸港酒店四条固定案例。默认离线 Mock。")
    add_common_args(parser)
    args = parser.parse_args(argv)
    return run_live_or_mock(args.live, lambda client: run_demo(client, args.trace_dir))


if __name__ == "__main__":
    raise SystemExit(main())
