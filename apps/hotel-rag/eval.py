"""调用现有的 run_eval()，并用同一次作答附上追踪。默认离线 Mock。"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP_DIR = Path(__file__).resolve().parent
for _entry in (ROOT / "src", APP_DIR):
    _text = str(_entry)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from loop import add_common_args, run_live_or_mock, run_recorded_eval


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="岸港酒店小评测。默认离线 Mock，不改用例集。")
    add_common_args(parser)
    args = parser.parse_args(argv)
    return run_live_or_mock(args.live, lambda client: run_recorded_eval(client, args.trace_dir))


if __name__ == "__main__":
    raise SystemExit(main())
