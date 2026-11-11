"""每课练习都能在没有密钥的情况下跑完。"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

LESSONS = [
    "lessons/01-llm-api/exercise.py",
    "lessons/02-prompt-json/exercise.py",
    "lessons/03-embedding-chunking/exercise.py",
    "lessons/04-vector-store/exercise.py",
    "lessons/05-rag-cite-or-refuse/exercise.py",
    "lessons/06-tool-calling/exercise.py",
    "lessons/07-agent-loop/exercise.py",
    "lessons/08-tiny-eval/exercise.py",
    "lessons/09-hybrid-retrieval/exercise.py",
    "lessons/10-rerank/exercise.py",
    "lessons/11-clarify-missing-slot/exercise.py",
    "lessons/12-streaming-output/exercise.py",
    "lessons/13-material-is-data/exercise.py",
    "lessons/14-minimal-trace/exercise.py",
]


@pytest.mark.parametrize("relative", LESSONS)
def test_lesson_readme_has_the_lab_sections(relative: str) -> None:
    readme = ROOT / Path(relative).parent / "README.md"
    text = readme.read_text(encoding="utf-8")
    for section in ("学习目标", "概念笔记", "跑起来", "常见坑", "口头自查"):
        assert section in text


@pytest.mark.parametrize("relative", LESSONS)
def test_exercise_runs(relative: str) -> None:
    number = relative.split("/")[1][:2]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("OPENAI_API_KEY", None)
    completed = subprocess.run(
        [sys.executable, relative],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr + "\n" + completed.stdout
    assert f"第 {number} 课" in completed.stdout
