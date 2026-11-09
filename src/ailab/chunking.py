"""把长规章切成带重叠的小块。中文按字符计，不按空格分词。"""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str


def chunk_text(text: str, *, size: int = 80, overlap: int = 20) -> list[Chunk]:
    """先按空行分段落。段落放得下就整段保留，放不下再滑窗。

    窗口尽量在句号、问号、叹号或分号处切开。相邻窗口共享 ``overlap`` 个字符，
    避免一句话刚好被切在边界上之后，两边都看不到完整的半句。
    """

    if not isinstance(size, int) or size <= 0:
        raise ValueError("size 必须是正整数")
    if not isinstance(overlap, int) or overlap < 0 or overlap >= size:
        raise ValueError("overlap 必须是整数，且满足 0 <= overlap < size")

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[Chunk] = []
    for paragraph in paragraphs:
        start = 0
        steps = 0
        while start < len(paragraph):
            steps += 1
            if steps > len(paragraph) + 2:
                raise RuntimeError("切块没有前进，请检查 size 和 overlap")
            end = _window_end(paragraph, start, size)
            piece = paragraph[start:end].strip()
            if piece:
                chunks.append(Chunk(index=len(chunks), text=piece))
            if end >= len(paragraph):
                break
            next_start = end - overlap
            if next_start <= start:
                next_start = end
            start = next_start
    return chunks


def _window_end(text: str, start: int, size: int) -> int:
    hard_end = min(start + size, len(text))
    if hard_end >= len(text):
        return len(text)
    window = text[start:hard_end]
    half = len(window) // 2
    best = -1
    for mark in ("。", "！", "？", "；"):
        position = window.rfind(mark)
        if position > best:
            best = position
    if best >= half:
        return start + best + 1
    return hard_end
