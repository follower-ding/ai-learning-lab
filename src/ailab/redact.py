"""整理可能被写进日志或异常的文本，去掉常见密钥样式。"""

import re

_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_\-]+"),
    re.compile(r"Bearer\s+\S+", flags=re.IGNORECASE),
)


def redact(text: str) -> str:
    redacted = text
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub("[redacted]", redacted)
    return redacted
