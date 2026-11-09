"""调用接口时来回传递的消息。"""

from dataclasses import dataclass

ROLES = ("system", "user", "assistant", "tool")


@dataclass(frozen=True)
class Message:
    """一条对话消息。

    ``tool`` 表示代码执行工具后写回来的结果，不是模型自己编的。
    """

    role: str
    content: str
    name: str = ""

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"未知 role: {self.role}")


@dataclass(frozen=True)
class Completion:
    """一次补全的文本和用量。用量在 Mock 里是估算，不是账单。"""

    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
