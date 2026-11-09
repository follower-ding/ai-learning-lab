"""粗略的 token 估算，只用于课堂观察，不能拿去对账。"""


def estimate_tokens(text: str) -> int:
    """中文大约一字一 token，其余字符大约四字一 token。"""

    if not text:
        return 0
    cjk = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    other = len(text) - cjk
    return cjk + (other + 3) // 4
