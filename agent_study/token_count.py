import tiktoken


def count_tokens(
    text: str | None,
    encoding_name: str = "cl100k_base",
) -> int:
    """返回指定编码下的 Token 数量。"""
    if text is None:
        return 0
    encoding = tiktoken.get_encoding(encoding_name)
    token_ids = encoding.encode(text)
    return len(token_ids)
