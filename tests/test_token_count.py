import pytest

from agent_study.token_count import count_tokens

# (text, 字符数, token 数) —— cl100k_base 编码
CASES = [
    ("Large language models process text as tokens.", 45, 8),
    ("大型语言模型会将文本转换成 Token。", 20, 15),
    ("def add(a, b):\n    return a + b\n", 32, 12),
    ('{"user_id": 1001, "status": "active"}', 37, 14),
]


@pytest.mark.parametrize("text, chars, tokens", CASES)
def test_count_tokens_known_texts(text: str, chars: int, tokens: int):
    assert len(text) == chars
    assert count_tokens(text) == tokens


def test_empty_string_returns_zero_tokens():
    assert count_tokens("") == 0


def test_none_returns_zero_tokens():
    assert count_tokens(None) == 0
