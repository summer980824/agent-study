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


def test_whitespace_only_string_preserves_token_count():
    assert count_tokens(" \t\n") == 1


def test_o200k_base_encoding():
    assert count_tokens("hello world", encoding_name="o200k_base") == 2


def test_invalid_encoding_name_raises_value_error():
    with pytest.raises(ValueError, match="Unknown encoding"):
        count_tokens("text", encoding_name="not-a-real-encoding")
