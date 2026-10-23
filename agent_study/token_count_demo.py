import json

from agent_study.token_count import count_tokens

TEXTS = [
    "Large language models process text as tokens.",
    "大型语言模型会将文本转换成 Token。",
    """
def add(a, b):
    return a + b
""",
    '{"user_id": 1001, "status": "active"}',
]


def main() -> None:
    results = []
    for text in TEXTS:
        n_chars = len(text)
        n_tokens = count_tokens(text)
        results.append(
            {
                "text": text,
                "characters": n_chars,
                "tokens": n_tokens,
                "tokens_per_char": round(n_tokens / n_chars, 4) if n_chars else None,
            }
        )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
