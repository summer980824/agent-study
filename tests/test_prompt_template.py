import pytest

from agent_study.prompt_template import build_prompt


def test_build_prompt_returns_markdown_sections():
    assert build_prompt("Summarize the text.", "Alpha beta gamma.", "JSON") == (
        "## Instruction\n\n"
        "Summarize the text.\n\n"
        "## Context\n\n"
        "Alpha beta gamma.\n\n"
        "## Output Format\n\n"
        "JSON"
    )


def test_build_prompt_preserves_internal_whitespace():
    result = build_prompt(
        "  First line\n    indented line  ",
        "\nAlpha  beta\n\nGamma\n",
        "  key: value  ",
    )

    assert result == (
        "## Instruction\n\n"
        "First line\n    indented line\n\n"
        "## Context\n\n"
        "Alpha  beta\n\nGamma\n\n"
        "## Output Format\n\n"
        "key: value"
    )


def test_build_prompt_cleans_all_inputs():
    result = build_prompt(
        " \r\n cafe\u0301\x00 \r\n  Second\t line  ",
        "\r\nAlpha\t\t beta\u200b\r\n\r\n\r\nGamma\r\n",
        " \ufeffJSON\t ",
    )

    assert result == (
        "## Instruction\n\n"
        "café \n  Second\t line\n\n"
        "## Context\n\n"
        "Alpha\t\t beta\n\n\nGamma\n\n"
        "## Output Format\n\n"
        "JSON"
    )


def test_build_prompt_preserves_code_and_json_layout():
    result = build_prompt(
        "\ndef main():\n    if True:\n\t\tprint('hi')  \n\n\nmain()\n",
        '\n{\n    "name": "A  B",\n    "active": true\n\n\n}\n',
        '\n```json\n{\n    "sentiment": "positive"\n}\n```\n',
    )

    assert result == (
        "## Instruction\n\n"
        "def main():\n    if True:\n\t\tprint('hi')  \n\n\nmain()\n\n"
        "## Context\n\n"
        '{\n    "name": "A  B",\n    "active": true\n\n\n}\n\n'
        "## Output Format\n\n"
        '```json\n{\n    "sentiment": "positive"\n}\n```'
    )


@pytest.mark.parametrize(
    ("instruction", "output_format"),
    [
        ("", "JSON"),
        (" \n\t ", "JSON"),
        ("\x00\u200b", "JSON"),
        ("Summarize", ""),
        ("Summarize", " \n\t "),
        ("Summarize", "\x00\u200b"),
    ],
)
def test_build_prompt_requires_instruction_and_output_format(
    instruction: str,
    output_format: str,
):
    with pytest.raises(ValueError):
        build_prompt(instruction, "context", output_format)


@pytest.mark.parametrize("context", ["", " \n\t ", "\x00\u200b"])
def test_build_prompt_uses_placeholder_for_empty_context(context: str):
    result = build_prompt("Summarize", context, "JSON")

    assert "## Context\n\n（无附加上下文）\n\n## Output Format" in result
