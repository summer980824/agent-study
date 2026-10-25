from agent_study.text_cleaning import clean_text, normalize_text


def _clean_prompt_part(text: str) -> str:
    return clean_text(normalize_text(text))


def build_prompt(
    instruction: str,
    context: str,
    output_format: str,
) -> str:
    instruction = _clean_prompt_part(instruction)
    context = _clean_prompt_part(context)
    output_format = _clean_prompt_part(output_format)

    if not instruction:
        raise ValueError("instruction must not be empty")
    if not output_format:
        raise ValueError("output_format must not be empty")
    if not context:
        context = "（无附加上下文）"

    return (
        f"## Instruction\n\n{instruction}\n\n"
        f"## Context\n\n{context}\n\n"
        f"## Output Format\n\n{output_format}"
    )
