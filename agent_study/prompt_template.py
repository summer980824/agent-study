from agent_study.text_cleaning import clean_prompt_text


def build_prompt(
    instruction: str,
    context: str,
    output_format: str,
) -> str:
    instruction = clean_prompt_text(instruction)
    context = clean_prompt_text(context)
    output_format = clean_prompt_text(output_format)

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
