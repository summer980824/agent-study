from agent_study.anthropic_chat import chat_message


def main() -> None:
    message = chat_message(
        prompt="今天三清山天气如何？",
        reasoning_effort="low",
    )
    print("".join(block.text for block in message.content if block.type == "text"))
    print(f"stop_reason={message.stop_reason} usage={message.usage}")


if __name__ == "__main__":
    main()
