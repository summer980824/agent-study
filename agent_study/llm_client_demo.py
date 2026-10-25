from agent_study.llm_client import chat


def main() -> None:
    response = chat(prompt="今天三清山天气如何？", reasoning_effort="low")
    print(response)


if __name__ == "__main__":
    main()