from agent_study.llm_client import chat_stream, chat_structured
from agent_study.structured_output import TaskSchema


def main() -> None:
    # 流式：边收边吐，完整正文和响应元信息留在 stream.result 上
    stream = chat_stream("用一句话解释 Agent", reasoning_effort="low")
    for delta in stream:
        print(delta, end="", flush=True)
    print()
    print("finish_reason:", stream.result.finish_reason)
    print("usage:", stream.result.usage)
    # done_received 为 False 说明没收到 [DONE]：流可能中途断了，正文只有半截
    print("done_received:", stream.done_received)

    # 自由文本 -> 结构化任务：模型需要自己抽取字段，不是抄输入
    note = (
        "下周三之前把三清山这几天的天气查一下，"
        "把气温、降水和风力整理成一段，发到项目群里，大概半小时能搞定。"
    )
    task = chat_structured(prompt=note, reasoning_effort="low")
    print(task.model_dump_json(indent=2))

    # JSON 输入 -> 结构化任务：输入字段和 schema 不同名，模型需要做映射
    payload = {
        "summary": "查三清山天气并发群",
        "deadline": "下周三",
        "effort_minutes": 30,
    }
    task2 = chat_structured(prompt=payload, reasoning_effort="low")
    print(task2.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
