from agent_study import user, task


def main():
    user_1 = {
        "id": 1001,
        "name": "Alice",
        "preferences": {
            "language": "zh",
            "model": "gpt"
        },
        "tasks": [
            {"id": 1, "title": "learn Python", "done": True},
            {"id": 2, "title": "learn RAG", "done": False}
        ]
    }

    user.update_user_name(user_1, "Summer")
    user.update_user_model(user_1, "Kimi")

    task.add_task(user_1, "learn Agent")
    task.add_task(user_1, "learn Java", True)

    user.print_user_summary(user_1, last_login="10/6", modified=True)
    user.print_tasks_not_done(user_1)


if __name__ == "__main__":
    main()