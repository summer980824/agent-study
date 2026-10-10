from agent_study.user import User
from agent_study import task


def main():
    user_1 = User.from_dict({
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
    })

    user_1.name = "Summer"
    user_1.preferences.model = "Kimi"

    try:
        user_1.add_task("learn Agent")
        user_1.add_task("learn Java", True)
        user_1.add_task("")
    except ValueError as e:
        print(f"Error: {e}")
    finally:
        print("Add tasks complete")

    user_1.print_user_summary(last_login="10/6", modified=True)
    user_1.print_tasks_not_done()

    user_2 = User(id=1002, name="Tom")
    user_2.add_task("learn sth")

    file_path = "agent_study/task.txt"
    task.save_tasks(user_1.pending_tasks(), file_path)
    print(task.load_tasks(file_path))
    task.append_task("adhoc task", file_path)
    print(task.load_tasks(file_path))

    user_3 = User.load('{"id": 1003, "name": "Jack"}')
    print(user_3)


if __name__ == "__main__":
    main()
