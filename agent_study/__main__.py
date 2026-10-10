from agent_study.user import User
from agent_study import task


def main():
    user_1 = User({
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

    user_1.update_user_name("Summer")
    user_1.update_user_model("Kimi")

    try:
        task.add_task(user_1.user, "learn Agent")
        task.add_task(user_1.user, "learn Java", True)
        task.add_task(user_1.user, "")
    except ValueError as e:
        print(f"Error: {e}")
    finally:
        print("Add tasks complete")

    user_1.print_user_summary(last_login="10/6", modified=True)
    user_1.print_tasks_not_done()

    user_2 = User({
        "id": 1002,
        "name": "Tom",
        "preferences": {
            "language": "zh",
            "model": "gpt"
        },
        "tasks": [
        ]
    })

    print(user_2)

    try:
        task.add_task(user_2.user, "learn sth")
    except ValueError as e:
        print(f"Error: {e}")


    file_path = "agent_study/task.txt"
    task.save_tasks(task.get_pending_tasks(user_1.user['tasks']), file_path)
    print(task.load_tasks(file_path))
    task.append_task("adhoc task", file_path)
    print(task.load_tasks(file_path))


if __name__ == "__main__":
    main()
