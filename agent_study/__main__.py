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

    try:
        task.add_task(user_1, "learn Agent")
        task.add_task(user_1, "learn Java", True)
        task.add_task(user_1, "")
    except ValueError as e:
        print(f"Error: {e}")
    finally:
        print("Add tasks complete")

    user.print_user_summary(user_1, last_login="10/6", modified=True)
    user.print_tasks_not_done(user_1)

    
    user_2 = {
        "id": 1002,
        "name": "Tom",
        "preferences": {
            "language": "zh",
            "model": "gpt"
        },
        "tasks": [
        ]
    }

    try:
        task.add_task(user_2, "learn sth")
    except ValueError as e:
        print(f"Error: {e}")

    file_path = "agent_study/task.txt"
    task.save_tasks(user.get_pending_tasks(user_1), file_path)
    print(task.load_tasks(file_path))
    task.append_task("adhoc task", file_path)
    print(task.load_tasks(file_path))


if __name__ == "__main__":
    main()