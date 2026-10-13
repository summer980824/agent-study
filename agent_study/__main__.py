from urllib.error import HTTPError

from pydantic import ValidationError

from pathlib import Path

from agent_study.github_user import GithubUser
from agent_study.user import User
from agent_study import task
from agent_study.config import settings


def main():
    user_1 = User.from_dict({
        "id": 1001,
        "name": "Alice",
        "preferences": {
            "language": "zh",
            "model": "gpt"
        },
        "tasks": [
            {"id": 1, "title": "learn Python", "status": "done"},
            {"id": 2, "title": "learn RAG"}
        ]
    })

    user_1.name = "Summer"
    user_1.preferences.model = "Kimi"

    try:
        user_1.add_task("learn Agent")
        user_1.add_task("learn Java", status="done")
        user_1.add_task("")
    except ValueError as e:
        print(f"Error: {e}")
    finally:
        print("Add tasks complete")

    user_1.print_user_summary(last_login="10/6", modified=True)
    user_1.print_tasks_not_done()

    user_2 = User(id=1002, name="Tom")
    user_2.add_task("learn sth")

    file_path = Path("agent_study") / "task.txt"
    task.save_tasks(user_1.pending_tasks(), file_path)
    print(task.load_tasks(file_path))
    task.append_task("adhoc task", file_path)
    print(task.load_tasks(file_path))

    user_3 = User.load('{"id": 1003, "name": "Jack"}')
    try:
        user_3.add_task(" ")
    except ValidationError as ve:
        print(f"Error: {ve}")

    try:
        user_3.add_task("1", "not sure")
    except ValidationError as ve:
        print(f"Error: {ve}")

    print(settings.MODEL_NAME)


def restful():
    try:
        user1 = GithubUser.from_username("summer980824")
        print(user1.model_dump())

        user2 = GithubUser.from_username("summer92222")
        print(user2.model_dump())
    except HTTPError as e:
        print(f"{e.code}:{e.reason}")
    except ValidationError as ve:
        print(f"Error: {ve}")


if __name__ == "__main__":
    main()
    restful()
