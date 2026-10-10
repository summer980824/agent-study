from agent_study import util


def get_pending_tasks(tasks: list[dict]) -> list[dict]:
    return [task for task in tasks if not task["done"]]


def add_task(user, title, done=False):
    if not isinstance(title, str):
        raise TypeError("title must be a string")

    if not title.strip():
        raise ValueError("title cannot be empty")

    if not user["tasks"]:
        raise ValueError("tasks must be initialized")

    next_id = max(t["id"] for t in user["tasks"]) + 1
    user["tasks"].append({"id": next_id, "title": title, "done": done})


def save_tasks(tasks: list[dict], file_path_str: str) -> None:
    txt = "";
    for task in tasks:
        txt += task['title'] + "\n"
    util.write_file(file_path_str, txt)


def load_tasks(file_path_str: str) -> list[str]:
    txt = util.read_file(file_path_str)
    if txt:
        #split each line, drop empty lines
        return [line for line in txt.splitlines() if line.strip()]
    else:
        return []


def append_task(task: str, file_path_str: str) -> None:
    util.append_file(file_path_str, task + "\n")