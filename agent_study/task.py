from pathlib import Path

from agent_study import util
from agent_study.user import TaskItem


def save_tasks(tasks: list[TaskItem], file_path: Path | str) -> None:
    txt = "".join(t.title + "\n" for t in tasks)
    util.write_file(file_path, txt)


def load_tasks(file_path: Path | str) -> list[str]:
    txt = util.read_file(file_path)
    if txt:
        # split each line, drop empty lines
        return [line for line in txt.splitlines() if line.strip()]
    else:
        return []


def append_task(title: str, file_path: Path | str) -> None:
    util.append_file(file_path, title + "\n")
