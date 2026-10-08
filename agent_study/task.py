def get_pending_tasks(user):
    s = []
    for t in user["tasks"]:
        if not t["done"]:
            s.append(t)

    return s

def add_task(user, title, done=False):
    if not isinstance(title, str):
        raise TypeError("title must be a string")

    if not title.strip():
        raise ValueError("title cannot be empty")

    if not user["tasks"]:
        raise ValueError("tasks must be initialized")

    next_id = max(t["id"] for t in user["tasks"]) + 1
    user["tasks"].append({"id": next_id, "title": title, "done": done})