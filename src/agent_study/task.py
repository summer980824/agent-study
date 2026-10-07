def get_pending_tasks(user):
    s = []
    for t in user["tasks"]:
        if not t["done"]:
            s.append(t)

    return s

def add_task(user, title, done=False):
    next_id = max(t["id"] for t in user["tasks"]) + 1
    user["tasks"].append({"id": next_id, "title": title, "done": done})