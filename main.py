import json

user = {
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

user["name"] = "Peter"
user["preferences"]["model"] = "copilot"


def get_pending_tasks(user):
    print("Tasks not done:")
    s = []
    for t in user["tasks"]:
        if t["done"] == False:
            s.append(t)

    return s

def add_task(user, title):
    next_id = max(t["id"] for t in user["tasks"]) + 1
    user["tasks"].append({"id": next_id, "title": title, "done": False})

add_task(user, "learn Agent")

s = json.dumps(user, indent=2)
print(s)
print(json.dumps(get_pending_tasks(user)))




    