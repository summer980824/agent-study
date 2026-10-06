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
        if not t["done"]:
            s.append(t)

    return s

def add_task(user, title, done=False):
    next_id = max(t["id"] for t in user["tasks"]) + 1
    user["tasks"].append({"id": next_id, "title": title, "done": done})

add_task(user, "learn Agent")
add_task(user, "learn Java", True)

def print_user_summary(user, **keywords):
    print(user["id"], ":", user["name"])
    for kw in keywords:
        print(kw, ":", keywords[kw])

print_user_summary(user, last_login="10/6", modified=True)

s = json.dumps(user, indent=2)
print(s)
print(json.dumps(get_pending_tasks(user)))


data = json.loads(s)
print(data["name"])

    