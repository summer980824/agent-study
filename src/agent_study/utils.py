import json

def print_user_summary(user, **keywords):
    s = json.dumps(user, indent=2)
    print(s)
    for kw in keywords:
        print(kw, ":", keywords[kw])


def print_tasks_not_done(pending_tasks):
    print("Tasks not done:", json.dumps(pending_tasks))


def load(s):
    data = json.loads(s)
    print(data["name"])