from agent_study.task import get_pending_tasks
import json


def print_user_summary(user, **keywords):
    s = json.dumps(user, indent=2)
    print(s)
    for kw in keywords:
        print(kw, ":", keywords[kw])


def print_tasks_not_done(user):
    print("Tasks not done:", json.dumps(get_pending_tasks(user)))


def load(s):
    data = json.loads(s)
    print(data["name"])


def update_user_name(user, name):
    user["name"] = name


def update_user_model(user, model):
    user["preferences"]["model"] = model

