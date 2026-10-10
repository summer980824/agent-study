from dataclasses import dataclass, field
from agent_study.task import get_pending_tasks
import json


@dataclass
class User:

    user: dict

    def print_user_summary(self, **keywords):
        s = json.dumps(self.user, indent=2)
        print(s)
        for kw in keywords:
            print(kw, ":", keywords[kw])


    def print_tasks_not_done(self):
        print("Tasks not done:", json.dumps(get_pending_tasks(self.user['tasks'])))


    @staticmethod
    def load(s):
        data = json.loads(s)
        print(data["name"])


    def update_user_name(self, name):
        self.user["name"] = name


    def update_user_model(self, model):
        self.user["preferences"]["model"] = model

