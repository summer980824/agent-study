import json
from typing import Literal
from pydantic import BaseModel, Field


class Preferences(BaseModel):
    language: str = "zh"
    model: str = "gpt"


class TaskItem(BaseModel):
    id: int
    title: str = Field(min_length=1)
    done: bool = Field(default=False, strict=True)
    priority: int = Field(ge=1, le=5, default=1)
    status: Literal["todo", "doing", "done"] = "todo"


class User(BaseModel):
    id: int
    name: str
    preferences: Preferences = Preferences()
    tasks: list[TaskItem] = []
    active: bool = True

    def pending_tasks(self) -> list[TaskItem]:
        return [t for t in self.tasks if not t.done]

    def add_task(self, title: str, done: bool = False) -> None:
        if not isinstance(title, str):
            raise TypeError("title must be a string")
        
        next_id = max((t.id for t in self.tasks), default=0) + 1
        self.tasks.append(TaskItem(id=next_id, title=title, done=done))

    def print_user_summary(self, **keywords):
        print(json.dumps(self.model_dump(), indent=2))
        for kw in keywords:
            print(kw, ":", keywords[kw])

    def print_tasks_not_done(self):
        print("Tasks not done:", [t.model_dump_json() for t in self.pending_tasks()])

    def to_dict(self) -> dict:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict) -> "User":
        return cls.model_validate(data)

    @classmethod
    def load(cls, s: str) -> "User":
        return cls.model_validate_json(s)
