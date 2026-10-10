import json
from dataclasses import dataclass, field, asdict


@dataclass
class Preferences:
    language: str = "zh"
    model: str = "gpt"


@dataclass
class TaskItem:
    id: int
    title: str
    done: bool = False


@dataclass
class User:
    id: int
    name: str
    preferences: Preferences = field(default_factory=Preferences)
    tasks: list[TaskItem] = field(default_factory=list)
    active: bool = True

    def pending_tasks(self) -> list[TaskItem]:
        return [t for t in self.tasks if not t.done]

    def add_task(self, title: str, done: bool = False) -> None:
        if not isinstance(title, str):
            raise TypeError("title must be a string")
        if not title.strip():
            raise ValueError("title cannot be empty")
        next_id = max((t.id for t in self.tasks), default=0) + 1
        self.tasks.append(TaskItem(next_id, title, done))

    def print_user_summary(self, **keywords):
        print(json.dumps(asdict(self), indent=2))
        for kw in keywords:
            print(kw, ":", keywords[kw])

    def print_tasks_not_done(self):
        print("Tasks not done:", json.dumps([asdict(t) for t in self.pending_tasks()]))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "User":
        return cls(
            id=data["id"],
            name=data["name"],
            preferences=Preferences(**data.get("preferences", {})),
            tasks=[TaskItem(**t) for t in data.get("tasks", [])],
        )

    @classmethod
    def load(cls, s: str) -> "User":
        return cls.from_dict(json.loads(s))
