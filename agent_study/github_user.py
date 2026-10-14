"""RESTful GET 请求示例：获取 GitHub 用户信息并打印返回。"""
import json
from datetime import datetime
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from pydantic import BaseModel, Field, HttpUrl, field_validator



def get_github_user(username: str) -> dict:
    url = f"https://api.github.com/users/{username}"
    req = Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "agent-study"})
    with urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


class GithubUser(BaseModel):
    id: int
    login: str
    url: HttpUrl
    public_repos: int
    created_at: datetime

    @classmethod
    def from_username(cls, name: str) -> "GithubUser":
        data = get_github_user(name)
        return cls.model_validate(data)

    @classmethod
    def from_content(cls, content: dict) -> "GithubUser":
        return cls.model_validate(content)

