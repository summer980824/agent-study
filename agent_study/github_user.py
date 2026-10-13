"""RESTful GET 请求示例：获取 GitHub 用户信息并打印返回。"""
import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from pydantic import BaseModel, Field, field_validator


class GithubUser(BaseModel):
    id: int
    login: str
    url: str
    public_repos: int
    created_at: str


    def get_github_user(username: str) -> dict:
        url = f"https://api.github.com/users/{username}"
        req = Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "agent-study"})
        with urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
       

    @classmethod
    def from_username(cls, name: str) -> "GithubUser":
        data = cls.get_github_user(name)
        return cls.model_validate(data)

