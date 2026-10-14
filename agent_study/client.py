import httpx
from pydantic import BaseModel, model_validator


class GitHubAPIError(Exception):
    """封装所有 GitHub API 失败，调用方只需捕获这一个。"""


class GitHubClient(BaseModel):
    base_url: str
    token: str = ""

    @model_validator(mode="after")
    def _strip_trailing_slash(self):
        self.base_url = self.base_url.rstrip("/")
        return self

    def _session(self) -> httpx.Client:
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "agent-study",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return httpx.Client(base_url=self.base_url,
                            headers=headers,
                            timeout=10)

    def http_get(self, path: str, params: dict | None = None) -> httpx.Response:
        try:
            with self._session() as s:
                r = s.get(path, params=params)
                r.raise_for_status()
                return r
        except httpx.HTTPStatusError as e:
            raise GitHubAPIError(
                f"{e.response.status_code} {path}: {e.response.text[:200]} retry-after: {e.response.headers.get("Retry-After")}"
            ) from e
        except httpx.RequestError as e:
            raise GitHubAPIError(f"网络错误 {path}: {e}") from e