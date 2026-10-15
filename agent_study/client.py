import httpx
from pydantic import BaseModel, PrivateAttr, model_validator


class GitHubAPIError(Exception):
    """封装所有 GitHub API 失败，调用方只需捕获这一个。"""


class GitHubClient(BaseModel):
    base_url: str
    token: str = ""

    _client: httpx.Client = PrivateAttr()

    @model_validator(mode="after")
    def _build_client(self):
        self.base_url = self.base_url.rstrip("/")
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "agent-study",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        self._client = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=10,
        )
        return self

    def http_get(self, path: str, params: dict | None = None) -> httpx.Response:
        try:
            r = self._client.get(path, params=params)
            r.raise_for_status()
            return r
        except httpx.HTTPStatusError as e:
            retry_after = e.response.headers.get("Retry-After")
            message = (
                f"{e.response.status_code} {path}: "
                f"{e.response.text[:200]} "
                f"retry-after: {retry_after}"
            )
            raise GitHubAPIError(message) from e
        except httpx.RequestError as e:
            raise GitHubAPIError(f"网络错误 {path}: {e}") from e

    def close(self) -> None:
        self._client.close()