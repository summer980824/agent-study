from typing import Any

import httpx


class AsyncGitHubAPIError(RuntimeError):
    """异步 GitHub API 调用异常。"""


class AsyncGitHubClient:
    def __init__(
        self,
        base_url: str = "https://api.github.com",
        timeout: float = 10.0,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout),
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "agent-study",
            },
        )

    async def get_json(self, path: str) -> dict[str, Any]:
        if self._client.is_closed:
            raise RuntimeError("AsyncGitHubClient is closed")

        try:
            response = await self._client.get(path)
            response.raise_for_status()
            data = response.json()

        except httpx.TimeoutException as exc:
            raise AsyncGitHubAPIError(
                "GitHub API request timed out"
            ) from exc

        except httpx.HTTPStatusError as exc:
            raise AsyncGitHubAPIError(
                f"HTTP {exc.response.status_code}: {path}"
            ) from exc

        except httpx.RequestError as exc:
            raise AsyncGitHubAPIError(
                f"GitHub API network error: {path}"
            ) from exc

        if not isinstance(data, dict):
            raise AsyncGitHubAPIError(
                "Expected a JSON object"
            )

        return data

    async def aclose(self) -> None:
        await self._client.aclose()

    @property
    def closed(self) -> bool:
        return self._client.is_closed

    async def __aenter__(self) -> "AsyncGitHubClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()