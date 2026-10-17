from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


class AsyncGitHubAPIError(RuntimeError):
    """异步 GitHub API 调用异常基类。"""


class AsyncGitHubNotFoundError(AsyncGitHubAPIError):
    """资源不存在（HTTP 404）。"""

    def __init__(self, path: str) -> None:
        super().__init__(f"HTTP 404 Not Found: {path}")
        self.path = path


class AsyncGitHubRateLimitError(AsyncGitHubAPIError):
    """触发 GitHub 速率限制：

    - HTTP 429：primary rate limit 耗尽；
    - HTTP 403 且 ``X-RateLimit-Remaining: 0``：secondary rate limit（滥用检测、
      按资源的细粒度限流等）。

    携带的字段可直接驱动客户端退避策略。
    """

    def __init__(
        self,
        status: int,
        path: str,
        retry_after: float | None,
        reset_at: int | None,
        limit: int | None,
        remaining: int | None,
    ) -> None:
        super().__init__(f"HTTP {status} rate limited: {path}")
        self.status = status
        self.path = path
        self.retry_after = retry_after
        self.reset_at = reset_at
        self.limit = limit
        self.remaining = remaining

    @property
    def reset_in(self) -> float | None:
        """距离配额重置的秒数，若 ``reset_at`` 缺失则返回 ``None``。"""
        if self.reset_at is None:
            return None
        from time import time

        return max(0.0, float(self.reset_at) - time())


@dataclass(frozen=True)
class RateLimitInfo:
    """从 GitHub 响应头里提取的速率限制信息快照。"""

    limit: int | None
    remaining: int | None
    reset_at: int | None
    used: int | None

    @classmethod
    def from_headers(cls, headers: httpx.Headers) -> RateLimitInfo:
        return cls(
            limit=_safe_int(headers.get("x-ratelimit-limit")),
            remaining=_safe_int(headers.get("x-ratelimit-remaining")),
            reset_at=_safe_int(headers.get("x-ratelimit-reset")),
            used=_safe_int(headers.get("x-ratelimit-used")),
        )

    @property
    def is_exhausted(self) -> bool:
        return self.remaining == 0


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
        self.last_rate_limit: RateLimitInfo | None = None

    async def get_json(self, path: str) -> dict[str, Any]:
        if self._client.is_closed:
            raise RuntimeError("AsyncGitHubClient is closed")

        try:
            response = await self._client.get(path)
            self.last_rate_limit = RateLimitInfo.from_headers(
                response.headers
            )
            response.raise_for_status()
            data = response.json()

        except httpx.TimeoutException as exc:
            raise AsyncGitHubAPIError(
                "GitHub API request timed out"
            ) from exc

        except httpx.HTTPStatusError as exc:
            raise self._classify_http_error(exc, path) from exc

        except httpx.RequestError as exc:
            raise AsyncGitHubAPIError(
                f"GitHub API network error: {path}"
            ) from exc

        if not isinstance(data, dict):
            raise AsyncGitHubAPIError(
                "Expected a JSON object"
            )

        return data

    @staticmethod
    def _classify_http_error(
        exc: httpx.HTTPStatusError,
        path: str,
    ) -> AsyncGitHubAPIError:
        response = exc.response
        status = response.status_code

        if status == 404:
            return AsyncGitHubNotFoundError(path)

        if status == 429 or _is_secondary_rate_limit(response):
            return AsyncGitHubRateLimitError(
                status=status,
                path=path,
                retry_after=_parse_retry_after(response.headers),
                reset_at=_safe_int(
                    response.headers.get("x-ratelimit-reset")
                ),
                limit=_safe_int(
                    response.headers.get("x-ratelimit-limit")
                ),
                remaining=_safe_int(
                    response.headers.get("x-ratelimit-remaining")
                ),
            )

        return AsyncGitHubAPIError(f"HTTP {status}: {path}")

    async def aclose(self) -> None:
        await self._client.aclose()

    @property
    def closed(self) -> bool:
        return self._client.is_closed

    async def __aenter__(self) -> "AsyncGitHubClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()


def _is_secondary_rate_limit(response: httpx.Response) -> bool:
    """GitHub 用 403 + ``X-RateLimit-Remaining: 0`` 表示 secondary rate limit。"""
    if response.status_code != 403:
        return False
    return response.headers.get("x-ratelimit-remaining") == "0"


def _parse_retry_after(headers: httpx.Headers) -> float | None:
    raw = headers.get("retry-after")
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _safe_int(raw: str | None) -> int | None:
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None
