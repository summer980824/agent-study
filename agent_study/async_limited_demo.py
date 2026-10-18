import asyncio
import logging
import uuid
from time import perf_counter

from agent_study.async_client import AsyncGitHubClient
from agent_study.logging_config import (
    configure_logging,
    request_context,
)

logger = logging.getLogger("agent_study.async_limited_demo")

MAX_CONCURRENCY = 3

USERNAMES = [
    "summer980824",
    "octocat",
    "torvalds",
    "github",
    "defunkt",
    "pjhyett",
]


async def fetch_user(
    client: AsyncGitHubClient,
    username: str,
    semaphore: asyncio.Semaphore,
    stats: dict[str, int],
) -> dict:
    request_id = uuid.uuid4().hex

    async with semaphore:
        with request_context(request_id):
            started = perf_counter()
            # 记录当前进入请求区域的任务数
            stats["active"] += 1
            stats["max_active"] = max(
                stats["max_active"],
                stats["active"],
            )

            try:
                logger.info(
                    "GitHub API request started",
                    extra={
                        "event": "api_request_started",
                        "username": username,
                    },
                )
                data = await client.get_json(
                    f"/users/{username}"
                )
                logger.info(
                    "GitHub API request completed",
                    extra={
                        "event": "api_request_completed",
                        "username": username,
                        "duration_ms": round(
                            (perf_counter() - started) * 1000,
                            2,
                        ),
                        "status_code": 200,
                    },
                )

                return {
                    "login": data["login"],
                    "public_repos": data["public_repos"],
                }
            except Exception as exc:
                logger.exception(
                    "GitHub API request failed",
                    extra={
                        "event": "api_request_failed",
                        "username": username,
                        "duration_ms": round(
                            (perf_counter() - started) * 1000,
                            2,
                        ),
                        "error_type": type(exc).__name__,
                    },
                )
                raise
            finally:
                stats["active"] -= 1


async def main() -> None:
    configure_logging()

    logger.info(
        "Async GitHub demo started",
        extra={"event": "demo_started"},
    )

    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    stats = {
        "active": 0,
        "max_active": 0,
    }

    started = perf_counter()

    async with AsyncGitHubClient() as client:
        results = await asyncio.gather(
            *(
                fetch_user(
                    client,
                    username,
                    semaphore,
                    stats,
                )
                for username in USERNAMES
            ),
            return_exceptions=True,
        )

    for username, result in zip(USERNAMES, results):
        if isinstance(result, BaseException):
            print(f"{username}: FAILED - {result}")
            continue

        print(
            f"{result['login']}: "
            f"{result['public_repos']} repositories"
        )

    elapsed = perf_counter() - started

    print(f"Concurrency limit: {MAX_CONCURRENCY}")
    print(f"Max observed concurrency: {stats['max_active']}")
    print(f"Elapsed: {elapsed:.2f}s")

    assert stats["max_active"] <= MAX_CONCURRENCY


if __name__ == "__main__":
    asyncio.run(main())