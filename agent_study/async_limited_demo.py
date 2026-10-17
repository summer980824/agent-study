import asyncio
from time import perf_counter

from agent_study.async_client import AsyncGitHubClient


MAX_CONCURRENCY = 1

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
    async with semaphore:
        # 记录当前进入请求区域的任务数
        stats["active"] += 1
        stats["max_active"] = max(
            stats["max_active"],
            stats["active"],
        )

        try:
            data = await client.get_json(
                f"/users/{username}"
            )

            return {
                "login": data["login"],
                "public_repos": data["public_repos"],
            }

        finally:
            stats["active"] -= 1


async def main() -> None:
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