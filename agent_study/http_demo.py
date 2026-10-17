from agent_study.client import GitHubClient, GitHubAPIError
from agent_study.github_user import GithubUser
import asyncio
from time import perf_counter

from agent_study.async_client import (
    AsyncGitHubAPIError,
    AsyncGitHubClient,
)

async def test_async() -> None:
    started = perf_counter()

    try:
        async with AsyncGitHubClient() as client:
            results = await asyncio.gather(
                client.get_json("/users/summer980824"),
                client.get_json("/users/octocat"),
                client.get_json("/users/torvalds"),
                return_exceptions=True
            )

        users = [
            GithubUser.from_content(data)
            for data in results
        ]

        for user in users:
            print(
                f"{user.login}: "
                f"{user.public_repos} public repositories"
            )

        print(f"Elapsed: {perf_counter() - started:.2f}s")
        print(f"async client is closed: {client.closed}")
    except AsyncGitHubAPIError as exc:
        print(f"GitHub API error: {exc}")




def main() -> None:
    # 这里调用同步和异步演示函数
    print("Start test async")
    asyncio.run(test_async())


if __name__ == "__main__":
    
    client = GitHubClient(base_url="https://api.github.com")

    try:
        r = client.http_get("/users/summer980824")
        user = GithubUser.from_content(r.json())
        print(user.model_dump())

        r = client.http_get("/users/summer92222")
        user = GithubUser.from_content(r.json())
        print(user.model_dump())
    except GitHubAPIError as e:
        print("请求失败:", e)
    finally:
        client.close()

    main()