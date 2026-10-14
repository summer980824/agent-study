from agent_study.client import GitHubClient, GitHubAPIError
from agent_study.github_user import GithubUser



client = GitHubClient(base_url="https://api.github.com")

try:
    r = client.http_get("/users/summer980824")
    data = r.json()
    user = GithubUser.from_content(r.json())
    print(user.model_dump())
except GitHubAPIError as e:
    print("请求失败:", e)


try:
    r = client.http_get("/users/summer92222")
    data = r.json()
    user = GithubUser.from_content(r.json())
    print(user.model_dump())
except GitHubAPIError as e:
    print("请求失败:", e)


