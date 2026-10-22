"""agent_study 的命令行入口。

本模块只负责参数解析与调度，真正的业务逻辑仍留在各个模块中。
默认行为是显示帮助信息，不会执行演示流程，也不会发起任何网络请求；
需要联网的命令必须显式调用。
"""
from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from urllib.error import HTTPError

import typer
from pydantic import ValidationError

from agent_study import task as task_module
from agent_study.user import User

app = typer.Typer(
    name="agent-study",
    help="agent_study 工具集：本地演示、配置查看、任务文件与 GitHub 用户查询。",
    no_args_is_help=True,
    add_completion=False,
)

github_app = typer.Typer(
    help="访问 GitHub API（会联网）。",
    no_args_is_help=True,
)
tasks_app = typer.Typer(
    help="管理本地任务文件。",
    no_args_is_help=True,
)

app.add_typer(github_app, name="github")
app.add_typer(tasks_app, name="tasks")

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent

# 任务文件默认放在用户数据目录：与 CWD 无关，也不会写进 git 工作区。
USER_DATA_DIR = Path.home() / ".agent-study"
DEFAULT_TASK_FILE = USER_DATA_DIR / "tasks.txt"


def _mask(secret: str) -> str:
    """脱敏展示密钥，只保留末尾 4 位。"""
    if not secret:
        return ""
    if len(secret) <= 4:
        return "*" * len(secret)
    return f"{'*' * (len(secret) - 4)}{secret[-4:]}"


@app.command()
def demo() -> None:
    """运行本地演示流程（不联网）。"""
    user_1 = User.from_dict(
        {
            "id": 1001,
            "name": "Alice",
            "preferences": {"language": "zh", "model": "gpt"},
            "tasks": [
                {"id": 1, "title": "learn Python", "status": "done"},
                {"id": 2, "title": "learn RAG"},
            ],
        }
    )

    user_1.name = "Summer"
    user_1.preferences.model = "Kimi"

    try:
        user_1.add_task("learn Agent")
        user_1.add_task("learn Java", status="done")
        user_1.add_task("")
    except ValueError as e:
        print(f"Error: {e}")
    finally:
        print("Add tasks complete")

    user_1.print_user_summary(last_login="10/6", modified=True)
    user_1.print_tasks_not_done()

    user_2 = User(id=1002, name="Tom")
    user_2.add_task("learn sth")

    # 演示用的任务文件放在临时目录，既不污染 git 工作区，
    # 也不会覆盖用户自己的任务清单。
    with tempfile.TemporaryDirectory(prefix="agent-study-demo-") as tmp_dir:
        file_path = Path(tmp_dir) / "task.txt"
        task_module.save_tasks(user_1.pending_tasks(), file_path)
        print(task_module.load_tasks(file_path))
        task_module.append_task("adhoc task", file_path)
        print(task_module.load_tasks(file_path))

    user_3 = User.load('{"id": 1003, "name": "Jack"}')
    try:
        user_3.add_task(" ")
    except ValidationError as ve:
        print(f"Error: {ve}")

    try:
        user_3.add_task("1", "not sure")
    except ValidationError as ve:
        print(f"Error: {ve}")


@app.command()
def config() -> None:
    """打印当前生效的配置（API Key 脱敏）。"""
    try:
        from agent_study.config import settings
    except ValidationError as exc:
        missing = sorted(
            ".".join(str(part) for part in err["loc"])
            for err in exc.errors()
        )
        typer.secho(
            "配置加载失败，缺少环境变量：" + ", ".join(missing),
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    typer.echo(
        json.dumps(
            {
                "MODEL_NAME": settings.MODEL_NAME,
                "API_BASE_URL": settings.API_BASE_URL,
                "API_KEY": _mask(settings.API_KEY),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


@github_app.command("user")
def github_user(
    name: str = typer.Argument(..., help="GitHub 用户名"),
) -> None:
    """同步获取单个 GitHub 用户信息。"""
    from agent_study.github_user import GithubUser

    try:
        user = GithubUser.from_username(name)
    except HTTPError as e:
        typer.secho(f"{e.code}: {e.reason}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    except ValidationError as ve:
        typer.secho(f"Error: {ve}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    typer.echo(json.dumps(user.model_dump(mode="json"), indent=2, ensure_ascii=False))


@github_app.command("async")
def github_async(
    names: list[str] = typer.Argument(..., help="一个或多个 GitHub 用户名"),
    retries: int = typer.Option(0, "--retries", min=0, help="限流时的重试次数"),
) -> None:
    """并发获取多个 GitHub 用户信息。"""
    from agent_study.async_client import AsyncGitHubAPIError, AsyncGitHubClient
    from agent_study.github_user import GithubUser

    async def run() -> int:
        async with AsyncGitHubClient(max_retries=retries) as client:
            results = await asyncio.gather(
                *(client.get_json(f"/users/{name}") for name in names),
                return_exceptions=True,
            )

        failed = False
        for name, result in zip(names, results):
            if isinstance(result, BaseException):
                failed = True
                typer.secho(f"{name}: FAILED - {result}", fg=typer.colors.RED, err=True)
                continue

            user = GithubUser.from_content(result)
            typer.echo(f"{user.login}: {user.public_repos} public repositories")

        return 1 if failed else 0

    try:
        raise typer.Exit(code=asyncio.run(run()))
    except AsyncGitHubAPIError as exc:
        typer.secho(f"GitHub API error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


@tasks_app.command("list")
def tasks_list(
    task_file: Path = typer.Option(
        DEFAULT_TASK_FILE, "--file", "-f", help="任务文件路径"
    ),
) -> None:
    """列出任务文件中的所有任务。"""
    if not task_file.exists():
        typer.secho(
            f"任务文件不存在：{task_file}", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(code=1)

    for title in task_module.load_tasks(task_file):
        typer.echo(title)


@tasks_app.command("add")
def tasks_add(
    title: str = typer.Argument(..., help="任务标题"),
    task_file: Path = typer.Option(
        DEFAULT_TASK_FILE, "--file", "-f", help="任务文件路径"
    ),
) -> None:
    """向任务文件追加一条任务。"""
    try:
        task_module.append_task(title, task_file)
    except ValueError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    typer.echo(f"已添加: {title.strip()}")


def main() -> None:
    """CLI 入口。"""
    app()


if __name__ == "__main__":
    main()
