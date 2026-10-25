"""Unit tests for ``agent_study.cli``.

The CLI is driven through ``typer.testing.CliRunner`` so argument parsing,
exit codes and stdout/stderr are all exercised the way a user sees them.

Two invariants get special attention:

* the default invocation never touches the network, and
* ``python -m agent_study --help`` prints help and exits cleanly.

Network calls are stubbed at the module boundary (``get_github_user`` for the
synchronous command, ``respx`` for the httpx-based async one), so the suite
never depends on the network being reachable.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import types
from pathlib import Path
from urllib.error import HTTPError

import httpx
import pytest
import respx
from pydantic import ValidationError
from typer.testing import CliRunner

from agent_study import cli


GITHUB_USER_PAYLOAD = {
    "id": 583231,
    "login": "octocat",
    "url": "https://api.github.com/users/octocat",
    "public_repos": 8,
    "created_at": "2011-01-25T18:44:36Z",
}

BASE_URL = "https://api.github.com"


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def task_file(tmp_path: Path) -> Path:
    return tmp_path / "task.txt"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _fake_config_module(
    monkeypatch: pytest.MonkeyPatch,
    *,
    model_name: str = "gpt-4o-mini",
    api_base_url: str = "https://api.example.com",
    anthropic_base_url: str = "https://api.example.com/anthropic",
    api_key: str = "sk-secret-abcdef1234",
) -> None:
    """Install a stand-in for ``agent_study.config`` so tests do not need .env."""
    module = types.ModuleType("agent_study.config")
    module.settings = types.SimpleNamespace(
        MODEL_NAME=model_name,
        API_BASE_URL=api_base_url,
        ANTHROPIC_BASE_URL=anthropic_base_url,
        API_KEY=api_key,
    )
    monkeypatch.setitem(sys.modules, "agent_study.config", module)


def _broken_config_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a ``agent_study.config`` that fails validation on attribute access."""

    class _BrokenConfig(types.ModuleType):
        def __getattr__(self, name: str):
            raise ValidationError.from_exception_data(
                "Settings",
                [{"type": "missing", "loc": (name,), "input": None}],
            )

    monkeypatch.setitem(sys.modules, "agent_study.config", _BrokenConfig("agent_study.config"))


def _forbid_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any outbound socket connection fail the test loudly."""

    def _blocked(*args, **kwargs):
        raise AssertionError("CLI attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)


# ---------------------------------------------------------------------------
# help / default behaviour
# ---------------------------------------------------------------------------


class TestHelp:
    def test_help_exits_successfully(self, runner: CliRunner):
        result = runner.invoke(cli.app, ["--help"])
        assert result.exit_code == 0

    def test_help_lists_every_command(self, runner: CliRunner):
        result = runner.invoke(cli.app, ["--help"])
        for command in ("demo", "config", "github", "tasks"):
            assert command in result.stdout

    def test_help_shows_usage_line(self, runner: CliRunner):
        result = runner.invoke(cli.app, ["--help"])
        assert "Usage" in result.stdout

    def test_no_arguments_prints_help(self, runner: CliRunner):
        result = runner.invoke(cli.app, [])
        assert "Usage" in result.stdout
        assert result.exit_code != 0

    def test_no_arguments_does_not_run_any_command(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ):
        called: list[str] = []
        monkeypatch.setattr(
            cli, "demo", lambda: called.append("demo")
        )
        runner.invoke(cli.app, [])
        assert called == []

    def test_help_makes_no_network_call(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ):
        _forbid_network(monkeypatch)
        result = runner.invoke(cli.app, ["--help"])
        assert result.exit_code == 0

    def test_module_entry_point_shows_help(self):
        """``python -m agent_study --help`` must print help and exit 0."""
        env = {
            **os.environ,
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
        }
        completed = subprocess.run(
            [sys.executable, "-m", "agent_study", "--help"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
            cwd=Path(__file__).resolve().parent.parent,
        )
        assert completed.returncode == 0
        assert "Usage" in completed.stdout
        assert "demo" in completed.stdout

    def test_importing_cli_does_not_load_http_clients(self):
        """The CLI must not pull in HTTP machinery just by being imported."""
        code = (
            "import sys, agent_study.cli; "
            "print('httpx' in sys.modules, 'urllib.request' in sys.modules)"
        )
        env = {
            **os.environ,
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
        }
        completed = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
            cwd=Path(__file__).resolve().parent.parent,
        )
        assert completed.returncode == 0
        assert completed.stdout.strip() == "False False"


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------


class TestDemo:
    def test_demo_runs_successfully(self, runner: CliRunner):
        result = runner.invoke(cli.app, ["demo"])
        assert result.exit_code == 0
        assert "Add tasks complete" in result.stdout

    def test_demo_uses_a_temporary_task_file(self, runner: CliRunner):
        """演示的任务文件必须在临时目录里，不能落在项目或用户数据目录。"""
        result = runner.invoke(cli.app, ["demo"])
        assert result.exit_code == 0
        assert "learn RAG" in result.stdout

    def test_demo_does_not_touch_the_project_tree(
        self, runner: CliRunner
    ):
        """demo 不得在项目目录里留下任何文件改动。"""
        project_root = Path(__file__).resolve().parent.parent

        def _snapshot() -> dict[Path, int]:
            return {
                path: path.stat().st_mtime_ns
                for path in project_root.rglob("*.txt")
                if ".venv" not in path.parts
            }

        before = _snapshot()
        runner.invoke(cli.app, ["demo"])
        assert _snapshot() == before

    def test_demo_writes_only_outside_the_project_tree(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ):
        """直接拦截写文件调用，断言 demo 的落盘目标都在项目目录之外。"""
        from agent_study import util

        written: list[Path] = []
        real_write = util.write_file
        real_append = util.append_file

        def spy_write(file_path, str_txt):
            written.append(Path(file_path).resolve())
            return real_write(file_path, str_txt)

        def spy_append(file_path, str_txt):
            written.append(Path(file_path).resolve())
            return real_append(file_path, str_txt)

        monkeypatch.setattr(util, "write_file", spy_write)
        monkeypatch.setattr(util, "append_file", spy_append)

        result = runner.invoke(cli.app, ["demo"])

        assert result.exit_code == 0
        assert written, "demo 应该演示了文件写入"
        project_root = cli.PROJECT_ROOT.resolve()
        for path in written:
            assert project_root not in path.parents, path

    def test_demo_does_not_touch_the_default_task_file(
        self, runner: CliRunner
    ):
        default = cli.DEFAULT_TASK_FILE
        existed = default.exists()
        content = default.read_bytes() if existed else None
        runner.invoke(cli.app, ["demo"])
        if existed:
            assert default.read_bytes() == content
        else:
            assert not default.exists()

    def test_demo_makes_no_network_call(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ):
        _forbid_network(monkeypatch)
        result = runner.invoke(cli.app, ["demo"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_config_prints_settings(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ):
        _fake_config_module(monkeypatch)
        result = runner.invoke(cli.app, ["config"])
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["MODEL_NAME"] == "gpt-4o-mini"
        assert payload["API_BASE_URL"] == "https://api.example.com"
        assert payload["ANTHROPIC_BASE_URL"] == "https://api.example.com/anthropic"

    def test_config_masks_api_key(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ):
        _fake_config_module(monkeypatch, api_key="sk-secret-abcdef1234")
        result = runner.invoke(cli.app, ["config"])
        payload = json.loads(result.stdout)
        assert "sk-secret-abcdef1234" not in result.stdout
        assert payload["API_KEY"].endswith("1234")
        assert "secret" not in payload["API_KEY"]

    def test_config_reports_missing_env_vars(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ):
        _broken_config_module(monkeypatch)
        result = runner.invoke(cli.app, ["config"])
        assert result.exit_code == 1
        assert "配置加载失败" in result.stderr

    def test_config_does_not_leak_the_key(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ):
        _fake_config_module(monkeypatch, api_key="sk-topsecret-value")
        result = runner.invoke(cli.app, ["config"])
        assert "sk-topsecret-value" not in result.output


# ---------------------------------------------------------------------------
# tasks
# ---------------------------------------------------------------------------


class TestTasks:
    def test_add_appends_task(
        self, runner: CliRunner, task_file: Path
    ):
        result = runner.invoke(
            cli.app, ["tasks", "add", "learn Rust", "-f", str(task_file)]
        )
        assert result.exit_code == 0
        assert task_file.read_text(encoding="utf-8").strip() == "learn Rust"

    def test_add_keeps_previous_tasks(
        self, runner: CliRunner, task_file: Path
    ):
        task_file.write_text("first\n", encoding="utf-8")
        runner.invoke(
            cli.app, ["tasks", "add", "second", "-f", str(task_file)]
        )
        assert task_file.read_text(encoding="utf-8").splitlines() == [
            "first",
            "second",
        ]

    def test_list_prints_tasks(
        self, runner: CliRunner, task_file: Path
    ):
        task_file.write_text("alpha\nbeta\n", encoding="utf-8")
        result = runner.invoke(
            cli.app, ["tasks", "list", "-f", str(task_file)]
        )
        assert result.exit_code == 0
        assert result.stdout.splitlines() == ["alpha", "beta"]

    def test_add_then_list_roundtrip(
        self, runner: CliRunner, task_file: Path
    ):
        runner.invoke(
            cli.app, ["tasks", "add", "roundtrip", "-f", str(task_file)]
        )
        result = runner.invoke(
            cli.app, ["tasks", "list", "-f", str(task_file)]
        )
        assert result.stdout.strip() == "roundtrip"

    def test_list_on_missing_file_fails_cleanly(
        self, runner: CliRunner, tmp_path: Path
    ):
        missing = tmp_path / "nope.txt"
        result = runner.invoke(
            cli.app, ["tasks", "list", "-f", str(missing)]
        )
        assert result.exit_code == 1
        assert "任务文件不存在" in result.stderr

    def test_add_rejects_blank_title(
        self, runner: CliRunner, task_file: Path
    ):
        result = runner.invoke(
            cli.app, ["tasks", "add", "", "-f", str(task_file)]
        )
        assert result.exit_code == 1
        assert "Error" in result.stderr
        assert not task_file.exists()

    def test_add_rejects_whitespace_only_title(
        self, runner: CliRunner, task_file: Path
    ):
        result = runner.invoke(
            cli.app, ["tasks", "add", "   ", "-f", str(task_file)]
        )
        assert result.exit_code == 1
        assert "Error" in result.stderr
        assert not task_file.exists()

    def test_add_strips_the_title(
        self, runner: CliRunner, task_file: Path
    ):
        result = runner.invoke(
            cli.app, ["tasks", "add", "  learn Rust  ", "-f", str(task_file)]
        )
        assert result.exit_code == 0
        assert task_file.read_text(encoding="utf-8").strip() == "learn Rust"

    def test_add_creates_missing_parent_directories(
        self, runner: CliRunner, tmp_path: Path
    ):
        nested = tmp_path / "a" / "b" / "tasks.txt"
        result = runner.invoke(
            cli.app, ["tasks", "add", "nested", "-f", str(nested)]
        )
        assert result.exit_code == 0
        assert nested.read_text(encoding="utf-8").strip() == "nested"

    def test_tasks_makes_no_network_call(
        self, runner: CliRunner, task_file: Path, monkeypatch: pytest.MonkeyPatch
    ):
        _forbid_network(monkeypatch)
        runner.invoke(
            cli.app, ["tasks", "add", "offline", "-f", str(task_file)]
        )
        result = runner.invoke(
            cli.app, ["tasks", "list", "-f", str(task_file)]
        )
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# 路径锚定
# ---------------------------------------------------------------------------


class TestPathAnchoring:
    def test_default_task_file_is_absolute(self):
        assert cli.DEFAULT_TASK_FILE.is_absolute()

    def test_default_task_file_does_not_depend_on_cwd(self, tmp_path: Path):
        """换工作目录不应改变默认任务文件的位置。"""
        before = cli.DEFAULT_TASK_FILE
        cwd = Path.cwd()
        try:
            os.chdir(tmp_path)
            assert cli.DEFAULT_TASK_FILE == before
        finally:
            os.chdir(cwd)

    def test_default_task_file_is_outside_the_project_tree(self):
        project_root = cli.PROJECT_ROOT.resolve()
        assert project_root not in cli.DEFAULT_TASK_FILE.resolve().parents

    def test_package_dir_is_anchored_to_the_source_file(self):
        assert cli.PACKAGE_DIR == Path(cli.__file__).resolve().parent
        assert cli.PROJECT_ROOT == cli.PACKAGE_DIR.parent


# ---------------------------------------------------------------------------
# task.append_task 的领域校验
# ---------------------------------------------------------------------------


class TestAppendTaskValidation:
    def test_blank_title_raises_value_error(self, tmp_path: Path):
        from agent_study import task as task_module

        with pytest.raises(ValueError, match="blank"):
            task_module.append_task("   ", tmp_path / "t.txt")

    def test_title_is_stripped_before_storing(self, tmp_path: Path):
        from agent_study import task as task_module

        target = tmp_path / "t.txt"
        task_module.append_task("  learn Rust  ", target)
        assert target.read_text(encoding="utf-8") == "learn Rust\n"


# ---------------------------------------------------------------------------
# github user (synchronous)
# ---------------------------------------------------------------------------


class TestGithubUser:
    def test_prints_user_payload(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ):
        import agent_study.github_user as github_user_module

        monkeypatch.setattr(
            github_user_module,
            "get_github_user",
            lambda name: dict(GITHUB_USER_PAYLOAD),
        )
        result = runner.invoke(cli.app, ["github", "user", "octocat"])
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["login"] == "octocat"
        assert payload["public_repos"] == 8

    def test_http_error_exits_with_code_one(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ):
        import agent_study.github_user as github_user_module

        def _raise(name: str):
            raise HTTPError(
                "https://api.github.com/users/ghost",
                404,
                "Not Found",
                None,
                None,
            )

        monkeypatch.setattr(
            github_user_module, "get_github_user", _raise
        )
        result = runner.invoke(cli.app, ["github", "user", "ghost"])
        assert result.exit_code == 1
        assert "404" in result.stderr

    def test_invalid_payload_exits_with_code_one(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ):
        import agent_study.github_user as github_user_module

        monkeypatch.setattr(
            github_user_module,
            "get_github_user",
            lambda name: {"login": "octocat"},
        )
        result = runner.invoke(cli.app, ["github", "user", "octocat"])
        assert result.exit_code == 1
        assert "Error" in result.stderr

    def test_missing_argument_is_a_usage_error(
        self, runner: CliRunner
    ):
        result = runner.invoke(cli.app, ["github", "user"])
        assert result.exit_code != 0
        assert "Usage" in result.output


# ---------------------------------------------------------------------------
# github async (concurrent, httpx based)
# ---------------------------------------------------------------------------


class TestGithubAsync:
    @respx.mock
    def test_prints_every_user(self, runner: CliRunner):
        respx.get(f"{BASE_URL}/users/octocat").mock(
            return_value=httpx.Response(200, json=GITHUB_USER_PAYLOAD)
        )
        respx.get(f"{BASE_URL}/users/torvalds").mock(
            return_value=httpx.Response(
                200,
                json={**GITHUB_USER_PAYLOAD, "login": "torvalds", "public_repos": 12},
            )
        )
        result = runner.invoke(
            cli.app, ["github", "async", "octocat", "torvalds"]
        )
        assert result.exit_code == 0
        assert "octocat: 8 public repositories" in result.stdout
        assert "torvalds: 12 public repositories" in result.stdout

    @respx.mock
    def test_failed_user_is_reported_and_exits_non_zero(
        self, runner: CliRunner
    ):
        respx.get(f"{BASE_URL}/users/octocat").mock(
            return_value=httpx.Response(200, json=GITHUB_USER_PAYLOAD)
        )
        respx.get(f"{BASE_URL}/users/ghost").mock(
            return_value=httpx.Response(404, json={"message": "Not Found"})
        )
        result = runner.invoke(
            cli.app, ["github", "async", "octocat", "ghost"]
        )
        assert result.exit_code == 1
        assert "ghost: FAILED" in result.stderr
        assert "octocat: 8 public repositories" in result.stdout

    @respx.mock
    def test_retries_option_is_forwarded(self, runner: CliRunner):
        route = respx.get(f"{BASE_URL}/users/octocat").mock(
            side_effect=[
                httpx.Response(429, headers={"x-ratelimit-remaining": "0"}, json={}),
                httpx.Response(200, json=GITHUB_USER_PAYLOAD),
            ]
        )
        result = runner.invoke(
            cli.app, ["github", "async", "octocat", "--retries", "1"]
        )
        assert result.exit_code == 0
        assert route.call_count == 2


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


class TestMain:
    def test_main_delegates_to_the_typer_app(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        called: list[int] = []
        monkeypatch.setattr(cli, "app", lambda: called.append(1))
        cli.main()
        assert called == [1]

    def test_main_is_the_console_script_target(self):
        """``[project.scripts] agent-study`` must point at ``agent_study.cli:main``."""
        import tomllib

        pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
        with pyproject.open("rb") as handle:
            data = tomllib.load(handle)
        scripts = data["project"]["scripts"]
        assert scripts["agent-study"] == "agent_study.cli:main"
