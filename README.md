An agent study project

Senior Java Developer → AI Agent Engineer

## Quick Start

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync              # create .venv and install deps
uv run agent-study --help  # show available commands
```

无需手动激活虚拟环境——`uv run` 会自动使用项目自带的 `.venv`。

项目已注册为可执行命令 `agent-study`（入口 `agent_study.cli:main`）。
在项目目录内用 `uv run agent-study ...`，
或直接用 `.venv/Scripts/agent-study.exe`（Windows）。
`python -m agent_study --help` 同样可用。

## CLI

`python -m agent_study` 默认只显示帮助，不执行演示，也不发起任何网络请求。

```bash
uv run agent-study demo             # 本地演示（不联网）
uv run agent-study config           # 打印当前配置（API Key 脱敏）
uv run agent-study tasks list       # 列出任务文件内容
uv run agent-study tasks add "标题"  # 追加一条任务
uv run agent-study github user octocat          # 联网：单个用户
uv run agent-study github async octocat torvalds # 联网：并发多个用户
```