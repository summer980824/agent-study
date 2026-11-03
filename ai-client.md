# AI Client（Day 26 CLI）项目简介

## 这个 CLI 解决什么问题

调用 LLM API 看起来只是"发一个 POST 请求"，但真要在命令行里可靠地用，至少要处理这些事：

- **配置管理**：API Key、Base URL、模型名不能硬编码在代码里，要能按环境（dev / prod）切换。
- **错误处理**：401（Key 无效）、429（限流）、超时、断连，每种错误的含义和对用户的提示都不同。
- **响应校验**：HTTP 200 不代表拿到完整答案——`finish_reason=length`（输出被截断）、`content_filter`（内容被拦截）都可能藏在成功响应里。
- **可观测性**：出问题时需要 `request_id` 把一次请求的所有日志串起来。
- **可测试**：网络请求必须能被 mock 掉，测试不联网。

`agent-study ai` 命令（`agent_study/cli.py`）把 **Config + Client + Logging** 串成一条最小但完整的 LLM 调用链路，同时把上面这些问题都处理了。它是新人理解本项目架构的最佳入口。

## 架构说明

调用链：`CLI → Config → LLM Client → API`

```
用户
  │  uv run agent-study ai "用三句话解释什么是 RAG" [--model ...] [--timeout ...]
  ▼
agent_study.cli · ai()  (typer 命令)
 职责：参数解析、调度、错误 → 退出码翻译、stdout/stderr 分离
  │
  ├─▶ config.py · Settings      从 .env / .env.dev 加载模型/地址/Key
  ├─▶ logging_config            装 JSON 日志 handler，注入 request_id
  └─▶ llm_client.py · chat_response()
        ├─ _build_payload()     可选参数为 None 时不进请求体
        ├─ _post()  → 共享 httpx.Client（进程内连接池，只建一次）
        └─ _parse_response()    校验 ChatResult，识别截断/拦截
                            │  POST {API_BASE_URL}/chat/completions
                            ▼
                  OpenAI 兼容 API（默认 MiniMax）
```

设计要点：

- **CLI 只调度**：`ai()` 不做任何 HTTP / 配置细节，只负责"读配置 → 调 client → 处理结果 → 打印 + 记日志"。依赖在函数体内懒加载，`import agent_study.cli` 不引入 `httpx`。
- **配置单向流动**：`base_url` / `api_key` / `model` 由 CLI 从 `settings` 取出后显式传参给 client，测试时可以整体替换 config 模块。
- **日志是旁路**：logging 不参与返回值，只通过 `request_context` 注入 `request_id`，用 `extra` 携带结构化字段。

## 技术栈

| 技术 | 用途 |
| --- | --- |
| Python ≥ 3.11 | 语言 |
| [Typer](https://typer.tiangolo.com/) | CLI 框架（参数解析、`--help`、退出码） |
| [httpx](https://www.python-httpx.org/) | 同步 HTTP 客户端（连接池复用） |
| [pydantic / pydantic-settings](https://docs.pydantic.dev/) | 配置加载与响应结构校验 |
| pytest / pytest-asyncio | 测试（respx mock HTTP，不联网） |
| [uv](https://docs.astral.sh/uv/) | 依赖管理与运行（`uv sync` / `uv run`） |

## 环境配置

配置来自项目根目录的 `.env` 与 `.env.dev`（用 `APP_ENV` 环境变量切换，默认 `dev`），最小可用配置如下：

```bash
# .env  （复制 .env.example 后填写）
API_KEY=sk-...                # 必填，缺失或无效会返回 401 invalid api key
API_BASE_URL=https://api.minimax.io/v1    # OpenAI 兼容接口地址，可省略用默认
MODEL_NAME=MiniMax-M3.1-Flash             # 模型名，可省略用默认
```

写成空值（`KEY=`）等同于不配置，走默认值。可用 `uv run agent-study config` 查看当前生效的配置（API Key 自动脱敏），加载失败时会列出缺少的变量并以退出码 1 结束。

## 安装步骤

```bash
git clone <repo-url>          # 1. 克隆项目
cd agent-study                # 2. 进入目录
uv sync                       # 3. 创建 .venv 并安装依赖
cp .env.example .env          # 4. 准备配置文件
#   编辑 .env，填入 API_KEY
uv run agent-study config     # 5. 验证配置已生效（Key 脱敏）
uv run agent-study ai "用一句话解释 Agent"   # 6. 运行！
```

无需手动激活虚拟环境——`uv run` 会自动使用项目自带的 `.venv`。

## 使用示例

正常调用：

```bash
$ uv run agent-study ai "用三句话解释什么是 RAG"
RAG（检索增强生成）是一种把外部知识检索和 LLM 生成结合的技术……
```

覆盖模型与超时：

```bash
uv run agent-study ai "用三句话解释什么是 RAG" -m MiniMax-M3.1 --timeout 60
```

错误处理（错误信息走 stderr，退出码均为 1）：

```bash
$ uv run agent-study ai "hello"          # API Key 缺失/无效
配置加载失败，缺少环境变量：API_KEY
# 或服务端返回：401 invalid api key

$ uv run agent-study ai "hello"          # 网络层错误
网络错误（ConnectError）：...

$ uv run agent-study ai "hello"          # HTTP 200 但没拿到完整答案
输出被 max_completion_tokens 截断，只有半截正文
内容被内容过滤器拦截
```

CLI 不会把半截答案当成完整答案返回：结构校验失败抛 `LLMResponseError`，被截断（`finish_reason == "length"`）抛 `LLMResponseTruncatedError`。每次调用（无论成败）都会按 JSON 格式输出日志，且所有日志共享同一个 `request_id`。

## 测试说明

```bash
uv run pytest -q          # 运行全部测试（不联网）
```

- CLI 行为测试：`tests/test_cli.py`（用 typer 的 `CliRunner`，respx 拦截 HTTP）。
- LLM client 单元测试：`tests/test_llm_client.py`（mock `httpx.Client`，覆盖截断、结构错误、限流等分支）。

## 项目目录结构

```
agent-study/
├── agent_study/            # 包源码
│   ├── cli.py              # CLI 入口（ai / config / demo / tasks / github）
│   ├── config.py           # Settings：.env 加载、dev/prod 切换
│   ├── llm_client.py       # OpenAI 兼容协议 client（chat / chat_response / chat_stream）
│   ├── anthropic_chat.py   # Anthropic 协议 client（同一模型的另一种协议写法）
│   ├── logging_config.py   # JSON 日志 + request_context
│   ├── task.py             # 任务文件读写（tasks 命令用）
│   ├── github_user.py      # GitHub 用户模型（github 命令用）
│   ├── async_client.py     # 异步并发客户端（github async 用）
│   └── ...                 # prompt_template / token_count / text_cleaning 等学习模块
├── tests/                  # pytest 测试（mock HTTP，不联网）
├── note/                   # 学习笔记（Markdown）
├── docs/                   # 项目文档
├── .env / .env.dev / .env.prod   # 环境配置
├── pyproject.toml          # 项目元数据与依赖（uv 管理）
└── README.md               # 主文档（命令速查、协议差异、配置说明）
```
