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

## 运行 LLM 客户端示例

`agent_study/llm_client_demo.py` 调用 MiniMax-M3.1-Flash（OpenAI 兼容接口），会**联网**。

```bash
uv sync                                   # 首次运行前安装依赖
uv run python -m agent_study.llm_client_demo   # 以模块方式运行
uv run agent_study/llm_client_demo.py          # 以脚本方式运行（等价）
```

在代码中直接调用：

```python
from agent_study.llm_client import chat

print(chat("用一句话解释 Agent", reasoning_effort="low"))
```

### 配置

配置来自项目根目录的 `.env` 与 `.env.dev`（`APP_ENV` 环境变量可切换，默认 `dev`）：

| 变量 | 说明 |
| --- | --- |
| `MODEL_NAME` | 模型名，如 `MiniMax-M3.1-Flash` |
| `API_KEY` | MiniMax 订阅 Key，缺失或无效会返回 `401 invalid api key` |
| `API_BASE_URL` | OpenAI 兼容接口地址，如 `https://api.minimax.cn/v1` |
| `ANTHROPIC_BASE_URL` | Anthropic 兼容接口地址，如 `https://api.minimax.cn/anthropic`，留空则回退到 `API_BASE_URL` |

### 说明

- `reasoning_effort` 只接受小写 `low` / `medium` / `high` / `xhigh` / `max`，省略时服务端按 `max` 处理。
- 思考过程在 `reasoning_content` 字段，`chat()` 只返回 `content` 中的最终答案。
- `temperature` / `top_p` 当前不生效：传什么都不报错，但服务端固定按 `temperature=1`、`top_p=0.95` 处理。

## 运行 Anthropic SDK 示例

`agent_study/anthropic_chat.py` 用 **Anthropic SDK** 调同一个 MiniMax 模型（Anthropic 兼容接口），会**联网**。
它和 `llm_client.py` 是同一模型的两种协议写法，可以对照着看。

```bash
uv sync                                            # 首次运行前安装依赖
uv run python -m agent_study.anthropic_chat_demo        # 以模块方式运行
uv run agent_study/anthropic_chat_demo.py               # 以脚本方式运行（等价）
```

在代码中直接调用：

```python
from agent_study.anthropic_chat import chat, chat_message

print(chat("用一句话解释 Agent", reasoning_effort="low"))

message = chat_message("用一句话解释 Agent")   # 需要 stop_reason / usage 时用这个
print(message.stop_reason, message.usage)
```

### 两套协议的差异

| 概念 | OpenAI 兼容（`llm_client.py`） | Anthropic Messages（`anthropic_chat.py`） |
| --- | --- | --- |
| 端点 | `POST /v1/chat/completions` | `POST /v1/messages` |
| system 提示词 | `messages[0].role=system` | 顶层 `system` 参数 |
| 输出上限 | `max_completion_tokens` | `max_tokens`（**必填**，本模块默认 1024） |
| 推理强度 | `reasoning_effort` | `output_config.effort` |
| 结束原因 | `finish_reason` | `stop_reason` |
| 思考内容 | `reasoning_content` 字段 | `content` 里的 `thinking` block |
| 正文位置 | `choices[0].message.content` | `content` 里的 `text` block |

### 说明

- `chat()` 只返回正文文本；`chat_message()` 返回完整的 `Message`，可以拿到 `stop_reason` 和 `usage`。
- `stop_reason` 要检查：`max_tokens` 表示输出被截断，但 HTTP 状态码仍是 200。
- MiniMax 的 Anthropic 端点对 `temperature` / `top_p` 暂不生效，`stop_sequences` / `top_k` 会被忽略。