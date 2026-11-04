# Phase 1 Retrospective（STAR 面试版）

> 每一节对应一个面试问题，答案按 **S（Situation 情境） / T（Task 任务） / A（Action 行动） / R（Result 结果）** 展开。
> 涉及的文件均可在仓库中找到：`agent_study/cli.py`、`config.py`、`llm_client.py`、`logging_config.py`、`async_client.py`、`tests/`。

---

## 1. What I Built
### 面试题：请介绍一下你在这个项目里做了什么？
- **S**：我是一名有多年 Java 后端经验的工程师，正在转型 AI Agent 工程；Phase 1 的目标是从零搭出一条**最小但完整**的 LLM 调用链路，而不是做玩具 demo。
- **T**：我给自己定的交付标准是"能被当作真实工程代码使用"：有配置体系、有结构化日志、有错误分类、有离线可跑的测试。
- **A**：我实现了四块内容：`agent_study/llm_client.py` 提供 `chat` / `chat_response` / `chat_stream` / `chat_structured` 四个入口；`agent_study/cli.py` + `config.py` 串起 **CLI → Config → LLM Client → API** 的主链路；`agent_study/logging_config.py` 输出 JSON 日志并用 contextvars 给一次请求注入同一个 `request_id`；`tests/` 用 pytest + respx / httpx2 `MockTransport` 打桩，覆盖离线测试。
- **R**：项目可以 `uv run agent-study ai "用一句话解释 Agent"` 一条命令跑通真实调用，也能 `uv run pytest -q` 全量离线回归；尤其我把"输出被 `max_completion_tokens` 截断"和"内容被过滤"都定义成失败并以退出码 1 结束，杜绝了把半截答案当成完整结果交付的情况。

---

## 2. Architecture
### 面试题：讲讲你的架构设计和请求流程？
- **S**：一个 Agent 工程的骨架如果一开始就把配置、网络、日志搅在一起，后面接工具调用、接多个模型时会迅速失控，所以我先想清楚分层。
- **T**：我要实现一条职责单一、可测试、可替换的请求链路：**CLI → Config → LLM Client → API**，每个组件只干一件事。
- **A**：`cli` 只做参数解析、调度和"异常 → 退出码"的翻译；`config` 只负责把环境变量变成校验过的 `Settings`；`llm_client` 只负责构造请求体和校验响应结构；`logging_config` 只负责可观测性。配置由 `base_url` / `api_key` / `model` **显式传入** client，client 不自己猜配置来源；HTTP 层用进程内共享的 `httpx.Client` 复用 keep-alive 连接。
- **R**：四个模块之间没有反向依赖，任何一层都能单独替换或打桩；新增一个"多轮对话"CLI 只要复用同一条 Config → LLM Client 链路即可，不用改底层。

### 面试题：你的错误处理策略是什么？
- **S**：LLM 接口最坑的地方是"HTTP 200 但结果不能用"——被截断、被内容过滤、返回 HTML 故障页、JSON 结构畸形。
- **T**：我要求任何"不完整的成功"都必须被显式识别成失败，并且以可读的中文提示和稳定的退出码暴露给调用方。
- **A**：我做了三层分类：网络层用 `httpx.HTTPError`（超时 / 断连），HTTP 状态层用 `RuntimeError`（`API 401: ...`），内容层用 `LLMResponseError` 家族（`LLMResponseTruncatedError`、`LLMStructuredOutputError`）。CLI 逐类捕获、记录 `error_type` 并以退出码 1 结束；流式响应额外用 `done_received` 标记判断是否收到 SSE 的 `[DONE]`。
- **R**：现在无论用户还是上层 Agent 都能靠退出码和错误类型区分"网络抖动"和"模型输出有问题"；日志里每条都带同一个 `request_id`，排查时可以直接串起整条链路。

---

## 3. Engineering Decisions
### 面试题：为什么选 httpx？
- **S**：我要在同一个项目里同时写同步调用（CLI）和异步并发调用（GitHub 批量查询），还要保证测试能离线拦截请求。
- **T**：选一个库要同时满足：同步/异步都支持、默认带超时、能复用连接池、可被 mock。
- **A**：我选了 httpx——它同步异步一套 API、默认超时、`Client` 线程安全可跨线程共享，而且自带 `MockTransport` 并有成熟的 respx 生态，正好命中"生产能用 + 可离线测试"。
- **R**：生产代码和测试代码用的是同一套 httpx 对象，不需要为了可测性做额外抽象；同时用 `asyncio.gather` 并发拉多个 GitHub 用户时也不需要再引入 aiohttp 之类的第二个库。

### 面试题：为什么用 pydantic-settings 而不是 os.environ？
- **S**：早期用 `os.getenv` 时，配置错误（Key 写错、漏配、写成空串）只能在请求返回 401 之后才暴露，排查成本高。
- **T**：我要让配置错误在**启动时**就失败，并且明确区分"没配"和"配成空值"。
- **A**：我用 pydantic-settings 定义类型化的 `Settings`，分层加载 `.env` 与 `.env.{APP_ENV}`，用 `field_validator` 把空串回退成默认值（`MODEL_NAME=` → 默认模型）、把 `API_KEY` 空值直接拒绝，并给出 `MODEL_NAME` / `API_BASE_URL` / `ANTHROPIC_BASE_URL` 的默认值。
- **R**：现在 `uv run agent-study config` 就能一眼看出当前生效配置（Key 脱敏），缺 Key 时 CLI 直接提示"配置加载失败，缺少环境变量：API_KEY"并退出 1；还能顺手区分 OpenAI 兼容与 Anthropic 兼容两套 base_url，避免填错协议导致 404。

### 面试题：为什么要做结构化日志？
- **S**：一次 LLM 调用会横跨配置、请求、响应、解析多个环节，用人类可读的 print 串不起来，线上出问题基本靠猜。
- **T**：我需要日志能按一次请求聚合，并且能被机器消费做监控和成本分析。
- **A**：我实现了 `JsonFormatter`，把 `timestamp`、`level`、`logger`、`message`、`request_id` 作为固定字段，业务字段（`event`、`model`、`duration_ms`、`status_code`、`error_type`、`finish_reason`、`usage`）按白名单附加；`request_id` 通过 contextvars 的 `request_context` 上下文管理器注入，保证同一次请求的所有日志共享一个 ID。
- **R**：一次 `agent-study ai` 调用会输出 start / completed / failed 三条带同一 `request_id` 的 JSON；`duration_ms`、`finish_reason` 和 `usage` 直接可用于延迟监控与 token 成本核算，而"仅输出白名单字段"也避免了把 prompt 或 Key 意外写进日志。

### 面试题：为什么用 mock HTTP 而不是打真实接口？
- **S**：真实 LLM 接口既花钱又不稳定，错误路径（超时、429、畸形 JSON、截断）在真实环境里很难稳定复现。
- **T**：我要让测试离线、确定性、可重复，并且能覆盖错误分支。
- **A**：`tests/test_llm_client.py` 用 respx 拦截 httpx v1 的请求，`tests/test_anthropic_chat.py` 因为 Anthropic SDK 2.x 底层是 httpx2、respx 拦不到，改用 httpx2 自带的 `MockTransport` 打桩，其余异步测试同样用 respx。
- **R**：整个测试套件不联网也能跑完，正常响应、超时、4xx/5xx、限流、截断、内容过滤、结构化输出非法 JSON 等分支都有断言，CI 里可以放心反复执行。

---

## 4. What I Learned
### 面试题：从这个项目你学到了什么？
- **S**：Phase 1 是我第一次完整走通"从 Java 后端思维切换到 Python AI 工程"的流程，踩坑点比写代码本身更有价值。
- **T**：我要求每条经验都能对应到具体代码，而不是停留在"我学到了 Python 很强"这种空话。
- **A**：我把收获拆成五块并逐条落到代码上：① **Python 工程**——用 dataclass / pydantic 建模、用 uv + pyproject 管理依赖打包、用 `__main__.py` 与 entry point 同时提供 `python -m` 和可执行命令；② **HTTP / REST**——搞懂了 OpenAI 兼容与 Anthropic Messages 两套协议在 system 位置、`max_completion_tokens` vs `max_tokens`、`finish_reason` vs `stop_reason`、思考内容字段上的差异，以及"HTTP 200 不等于结果可用"；③ **异步与并发**——用 `asyncio.gather(..., return_exceptions=True)` 并发、`asyncio.run` 承接事件循环、共享异步客户端复用连接，并区分 primary / secondary rate limit 按 `retry_after` 与指数退避重试；④ **LLM API 集成**——`reasoning_effort` 与思考内容分离、SSE 流式解析与 `[DONE]` 收尾标记、JSON Schema 结构化输出、把 `finish_reason` 和 `usage` 当一等返回值；⑤ **测试与可观测性**——pytest fixture + respx、区分 httpx v1 与 httpx2 两种打桩、断言异常类型而不只断言返回值、用结构化日志串起耗时与 token 用量。
- **R**：这五块直接支撑了 Phase 1 的稳定性：并发与退避逻辑让批量调用不被单个 429 拖垮，流式与 `[DONE]` 判断让"流没正常收尾"能被识别，而结构化日志让我能在不打印敏感内容的前提下复盘问题。

---

## 5. Gaps and Improvements
### 面试题：目前哪些地方还不够生产可用？你打算怎么改进？
- **S**：诚实地评估，这套代码目前是一个**单机的、面向学习用途**的骨架：只有 CLI 入口，密钥靠本地 `.env`，LLM 调用本身没有任何重试与熔断。
- **T**：我要说清"哪里不达标"和"下一步的优先级"，而不是泛泛地说要继续优化。
- **A**：当前缺口包括：没有多进程/服务化部署形态，配置与密钥没有集中管理和轮换；`llm_client` 缺少带抖动的重试、退避和幂等语义，超时与连接池参数没有分级；没有基于 `usage` 的成本预算与配额闸门；请求与响应没有落盘、链路追踪或 prompt 版本管理；共享 `httpx.Client` 也没按并发量调过连接池。改进顺序上，我会先给 `llm_client` 补带抖动的重试与统一超时/连接池配置，然后把 `usage` 落成可查询的成本账本并加上预算上限，再把这条链路接进真正的 agent 循环（工具调用 + 多轮状态管理），最后补 CI、覆盖率门槛和密钥托管。
- **R**：这份清单能让我知道 Phase 1 的产出复用价值在哪——配置、日志、错误分类和 mock 测试这四层可以直接进入 Phase 2，需要重做的只有调用可靠性和 agent 编排部分。

---

## 6. Phase 2 Preparation
### 面试题：Phase 2 你准备做什么？
- **S**：Phase 1 让我看清了"调用模型"这一层，但离"理解并训练模型"还差一个数学底座；要进入模型内部，就必须先补上深度学习的计算基础。
- **T**：我为 Phase 2 定的目标是把 Tensor → 矩阵运算 → 广播 → 自动微分这条主线打通，并能动手实现最小可训练网络。
- **A**：具体分四块推进：**Tensor** 作为带 shape / dtype / device 元数据的多维数据容器（对应 Phase 1 里 `ChatResult` 把 `content`、`finish_reason`、`usage` 明确建模的做法）；**矩阵运算**（matmul / 转置 / reshape）作为注意力的计算原语，把 Phase 1 里"输入 × 权重 → 输出"的黑盒变成可手算的公式；**Broadcasting** 讲清不同形状中间结果的元素对齐规则，对应 Phase 1 中把不同形态的 prompt 统一渲染成文本再交给同一接口处理；**Autograd** 通过计算图和梯度累积解释参数如何被更新、进而支撑微调。
- **R**：完成这四块后，我就能把 Phase 1 的 LLM 调用链路从"调用别人的模型"推进到"理解模型内部并具备最小训练能力"，同时 Phase 1 建立的 mock 测试与结构化日志习惯会继续沿用到 Phase 2 的实现里。
