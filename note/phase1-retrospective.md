# Phase 1 Retrospective

## 1. What I Built
- Minimal AI Client：`agent_study/llm_client.py` 提供 `chat` / `chat_response` / `chat_stream` / `chat_structured` 四个入口，走 OpenAI 兼容接口并对响应做结构校验，截断的半截答案一律当失败抛出而不是当成完整结果返回。
- CLI + Config + LLM Client：`agent_study/cli.py` 用 Typer 提供 `agent-study ai` 命令负责参数解析与调度，`agent_study/config.py` 用 pydantic-settings 从 `.env` / `.env.dev` 读模型、地址和 Key，三者串成 CLI → Config → LLM Client → API 一条链路。
- Structured Logging：`agent_study/logging_config.py` 用 `JsonFormatter` 输出 JSON 行日志，并通过 contextvars 的 `request_context` 给一次请求的所有日志注入同一个 `request_id`。
- Mock-based Tests：`tests/` 下用 pytest + respx（httpx v1）和 httpx2 的 `MockTransport`（Anthropic SDK）打桩，整个测试套件不联网也能覆盖超时、4xx/5xx、截断等错误分支。

## 2. Architecture
- Request flow：CLI → Config → LLM Client → API，用户在命令行给出 prompt，CLI 读配置并显式把 `base_url` / `api_key` / `model` 传给 client，client 通过共享的 `httpx.Client` 向 OpenAI 兼容端点发一次 `/chat/completions` 请求。
- Component responsibilities：`cli` 只做参数解析、调度和"异常 → 退出码"的翻译，`config` 只负责把环境变量变成校验过的 `Settings`，`llm_client` 只负责构造请求体和校验响应结构，`logging_config` 只负责可观测性，四者之间没有反向依赖。
- Error handling strategy：分层异常——网络层用 `httpx.HTTPError`、HTTP 状态层用 `RuntimeError`、内容层用 `LLMResponseError` 家族（含 `LLMResponseTruncatedError`、`LLMStructuredOutputError`），CLI 逐类捕获后打印中文提示并统一以退出码 1 结束，`content_filter` 与截断同样按失败处理。

## 3. Engineering Decisions
- Why httpx？它同时覆盖同步和异步、默认带超时、线程安全的客户端能复用 keep-alive 连接池，而且自带 `MockTransport` 并有成熟的 respx 生态，正好满足"生产能用 + 可离线测试"两个要求。
- Why pydantic-settings？它用类型化的 `Settings` 把 `.env` 与 `.env.dev` 的分层加载、默认值、空串回退默认（`MODEL_NAME=`）、以及 `API_KEY` 必填等规则收敛成启动时的快速失败，而不是等到请求时才暴露配置错误。
- Why structured logging？因为一次 LLM 调用会横跨配置、请求、响应多个环节，JSON 行 + 统一 `request_id` 让日志能被机器解析和聚合，而 `duration_ms` / `finish_reason` / `usage` 这些字段可以直接用于延迟监控和成本核算。
- Why mock HTTP requests？因为 LLM 接口既贵又不稳定，用 respx / MockTransport 拦截可以让测试离线、确定性地跑完，还能精确构造出超时、限流、畸形 JSON、截断这些线上偶发的错误路径。

## 4. What I Learned
- Python engineering：把数据模型交给 dataclass 和 pydantic、把配置交给 pydantic-settings、用 uv + pyproject 管理依赖和打包、以及用 `__main__.py` 和 entry point 同时提供 `python -m` 与可执行命令。
- HTTP / REST：真正搞懂了 OpenAI 兼容协议与 Anthropic Messages 协议在 system 提示词位置、`max_completion_tokens` vs `max_tokens`、`finish_reason` vs `stop_reason`、思考内容字段上的差异，以及"HTTP 200 不等于结果可用"。
- Async and concurrency：用 `asyncio.gather(..., return_exceptions=True)` 并发拉取多个用户、用 `asyncio.run` 承接事件循环、共享异步客户端复用连接，并学会区分 primary / secondary rate limit 并按 `retry_after` 与指数退避重试。
- LLM API integration：`reasoning_effort` 与思考内容的分离、SSE 流式解析与 `[DONE]` 收尾标记、通过 JSON Schema 做结构化输出，以及把 `finish_reason` 和 `usage` 当作一等返回值来对待。
- Testing and observability：用 pytest fixture + respx 拦截请求、区分 httpx v1 与 httpx2 两种打桩方式、断言异常类型而不是只断言返回值，并用结构化日志把一次请求的耗时、结束原因和 token 用量串起来。

## 5. Gaps and Improvements
- What is not production-ready？目前只有单机 CLI、配置和密钥全部依赖本地 `.env`、LLM 请求没有重试与熔断也没有请求超时分级、没有 token 成本预算与配额控制、请求与响应没有落盘或链路追踪、共享 `httpx.Client` 也没有按并发量调过连接池参数。
- What should be improved next？优先给 `llm_client` 补上带抖动的重试/退避与幂等语义、统一超时与连接池配置、把 `usage` 落成可查询的成本账本，再把这条链路接进真正的 agent 循环（工具调用 + 多轮状态）并补上 CI 与覆盖率门槛。

## 6. Phase 2 Preparation
- Tensor：Phase 2 的底层对象是带 shape / dtype / device 元数据的多维张量，对应 Phase 1 里 `ChatResult` 把 `content`、`finish_reason`、`usage` 明确建模的做法。
- Matrix Operations：矩阵乘法、转置、reshape 是注意力机制的计算原语，正好对应 Phase 1 里"一次 chat 调用 = 输入 × 权重 → 输出"的黑盒视角。
- Broadcasting：张量广播让不同形状的中间结果能按元素对齐，思路接近 Phase 1 中把 prompt 统一渲染成文本再交给同一个接口处理的不同输入形态。
- Autograd：反向模式自动微分要依赖计算图和梯度累积，是 Phase 2 里理解模型参数如何被更新、如何做微调的核心前置知识。
