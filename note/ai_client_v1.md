# 最小 AI Client：ai 命令的组件依赖与工程能力清单

对应代码：`agent_study/cli.py` 的 `ai` 命令，把 **Config + Client + Logging** 串成一条最小但完整的 LLM 调用链路。

## 1. ai 命令的组件依赖图

### 1.1 静态结构

```text
用户
  │  uv run agent-study ai "用三句话解释什么是 RAG" [--model ...] [--timeout ...]
  ▼
┌─────────────────────────────────────────────────────────────┐
│ agent_study.cli  ·  ai()  (typer 命令)                       │
│  职责：参数解析、调度、错误 → 退出码翻译、stdout/stderr 分离    │
└──────┬──────────────┬──────────────────────┬────────────────┘
       │              │                      │
       ▼              ▼                      ▼
┌────────────┐ ┌──────────────┐ ┌──────────────────────────────┐
│ config.py  │ │logging_config│ │ llm_client.py                │
│ Settings   │ │              │ │  chat_response()             │
│ (.env 加载) │ │configure_    │ │   ├─ _build_payload()        │
│ 模型/地址/Key│ │ logging()    │ │   ├─ _post()  → httpx.Client │
│            │ │request_      │ │   └─ _parse_response()       │
│            │ │ context()    │ │      (ChatResult 校验)        │
└────────────┘ └──────────────┘ └───────────┬──────────────────┘
                                            ▼
                              共享 httpx.Client（进程内连接池，只建一次）
                                            │  POST {API_BASE_URL}/chat/completions
                                            ▼
                              MiniMax OpenAI 兼容接口（真实网络）
```

要点：

- **CLI 只调度**：`ai()` 不做任何 HTTP / 配置细节，只负责"读配置 → 调 client → 处理结果 → 打印 + 记日志"。三个依赖都在函数体内**懒加载**（`import agent_study.cli` 不会引入 `httpx`），保持默认无副作用。
- **配置单向流动**：`base_url` / `api_key` / `model` 由 CLI 从 `settings` 取出后**显式传参**给 client。client 内部虽然也引用了 settings 兜底，但 CLI 层不依赖这个兜底——这让测试可以整体替换 config 模块。
- **日志是旁路**：logging_config 不参与返回值，只通过 `request_context` 注入 `request_id`，用 `extra` 携带结构化字段。

### 1.2 一次调用的运行时数据流（含测试替身）

```text
用户输入 prompt
   │
   ▼
ai() ──导入──▶ config.settings  (失败: ValidationError → stderr "配置加载失败" → exit 1)
   │
   ├─ configure_logging()  （给 agent_study logger 装 JSON handler，绑定当前 stderr）
   ├─ request_id = uuid.uuid4().hex[:12]
   │
   ├─ [request_context(request_id)]
   │      │
   │      ├─ logger.info(event=ai_request, model=...)          ──▶ stderr (JSON)
   │      │
   │      ├─ llm_client.chat_response(prompt, base_url, api_key, model, timeout)
   │      │        │
   │      │        ├─ _build_payload: 可选参数为 None 时不进请求体
   │      │        ├─ _post ──▶ 共享 httpx.Client ──POST──▶ MiniMax API
   │      │        │                                    （测试时: respx 拦截）
   │      │        │        HTTP >= 400 → RuntimeError
   │      │        └─ _parse_response: choices/message 结构校验 → ChatResult
   │      │
   │      ├─ 失败分支: LLMResponseError / RuntimeError → logger.error → exit 1
   │      ├─ 业务分支: truncated / content_filter → logger.error → exit 1
   │      └─ logger.info(event=ai_response, duration_ms, usage) ──▶ stderr (JSON)
   │
   └─ typer.echo(response.content)                              ──▶ stdout（正文）
```

**stdout 与 stderr 的分离是刻意的**：正文走 stdout 可以被管道消费（`agent-study ai ... > answer.md`），日志走 stderr 不污染管道。测试里这也是两个独立的断言面。

### 1.3 测试侧的镜像结构

```text
tests/test_cli.py · TestAi
   │
   ├─ _fake_config_module()   sys.modules["agent_study.config"] 换成 SimpleNamespace
   │                          → ai() 里的"导入 config"拿到假配置，不读 .env
   ├─ respx.mock              拦截 httpx 的 POST → 返回构造的 chat completion JSON
   │                          → 全程零网络，零真实 API Key
   └─ CliRunner.invoke(cli.app, ["ai", ...])
                              走和真实命令完全相同的参数解析 / 退出码 / 输出路径
```

因为 CLI 把 config 和 client 都**当作外部依赖**对待（传参而非共享全局状态），测试才能在模块边界上整体替换，而不必钻进 client 内部 mock 函数。

## 2. 面试题：一个合格的 AI Client 至少应该具备哪些工程能力？

> 背景：AI Client 和普通 HTTP client 的差别在于——**每次调用都花钱、返回是非结构化文本、行为有随机性、协议还在快速演进**。合格的工程能力围绕这四个差异展开。

### 2.1 配置管理：密钥与环境分离

- 配置走 `pydantic-settings` 一类的机制：类型校验、默认值、`.env` / 环境变量分层覆盖，启动即失败（fail-fast）而不是调用时才报错。
- **密钥纪律**：API Key 只进请求头，不进日志、不进异常消息、展示时必须脱敏（本项目的 `config` 命令只显示末 4 位，且有测试钉住"任何输出都不含完整 Key"）。
- 空值语义要明确：`.env` 里 `KEY=` 是"用默认"还是"报错"，必须显式设计，不能看运气。

### 2.2 连接与资源管理

- **共享 client / 连接池复用**：每请求新建连接 = 每次重新 TLS 握手，延迟和开销都浪费（`llm_client._get_client()` 进程内只建一次；`httpx.Client` 线程安全可直接共享）。
- **超时是第一公民**：必须显式设默认超时并提供覆盖入口。LLM 响应慢是常态，"无限等"会把调用方拖死。连接超时、读超时最好分开。
- 资源释放要有出口（`close_client()`），流式场景下中途退出必须能关闭连接。

### 2.3 协议适配：兼容层要"收敛"而不是"透传"

- 对 OpenAI 兼容协议做一层收敛：**模型不支持的参数宁可不发**（`temperature` / `max_completion_tokens` 为 `None` 时从 payload 里消失），也不要把客户端默认值硬塞给服务端。
- 面向接口编程：换 provider 时变化的是 base_url / model / 少量字段语义，而不是调用方代码。

### 2.4 响应校验：不要相信"HTTP 200 = 能用的答案"

这是 AI Client 与普通 client 最大的不同。200 只是"服务器回了消息"，消息有没有用要客户端自己判：

- **结构校验**：`choices[0].message.content` 缺一层就不能当成功，抛带上下文的结构错误（`LLMResponseError`）。
- **`finish_reason` 是一等返回值**：必须随正文一起返回。`stop` 才是正常；`length` 是半截答案（本项目选择抛 `LLMResponseTruncatedError`，而不是把半截当完整——这是产品决策，但必须显式做决策）；`content_filter` 要单独处理。
- **流式的完整性是另一个问题**：SSE 的"正文吐完" ≠ "流正常结束"。只有收到 `[DONE]` 才算收尾（`done_received`），部分兼容端不发 `[DONE]`，客户端要给出状态而不是假装成功。

### 2.5 错误处理分层：不同层级的失败用不同策略

| 层级 | 例子 | 合格的处理 |
| --- | --- | --- |
| 传输层 | 超时、断连 | 可重试（幂等场景），退避 + 上限 |
| HTTP 层 | 401 / 429 / 5xx | 4xx 不重试（除了 429 限流），带状态码和响应片段报错 |
| 协议层 | 200 但结构不合预期 | 结构错误，附原始片段便于排查 |
| 内容层 | 截断、内容过滤、非法 JSON | 与正常返回**区分**——这是业务信号，不是异常噪音 |

关键原则：**截断不是"答案短一点"，内容过滤不是"模型脾气差"**，它们和 HTTP 错误完全不同级，混在一起会让上层做出完全错误的决策。

### 2.6 可观测性：每次调用都要能回答"谁、何时、花了多少"

- **结构化日志**（JSON）而非自由文本，字段固定，可被日志系统直接索引。
- **request_id 贯穿**：一次调用产生的所有日志共享同一个 id（`contextvars` 实现，异步/线程安全），分布式排障时能把日志按调用串起来。
- **usage 是计费唯一权威**：token 用量必须从响应里取并记录，不要本地估算。成本报表、限流、告警都建在这上面。
- 日志走 **stderr**，正文走 **stdout**，两条通道互不污染。

### 2.7 成本控制意识

- 默认参数朝"省"的方向设计：不传不必要的采样参数、不为探索性调用默认开大 `max_tokens`。
- 截断检测（2.4）本质上也是成本控制：半截输出 = 白花的 prompt token。
- 重试要有次数上限和退避；429 才重试，4xx 业务错误重试只是烧钱。

### 2.8 可测试性：依赖可替换，网络可关闭

- client 的核心参数（`base_url` / `api_key` / `model` / `timeout`）**可注入**，默认值只来自配置层——测试时整体替换 config 即可，不用 mock 内部函数。
- HTTP 用 `respx` 这类拦截器 mock，测试套件**零网络依赖**；并用"封死 socket"的测试反向保证默认路径真的不联网。
- CLI 层用 `CliRunner` 做集成测试：参数解析、退出码、stdout/stderr 全部被真实执行到。

### 2.9 一句话总结

普通 HTTP client 保证"请求送达、响应可解析"；合格的 AI Client 还要额外保证四件事：**答案完整（finish_reason 语义）、失败可分诊（错误分层）、花费可核算（usage + request_id）、行为可替换（依赖注入 + 零网络测试）**。前两条决定正确性，后两条决定它能不能进生产。
