## SSE 客户端
SSE 客户端就是按照 SSE（Server-Sent Events）协议规范，从 HTTP 响应流里持续读取并解析事件的服务端/前端代码。服务端把响应头设为 Content-Type: text/event-stream，然后不断往连接里写事件；客户端则保持这个连接，逐条取出事件交给业务逻辑。
规范要点：
- 事件与事件之间用空行分隔
- 每个事件由若干字段行组成，最常用的是 data:，还有 event:（自定义事件名）、id:、retry:（重连间隔）
- 以冒号 : 开头的行是注释/心跳，客户端必须忽略（这就是 OpenAI 流里 reasoning 期间那些 keep-alive 行）
- 一个事件的多个 data: 行按换行拼接成完整 payload
一句话总结：一个「读 HTTP 流并按 data:/:/空行 规则切分事件」的解析器，浏览器可以用 EventSource 白嫖，但调 LLM 时通常要自己用 fetch/httpx 手搓一个。

# 流式 chat 接口设计

## 目标

给 `agent_study/llm_client.py` 加 `chat_stream()`，走 OpenAI 兼容的 SSE 流式接口，
边收边吐正文增量，同时把 `finish_reason` / `usage` 拿回来。

## 背景

当前 `llm_client.py` 只有同步非流式调用（`chat` / `chat_response` /
`chat_structured` / `chat_structured_response`），返回结构是 `ChatResponse`
（`content` / `finish_reason` / `usage`）。流式需要一个既能吐增量、又能带元信息的
返回结构，于是顺手把 `ChatResponse` 统一成 `ChatResult`，同步和流式共用同一个类。

## 决策

| 决策点 | 结论 | 理由 |
| --- | --- | --- |
| 同步还是异步 | 同步 generator | 模块内现有 API 全是同步，共享 `httpx.Client` 能复用连接池；`aiter_lines()` 只有异步响应才有，同步对应 `iter_lines()` |
| 元信息出口 | `ChatStream` 对象，属性 `.result` | `Iterator[str]` 本身带不出 `finish_reason` / `usage`；`.result` 是同一个对象，流进行中 `content` 实时累加 |
| 命名 | `ChatResponse` → `ChatResult`，保留旧别名 | 同步与流式统一到一个类；`ChatResponse = ChatResult` 别名让旧 import 不炸 |
| 流式错误语义 | 截断 / 内容过滤不抛异常，只用标志位 | 正文已经吐出去了，抛异常会打断并丢掉已生成的结果；只有 HTTP 状态码错误在流开始前抛 |
| `chat_stream()` 的时机 | **急切**（eager）：调用即建连、查状态码 | 与 `chat()` / `chat_response()` "调用即报错"的约定一致；HTTP 错误在返回 `ChatStream` 之前抛出 |
| `stream_options` | 可选 kwarg `include_usage`，默认 `False` | 默认不发的兼容端可能直接 400；开了才向服务端要 usage |

## 数据结构

```python
@dataclass
class ChatResult:
    content: str = ""
    finish_reason: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    reasoning_content: str = ""
```

与原来的 `ChatResponse` 的区别：

- 去掉 `frozen=True` —— 流式需要边流边写 `content`。同步路径一次性构造完整实例，语义不变。
- 副作用：非 frozen dataclass 没有 `__hash__`，`hash(ChatResult(...))` 会抛 `TypeError`。
  仓库内没有地方对它做哈希，是有意的取舍。
- 新增 `reasoning_content`：模块文档明确写了 OpenAI 兼容协议下思考内容在
  `reasoning_content` 字段。流式下丢弃思考增量会静默吞掉 token，所以单独累加，
  同步路径 `_parse_response` 也一并填充。

`ChatResponse = ChatResult` 作为兼容别名。

## API

```python
def chat_stream(
    prompt: str,
    *,
    base_url: str = "",
    api_key: str = "",
    model: str = settings.MODEL_NAME,
    reasoning_effort: str | None = None,
    temperature: float | None = None,
    max_completion_tokens: int | None = None,
    include_usage: bool = False,
    timeout: float = 120.0,
) -> ChatStream:
    ...
```

`ChatStream` 既是 `Iterator[str]`，也带 `.result`：

```python
stream = chat_stream("用一句话解释 Agent", api_key="...")
for delta in stream:
    print(delta, end="", flush=True)
print(stream.result.content, stream.result.finish_reason, stream.result.usage)
```

`ChatStream` 的约定：

- `__iter__` 返回 **self**（不是被包装的生成器），否则每次 `for` 都会重发请求
- **一次性**：迭代到底后不可重来
- `close()`：调用方 `break` 出循环后可主动释放连接，不必等 GC 回收生成器
- `.result` 是**同一个对象**，流进行中 `content` / `reasoning_content` 实时累加，
  `finish_reason` / `usage` 在对应 chunk 到达时落定。迭代完成前不要把它当最终结果用

## 请求

- 请求体：`payload = _build_payload(...)` 之后加 `payload["stream"] = True`；
  `include_usage=True` 时再加 `payload["stream_options"] = {"include_usage": True}`。
  **不改 `_build_payload`**，否则所有同步请求都会带上 `stream`
- 请求头：`Authorization: Bearer {api_key or settings.API_KEY}`（与 `_post` 一致）
  加 `Accept: text/event-stream`
- `timeout` 是**每次读取**的超时，不是总时长

## SSE 解析

```python
with _get_client().stream("POST", url, json=payload, headers=headers, timeout=timeout) as response:
    for line in response.iter_lines():
        ...
```

`httpx` 不做 SSE 解析，`iter_lines()` 按 `splitlines()` 语义切分，产出的行**带
`data:` 前缀**，空行产出 `""`。

逐行规则（顺序重要）：

1. 空行 → 跳过
2. 不以 `data:` 开头 → 跳过（keep-alive、注释行）
3. `data:` 后 `strip()` 为空 → 跳过
4. `strip()` 后是 `[DONE]` → 终止
5. `json.loads`；解析失败 → **跳过这一行**，不中断整个流
6. 不是 dict → 跳过
7. 有 `error` 字段 → 抛 `RuntimeError`（流内错误不能被静默吞掉）
8. 有非空 `usage` → 记下。**必须在校验 choices 之前**：OpenAI 兼容协议里
   带 usage 的尾块常常是 `{"choices": [], "usage": {...}}`，先校验 choices 会把
   唯一带 usage 的块吃掉
9. `choices` 缺失 / 非 list / 空 list → 跳过（纯元信息块，继续读）
10. `choices[0]` 不是 dict → 跳过
11. `choices[0].finish_reason` 非空 → 记下（后到覆盖先到）
12. `delta` 缺失 / 为 null / 不是 dict → 跳过（首个只带 `role` 的块属于这种）
13. `delta.reasoning_content` 非空 → 累加到 `result.reasoning_content`
14. `delta.content` 非空 → 累加到 `result.content` 并 yield（空串不 yield）

## 错误处理

- HTTP >= 400：**在建连后、迭代前**抛 `RuntimeError(f"API {status}: {body[:300]}")`。
  取 body 必须先 `response.read()`：流式响应下 `response.text` / `response.content`
  会抛 `httpx.ResponseNotRead`（httpx 0.28 不再自动读取）。
  注意 respx 会预读 `ByteStream` 响应，所以 respx 测不出这个坑，
  需要额外用 `httpx.MockTransport` 写一条回归测试。
- 单行 JSON 坏掉 → 跳过该行，流继续
- 流内 `error` 对象 → 抛 `RuntimeError`
- `finish_reason == "length"` / `"content_filter"`：**不抛异常**，
  只体现在 `result.finish_reason` 与 `result.truncated` 上，由调用方判断
- 流没有收到 `[DONE]` 就结束 → 不抛异常，`result.finish_reason` 保持最后一次拿到的值
  （可能为空字符串）。空 `finish_reason` 即表示"没拿到结束原因"，多半是连接中断

## 测试

`tests/test_llm_client.py` 新增 `TestChatStream`，用 respx 打桩 SSE 文本流。
SSE body 必须自带 `\n\n` 分隔符和结尾换行（`httpx.Response(text=...)` 不会自动补）。

注意：respx 会预读 `ByteStream` 响应，测试里 body 是**一次性到达**的，
所以不要断言"增量到达"，只断言 **yield 出来的增量序列**。

用例：

- 请求体带 `stream: true`，请求头带 `Accept: text/event-stream` 和 `Authorization`
- 默认不带 `stream_options`；`include_usage=True` 时带上
- 逐块 yield，拼接结果等于完整正文
- `.result.content` 完整、`.result.finish_reason == "stop"`、`.result.usage` 取到
- `[DONE]` 终止
- 跳过空行、非 `data:` 行、坏 JSON 行
- `data:` 后**不带空格**也能解析
- usage-only 尾块 `{"choices": [], "usage": {...}}` → `usage` 有值且不报 IndexError
- 首个只带 `role` 的块 / 没有 `delta` 的块 → 不崩、不 yield 空串
- 流内 `{"error": {...}}` → 抛 `RuntimeError`
- HTTP 401 → `RuntimeError`（在 `chat_stream()` 调用处就抛）
- 错误 body 的读取用 `httpx.MockTransport` 单独回归（绕开 respx 的预读）
- `finish_reason == "length"` 不抛异常，`truncated is True`
- 缺 usage → `usage == {}`
- `reasoning_content` 增量累加到 `.result.reasoning_content`


## 面试题

### Q: Streaming 对用户体验和系统架构有什么影响？

**一句话结论：流式改变的是"等待的形状"，不是"等待的长度"。**
它把一次"干等 N 秒、然后一次性砸出一大坨"的交互，换成"看着它一点点写出来"，
同时把后端从"请求-响应"模型逼进"长连接、有状态、会中途失败"的模型。
前者是产品红利，后者是架构代价 —— 两者必须一起设计，不能只要前半段。

---

#### 一、对用户体验的影响

**1. 感知延迟（perceived latency）比真实延迟更重要**

流式不减少总耗时。模型吐 500 个 token 还是要那么久，token 成本一分不少。
它减少的是**用户"不知道发生了什么"的时间**。

指标要从"端到端延迟"拆成两个：

| 指标 | 含义 | 流式的影响 |
|---|---|---|
| TTFT (Time To First Token) | 发出请求到拿到第一个字 | 决定"用户觉得快不快"，通常 200-500ms 就够用 |
| 总时长 | 到最后一个 token | 通常没变，甚至因为流式增加了协议开销 |
| 吞吐 (tokens/sec) | 出字速度 | 通常略低于非流式（每 chunk 有序列化开销） |

用户感知的等待 ≈ TTFT，而不是总时长。心理学上有个稳定结论：
**只要有可见的进展，用户愿意等的时间会显著变长**。一个 20 秒的请求，
如果 300ms 就开始出字，用户会觉得"挺快"；如果 20 秒后一次性弹出，用户会觉得"卡死了"。

**2. 认知节奏要匹配**

非流式下，模型产出速度（tokens/sec）远超人阅读速度，差值全变成"看之前白等"的
冗余时间。流式把生成速率拉回阅读速率附近，用户的注意力正好跟得上。

但反面也要知道：**打字机效果 ≠ 真流式**。很多产品会先拿到完整结果、再用前端
定时器逐字播放 —— 那是假的流式，用户体验一样（甚至更顺，因为节奏可控），
但 TTFT 没变，还多了一次全量缓冲。对"内容必须完整才能用"的场景，
假流式往往比真流式更划算。

**3. 可中断 = 可放弃 = 可省钱**

流式下用户看到第一段就知道跑偏了，可以立刻停。
这是**唯一能把"模型在错误方向上多吐 4000 token"变成"只吐 200 token"**的机制，
直接对应真金白银。前提是取消要真的往下游传：
关掉上游连接 → 服务端感知到 → 停止推理。

**4. 错误暴露时机前移，但错误语义变复杂**

非流式：成功或失败，二值。
流式：**连接前失败**（鉴权 401、参数 400、限流 429）、**流中失败**（上游中途挂了、
内容过滤触发）、**流未正常结束**（连接断了，没收到 `[DONE]`）—— 三种都得区别对待。

这就是我们实现里为什么把错误分了层：

- HTTP >= 400 → 在**返回流之前**就抛（`chat_stream()` 是急切的：建连即查状态码）
- 流内收到 `{"error": ...}` → 中途抛 `RuntimeError`
- 收到 `[DONE]` 之外的结束 → 不抛，`finish_reason` 留空，由调用方判断

**5. 不是所有场景都该用流式**

| 场景 | 该流式吗 | 原因 |
|---|---|---|
| 对话、写作、代码生成 | 该 | 人在等，读得越早越好 |
| 长任务（跑 agent、查资料） | 该 | 没有流式就是"转圈 3 分钟" |
| 结构化输出（JSON Schema） | 不该 | 用户看到半截 JSON 反而困惑；校验失败要整份回滚 |
| 工具调用 | 不该 | 参数必须完整才能执行；中途执行有副作用风险 |
| 批处理 / 后台任务 | 不该 | 没人盯着，多一次网络往返 |
| 表格、图表数据 | 看情况 | 数据本身是原子的，分片显示可能反而更难读 |

**核心判断：流式的前提是"人会在结果未完整时就开始消费它"。**
如果不成立，流式只是白白增加复杂度。

**6. 无障碍与失败反馈**

流式的"逐步推进"天然是可访问性友好——屏幕阅读器可以逐段朗读；
但也意味着**必须有明确的中断反馈**：连接断在 60% 时，
UI 上要显示"生成中断，已生成的部分如下"，而不是让用户对着半个答案发呆。
这是流式产品最容易漏掉的一环。

---

#### 二、对系统架构的影响

**1. 连接生命周期：从短连接变成长连接**

这是最硬的架构约束。

- HTTP/1.1 浏览器对同源有 6 条并发连接的限制，流式把连接占用时间拉长一个数量级
  → 用户开 6 个对话，第 7 个就排队。
- 反向代理**默认会缓冲**，一旦缓冲，流式就退化成"等全部到齐再一次性吐出"，
  白做。Nginx 必须 `proxy_buffering off;`（或让应用发 `X-Accel-Buffering: no`）。
- 网关 / LB / CDN 的 idle timeout 需要单独调大，
  但**读超时是"两次 chunk 之间的间隔"，不是总时长** ——
  这个语义区别在代码里必须写清楚，否则会配错。

我们代码里的 timeout 就是这个语义：

```python
# timeout 是每次读取的超时，不是总时长
response = stack.enter_context(
    _get_client().stream("POST", url, json=payload, headers=headers, timeout=timeout)
)
```

**2. 错误处理：从"二值"变成"部分成功"**

这是架构上最本质的变化。非流式接口只有两种结局；流式接口引入了
**"我拿到了一半，而且不知道剩下的一半还有没有"** 这个第三状态。

处理这个第三状态要三件套：

1. **区分错误发生在流前还是流后**（决定能不能直接抛）
2. **保留已产出的内容**（不要因为一个尾部错误把前 80% 扔掉）
3. **明确"正常结束"的标志**（`[DONE]` / `finish_reason`）

我们选择：流式的截断和内容过滤**不抛异常**，只用 `result.truncated` /
`result.finish_reason` 表达 —— 因为正文已经吐出去了，抛异常会把已生成的内容一起丢掉。
这跟同步路径"截断就抛 `LLMResponseTruncatedError`"是有意的差异。

**3. 取消与幂等：一个被忽略的坑**

用户关掉页面 / 点"停止"，客户端断开。上游会不会继续推理到结束？

- 如果不取消：token 白烧，连接白占。用户体验是"我明明停了它还在算"。
- 如果取消：**重试语义变复杂**。同一个请求重发一次，是接着上次继续，还是从头再来？
  计费上这是两次推理。

所以流式系统必须显式设计：取消如何传播、幂等键怎么给、断点续传怎么做
（SSE 规范里有 `id:` 和 `Last-Event-ID` 可以用，但 OpenAI 兼容接口基本没实现）。

**4. 背压（backpressure）：生产快于消费怎么办**

服务端产 token 的速度由 GPU 决定，客户端渲染速度由浏览器决定。
慢客户端如果无限缓冲，内存会爆；直接断连，用户体验更差。

- TCP 层面有背压：客户端读得慢 → 服务端的 socket 发送缓冲区满 → 服务端写阻塞。
  这天然限流，但**要求你不能把 chunk 攒在内存队列里**。
- 应用层如果用了队列 / 消息中间件做扇出，队列就是新的背压点，必须有界。

**5. 水平扩展：流式其实更好扩，但"断线重连"变难**

好的方面：每个流是一个独立的 HTTP 请求，天然无状态，
可以随便往哪个实例上加 —— 比长轮询、WebSocket 那种有状态会话好扩得多。

难的部分：**用户刷新页面 / 断网重连后能不能接上？**
- 不接上 → 重新生成一遍，成本翻倍
- 接上 → 需要把某个会话的流状态外置（Redis / Kafka），需要会话亲和

**6. 可观测性：指标口径全变**

非流式时代盯 latency、error rate、QPS 就够了。流式要加：

| 指标 | 为什么 |
|---|---|
| TTFT 分布（P50/P95/P99） | 用户感知的第一因，比总延迟重要 |
| 流中断率 / 提前断开率 | 连接问题的唯一信号 |
| 平均流持续时长 | 连接池容量规划依据 |
| 并发活跃流数 | 不是 QPS，是"同时挂着的连接数" |
| 输出 token / 秒 | 吞吐健康度 |

日志也要改成"每个 chunk 打点"或"聚合到流结束时打一条"，
否则日志量会随 token 数线性爆炸。

**7. 计费：token 不变，但"取消"直接影响账单**

流式不改变 token 计费规则。但因为可以在用户放弃时提前断流，
**实际账单会下降** —— 前提是取消真的传到了上游。
这也是很多团队上流式之后的意外收益。

反过来，`usage` 在流式里不一定拿得到：兼容端可能只在尾部 chunk 给，
或者根本不给（我们为此专门加了 `include_usage` 参数，默认关闭，
因为不认 `stream_options` 的端会直接 400）。**拿不到 usage 就没法精确记账**，
这是流式方案必须提前想清楚的事。

---

#### 三、面试怎么答（结构建议）

按这个顺序讲，30 秒到 3 分钟都能收住：

1. **先给结论**："流式不减少总耗时，它把等待从'不可见'变成'可见'，
   降低的是感知延迟；同时它把后端从请求-响应模型逼进长连接模型，
   带来连接管理、错误语义、取消幂等、监控口径四个架构负担。"
2. **体验**：TTFT vs 总时长、可中断省 token、错误暴露前移，
   再补一句"不是所有场景都该流式（结构化输出、工具调用）"。
3. **架构**：挑最有把握的两三个讲透 ——
   长连接（代理缓冲要关）、部分成功（错误分三层）、
   取消与幂等（最容易被问到，也最能体现深度）。
4. **收尾取舍**："所以流式的判断标准是**用户会不会在结果完整前就开始消费它**；
   会，就上；不会，就别上。"

面试官大概率会追问的三个点，提前准备好：

- **"那流式的 token 成本更高吗？"** —— 不更高，同一批 token；
  但协议开销（每 chunk 一次序列化 + 心跳）会略增。
  真正的收益是早停省 token。
- **"怎么保证流式和不流式返回的模型结果一致？"** —— 不保证完全一致。
  采样是随机的，流式的分块边界依赖服务端调度。
  要一致性就固定 `seed` + `temperature=0`，但那会牺牲多样性。
  实务上别承诺"流式和非流式逐字相同"。
- **"流式中途断了怎么办？"** —— 保留已产出内容 + 标记未完成状态 +
  允许重试（带幂等键）。这就是我们代码里"没有 `[DONE]` 就不抛、
  `finish_reason` 留空"的由来。

---

#### 四、和本仓库实现的对应关系

| 架构关注点 | 本仓库的做法 |
|---|---|
| 代理缓冲 | 客户端侧直连 httpx，不经代理；部署时记得关 `proxy_buffering` |
| 连接生命周期 | 共享 `httpx.Client` 复用连接池；`ChatStream.close()` 提前释放 |
| 读超时语义 | `timeout` 是每次读取的超时，不是总时长 |
| 错误分层 | 流前 HTTP 错误抛异常 / 流内 `error` 对象抛异常 / 流未正常结束不抛 |
| 部分成功 | 截断与内容过滤不抛，用 `result.truncated` / `finish_reason` 表达 |
| 心跳/注释行 | `data:` 之外的行（keep-alive）一律跳过 |
| 精确记账 | `include_usage=True` 才向服务端要 usage，默认关闭避免兼容端 400 |
| 取消 | `close()` 关闭响应；真正停止上游推理要看服务端是否感知到断开 |

