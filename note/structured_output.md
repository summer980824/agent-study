## structured output

### 问题：自由文本输出等于没有契约

让模型"输出 JSON"然后在代码里 `json.loads` 一下，看着能跑，实际没约束力：
字段名对不对、类型对不对、该有的字段有没有，全靠模型自觉。
失败时也不报错——`json.loads` 抛了，或者更糟，解析成功但字段是空的，
调用方拿到一个 `None` 继续往下跑。

结构化输出要解决的就是这件事：**把"期望的输出形状"从自然语言描述变成机器可校验的契约。**

### 核心：一份 schema，两个角色

关键在于 schema 同时干两件事，而且必须是**同一个**：

| 角色 | 作用 | 实现 |
|---|---|---|
| 生成前 | 告诉模型该输出什么形状 | `model_json_schema()` 注入提示词 |
| 生成后 | 检查模型实际输出对不对 | `model_validate()` 硬校验 |

只做前者，模型的输出没有任何保证；只做后者，模型不知道要输出什么，校验几乎必然失败。
两者用同一份 schema，提示词里的契约和代码里的契约才不会漂移。

完整链路是三段：

```
定义模型  →  schema 进 prompt  →  用模型类校验输出
```

`note/prompt_basic.md` 的「结构化输出：把 schema 放进 prompt」一节讲的是通用原理，
这里讲的是本项目怎么落地，以及踩过的坑。


### 坑一：输入形态决定模型行为

第一次跑 demo 时，模型把输入原样返回了。看起来像 bug，其实是**任务本身为空**：

| 输入形态 | 模型行为 | 本质 |
|---|---|---|
| 已经是 schema 的合法实例 | 原样返回 | 抄是最优解，没有可做的变换 |
| 字段名不同、部分缺失 | 映射 + 推断补齐 | 归一化 / 抽取 |
| 自由文本 | 抽取 + 推断 | 信息抽取 |

所以 `chat_structured` 的输入应该是**原始素材**，不是一条已经填好的记录。
同时给了 `instruction` 参数（`llm_client.py:54` `DEFAULT_STRUCTURED_INSTRUCTION`），
默认指令是「根据输入内容抽取并填写对象，信息缺失时按输入合理推断，不要原样照抄输入」——
指令放在 schema 之前，先说清"做什么"再说"按什么格式"。

这个机制本质上不是代码补全，而是**模板 + 素材 → 填槽**：
schema 是槽位模板，输入是素材，模型干的是把源材料里的信息归位到对应字段。

### 坑二：Pydantic 的约束顺序

`field_validator` 默认是 `mode="after"`，而 `min_length` / `max_length` 这类字段约束
在**核心 schema** 里求解，**先于** `after` 校验器执行：

```python
@field_validator("title", "description")
@classmethod
def not_blank(cls, v: str) -> str:
    stripped = v.strip()
    if not stripped:
        raise ValueError("value cannot be blank")
    return stripped
```

推论：

- `title` 的 100 字符上限算的是 **strip 之前的原始字符串**；
- 空串 `""` 被 `min_length=1` 拦下，抛 "String should have at least 1 character"；
- 纯空白 `"   "` 原始长度是 3，能过 `min_length`，于是走到 `not_blank` 才被拦下，
  抛 "value cannot be blank"；
- 想让长度按 strip 后算，得改用 `mode="before"` 的校验器。

本项目选的是"按原始字符串算"：JSON Schema 里的 `maxLength: 100` 会被原样写进提示词告诉模型，
校验就该用同一把尺子量，口径不一致才是 bug。

### 坑三：失败分三层，别搞混

`chat_structured` 是三层校验叠起来的：

| 层 | 检查什么 | 不通过时 |
|---|---|---|
| HTTP | 状态码 ≥ 400 | `RuntimeError`（"API 401: ..."） |
| 响应结构 | `choices` / `message` / `content` 是否存在、`finish_reason` | `LLMResponseError`（含截断、内容过滤的特化子类） |
| 业务 schema | 字段齐不齐、类型、`priority` 枚举、`estimated_minutes` 范围 | `LLMStructuredOutputError` |

顺序也有讲究：**先**判断截断和内容过滤（`llm_client.py:239`），**再**解析 JSON，最后才校验 schema。
被截断的输出即使恰好是合法 JSON，也是半截的，不该被当成有效结果。

另外，**只有输出校验，输入不校验**。`prompt` 传的是原始素材，本来就不该符合 schema，
把它当 schema 校验会直接拒掉。

### 坑四：结构化结果和响应元信息要一起拿

`chat_structured()` 原来只返回校验后的模型实例，`finish_reason` 和 `usage` 就丢了。
但这两个字段恰恰是成本和正确性的唯一依据（见 `note/chat_completion.md`）：
`usage` 用来记账，`finish_reason` 用来发现"看起来成功、其实是半截"的输出。

现在的分工是两个入口，共用同一套校验逻辑：

| 函数 | 返回 | 用在哪 |
|---|---|---|
| `chat_structured()` | 模型实例 | 只关心业务字段 |
| `chat_structured_response()` | `(模型实例, ChatResponse)` | 还要记账、判断截断 |

`chat_structured()` 是 `chat_structured_response()` 的便捷包装，抛错行为完全一致：
截断 → `LLMResponseTruncatedError`，内容过滤 → `LLMResponseError`，
JSON 非法或不符合 schema → `LLMStructuredOutputError`。

顺带一个 HTTP 层的坑：**每个请求新建 `httpx.Client` 等于每次重新握手**，
连接池一点都复用不上。客户端改成进程内共享一个实例（`_get_client()`），
超时按请求传（`timeout=` 是 httpx 的 per-request 参数），需要提前释放连接时调 `close_client()`。

### 局限

1. **这是应用层的软约束，不是解码层的硬约束。**
   真正的结构化输出（如 OpenAI 的 `response_format` strict `json_schema`、Guidance/Outlines
   在采样时屏蔽非法 token）能保证输出必定合法 JSON。这里是"schema 写进提示词 → 解析 →
   校验 → 不合法就抛异常"，模型完全可能生成非法内容然后被拒。
   `chat_structured` 留了 `response_format` 参数（默认不发），服务端原生约束能开就开，两者叠加最稳。

2. **校验通过 ≠ 内容正确。**
   schema 管形状不管语义。`estimated_minutes=30` 从"大概半小时"推出来、`priority=medium` 判得准不准，
   schema 都管不了。需要准确度就得另配校验或人工确认。

3. **失败即抛，没有自愈。**
   模型输出不合规就直接抛 `LLMStructuredOutputError`，没有把错误回喂给模型重试。
   在长流程的 agent 里这通常意味着该节点失败，生产上更常见的是"校验失败 → 带错误信息重试 N 次"。

4. **模型输出不能做逐字相等断言。**
   模型输出是采样的，即使 `temperature=0` 也不保证逐字一致。
   测试只能断言语义性质：能解析、字段齐全、类型合法。


## 面试问题

### 1. 为什么普通文本输出不适合直接交给业务代码

- **没有可执行的契约**：文本只说明"大概要什么"，字段名、类型、必填项全靠约定，而约定不会在运行时报错。
- **失败是静默的**：`json.loads` 抛异常还算好的；更糟的是解析成功但字段缺失/为空，业务代码拿到 `None` 继续跑，错误一路飘到下游。
- **形状不可预期**：调用方只能靠正则和 `try/except` 硬抠，模型今天返回 `{"title": ...}`、明天返回 `{"task": ...}`，代码跟着改。
- **无法做类型检查和回归测试**：断言只能写"包含某几个词"，写不了"字段齐全、类型合法"。

### 2. JSON Schema 与 Pydantic 分别解决什么

| | 解决的问题 |
|---|---|
| **JSON Schema** | 契约的**语言无关描述格式**。给模型看（生成前约束输出形状），也给其他语言/服务消费。纯数据，无执行能力。 |
| **Pydantic** | 契约在 **Python 侧的落地与执行**：类型、默认值、自定义校验器、序列化，且能**反向生成** JSON Schema。 |

关系是单向的：**Pydantic → JSON Schema**（`model_json_schema()`），反过来不行。
所以「用 Pydantic 定义、用 JSON Schema 传话、用 Pydantic 校验」，全程一份契约，不会漂移。

### 3. Prompt 约束与服务端原生结构化输出的区别

| | Prompt 约束 | 服务端原生 |
|---|---|---|
| 强制力 | 软约束，模型可能不听 | 解码期约束（constrained decoding / CFG / 编译文法），保证合法 |
| 失败表现 | 输出不合法，靠事后校验拦 | 基本不失败 |
| 灵活性 | 高，换模型/换 schema 都能用 | 受服务商与模型支持度限制，且 strict schema 有子集限制 |
| 代价 | 失败率 + 需要重试兜底 | 少数模型/服务商不支持 |

两者叠加最稳：**原生保证形状，prompt 保证内容质量**。

### 4. 无效 JSON 和字段校验失败应该如何处理

- **无效 JSON / 不是对象** → 抛 `LLMStructuredOutputError`（`llm_client.py:389` `_extract_json_object`）。
- **字段校验失败** → 同样抛 `LLMStructuredOutputError`（`llm_client.py:251` `model_validate`），错误信息带原文片段便于排查。
- **铁律：不要 `except` 之后返回默认值或 `None` 继续跑**——那等于把显式失败换成了静默降级
  （就是上面「只完成结构化的一半」那个坑）。
- **更好的做法（当前未实现）**：把校验错误回喂给模型让它自己改一遍，重试 N 次；并区分
  「格式错」（可重试）与「语义约束不可能满足」（不可重试，直接失败）。

### 5. 当前服务商/模型是否支持原生结构化输出

**实测结论：不支持**（就本项目配置的端点与模型而言）。
实测时间：2026-10-26。对象：`api.minimax.io/v1` + `MiniMax-M3.1-Flash`。

| 探针 | 结果 |
|---|---|
| `response_format={"type":"json_object"}`，提示词**不含** "JSON" | 3/3 返回 Markdown 散文，**不是 JSON** |
| `response_format={"type":"json_schema","strict":true,"schema":TaskSchema}`，提示词含 "JSON" | 3/3 返回 JSON，但字段名是模型自拟的（`task`/`status`/`suggested_outline`/`duration_minutes`…），**3 次 3 套，均不符合 TaskSchema** |
| **完全不带** `response_format`，提示词含 "JSON" | 3/3 输出 JSON → 输出是不是 JSON **由提示词决定，不由参数决定** |
| 传非法 `type="not_a_real_type"`、或完全未知的参数 | 同样 HTTP 200 → 服务端不校验未知参数，**所以 "HTTP 200" 不能当作"支持"的证据** |

**明确标注未验证的部分**：

- MiniMax 官方文档页面是 JS 渲染的，本次没抓到正文 → **「官方是否声明支持」未验证**；
  以上结论只来自黑盒实测。
- 其他 MiniMax 模型、Anthropic 兼容端点（`agent_study/anthropic_chat.py` 走的那条）→ **未验证**。
- OpenAI / Anthropic / Gemini 等其他服务商的原生结构化输出 → **未在本项目内验证**
  （仅引用其官方文档的一般性说法）。
- 网上说法互相矛盾：M2.5 的 [GitHub issue](https://github.com/MiniMax-AI/MiniMax-M2.5/issues/4)
  说两种 `response_format` 都不支持，第三方站点说 M3 支持 `json_schema` —— **以实测为准**。

**推论**：本项目走「schema 进 prompt + 事后 Pydantic 校验」是正确路线。
`chat_structured` 的 `response_format` 参数在本服务商上**发不发都一样**，
要么删掉，要么注释里写明"本服务商实测不生效"，免得日后有人误以为有服务端硬约束。
