# Chat Completion、temperature 与 max tokens

> 配套代码：`agent_study/llm_client.py`（一个最小调用函数，MiniMax-M3.1-Flash）
> 相关笔记：`note/token_basic.md`（token 怎么算）、`note/prompt_basic.md`（prompt 怎么写）
> 复跑方式：`uv run pytest -q`；实验数据的采集方式见下文标注

Chat Completion 是调用 LLM 最经典的接口：把一段对话（messages）发过去，模型回一段文本。
今天要吃透三件事：**这个接口长什么样**、**temperature 到底改了什么**、**max tokens 到底限的是什么**。

---

## 一、接口长什么样

### 1.1 端点与最小请求

端点是 `POST {API_BASE_URL}/chat/completions`，只有两个必填字段：`model` 和 `messages`。

```bash
curl https://api.openai.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "用一句话解释 Agent"}],
    "temperature": 0.2,
    "max_completion_tokens": 200
  }'
```

`messages` 是一条**有序**的消息数组，每条消息至少两个字段：

| 字段 | 作用 |
|---|---|
| `role` | 谁说的：`system` / `developer` / `user` / `assistant` / `tool` |
| `content` | 说了什么（文本） |
| `tool_calls` | 助手请求调用工具时的结构化调用信息 |

关键点：**API 是无状态的**。"上一轮我说过什么"不会自动记住，必须由调用方把历史消息原样带回来。多轮对话的成本、耗时、上下文压力都随历史线性增长——这和普通程序的"会话变量"是两种心智模型。

### 1.2 响应结构

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1760000000,
  "model": "gpt-4o-mini",
  "system_fingerprint": "fp_44709d6fc8",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "Agent 就是会自己决策的 LLM。"},
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 9,
    "total_tokens": 21
  }
}
```

真正要盯的三个字段：

- **`choices[].message.content`** —— 回答正文；
- **`choices[].finish_reason`** —— **为什么停下来的**（见下节），这是判断"是不是被截断了"的唯一依据；
- **`usage`** —— 真实 token 用量，计费和成本估算的唯一权威来源（不要用本地估算代替它）。

`system_fingerprint` 是后端配置的指纹：配了 `seed` 想要可复现时，它变了就说明后端换版本了，之前的结果不可比。

### 1.3 finish_reason：五种停止原因

官方定义的全部取值：

| finish_reason | 含义 | 该做什么 |
|---|---|---|
| `stop` | 正常结束，或命中了 `stop` 序列 | 正常处理 |
| `length` | **达到了请求里指定的输出 token 上限** | 视作"被截断"，要么放宽上限，要么改短输出要求 |
| `tool_calls` | 模型要调用工具 | 执行工具后把结果作为 `tool` 消息回传，继续下一轮 |
| `content_filter` | 内容被内容过滤器拦下 | 走兜底分支，不要当成正常回答 |
| `function_call` | 旧版函数调用（已废弃） | 新代码一律用 `tool_calls` |

`length` 是最容易被忽略的一个：它**不报错**，HTTP 200、内容也是完整 JSON，但内容是半截的。程序不检查 `finish_reason`，就会把半截答案当成完整答案写进数据库或返给用户。

### 1.4 和 Responses API 的关系

Chat Completions 不是唯一的接口。OpenAI 现在推荐新项目用 **Responses API**（`POST /v1/responses`），
官方明确说推理模型（gpt-6-astra 等）在 Responses 上"能拿到更好的模型智能与性能"，
Chat Completions 仍然支持但属于兼容路径。两者参数名对不上：

| 语义 | Chat Completions | Responses API |
|---|---|---|
| 对话输入 | `messages` | `input` |
| 指令 | `messages[0].role=system` | `instructions` |
| 输出上限 | `max_completion_tokens`（旧名 `max_tokens`） | `max_output_tokens` |
| 推理强度 | `reasoning_effort` | `reasoning.effort` |
| 结构化输出 | `response_format` | `text.format` |

本项目 `note/prompt_basic.md` 里用的是 Responses 写法，今天的 Chat Completions 是另一条路：
**两者都要会读**，因为老系统、多数国内网关（Kimi 等）、OpenAI 兼容代理都还是 Chat Completions 形态。

---

## 二、temperature：它改的是"采样时的随机度"

### 2.1 机制

模型每生成一个 token，先算出每个候选 token 的分数（logit），再按概率分布采样。
**temperature 就是在 softmax 之前，把所有 logit 同除以 T**：

```
p_i = exp(z_i / T) / Σ_j exp(z_j / T)
```

- T 很小 → 分布被拉尖，最高分 token 的概率接近 1 → 输出**稳定但呆板**；
- T 很大 → 分布被压平，长尾 token 也有了机会 → 输出**发散但可能跑偏**；
- **T = 0 → 等价于 greedy 解码，永远取 argmax**。

官方定义：取值范围 **0 ~ 2**，默认 1.0。官方还提醒：`temperature` 和 `top_p` **建议只调一个**，两者一起调很难推理。

### 2.2 实测：同一个 logits 下不同 temperature 的分布

固定一组 logits（Java=3.0, Python=2.0, Go=1.0, Rust=0.5），按公式算：

| temperature | Java 概率 | 前 2 名累计概率 | 熵 H | 有效候选数 e^H |
|---|---|---|---|---|
| 0.0 | 1.0000 | 1.0000 | -0.0000 | 1.000 |
| 0.2 | 0.9933 | 1.0000 | 0.0407 | 1.042 |
| 0.7 | 0.7546 | 0.9354 | 0.7395 | 2.095 |
| 1.0 | 0.6308 | 0.8629 | 0.9930 | 2.699 |
| 1.5 | 0.5087 | 0.7698 | 1.1890 | 3.284 |
| 2.0 | 0.4423 | 0.7106 | 1.2710 | 3.564 |

> 熵 H 用自然对数；`e^H` 可以理解成"等效的有效候选个数"——T=0.7 时大约在 2 个候选间摇摆，T=1.5 时接近 3.3 个。

这解释了 temperature 的真实作用：**它不是"让答案更好"，而是"让答案在多少种可能之间摇摆"**。

### 2.3 和 top_p 的区别

`top_p`（nucleus sampling）砍的是**累计概率的尾部**，`temperature` 缩放的是**整个分布**：

| 组合 | 进入候选集的 token | 候选集概率质量 |
|---|---|---|
| T=1.0, top_p=0.5 | Java | 0.6308 |
| T=1.0, top_p=0.9 | Java, Python, Go | 0.9482 |
| T=1.5, top_p=0.9 | Java, Python, Go | 0.9039 |
| T=0.7, top_p=1.0 | Java, Python, Go, Rust | 1.0000 |

实践中：**想要"稳定"就把 temperature 调低；想要"多样"就调高；需要"多样性但别跑太偏"才考虑 top_p 做护栏**。

### 2.4 实践建议

| 场景 | 建议 temperature |
|---|---|
| 结构化抽取 / 分类 / 代码生成 / 翻译 | 0 ~ 0.3 |
| 日常问答、工具调用、Agent 决策 | 0.2 ~ 0.7 |
| 创意文案、多方案头脑风暴 | 0.8 ~ 1.2 |
| 追求多样性（批量生成变体） | 1.0 以上 |

### 2.5 三个必须知道的坑

1. **temperature=0 不等于可复现**。它是 greedy（每步取 argmax），但底层还有浮点计算、批处理、模型版本、服务端并发调度等不确定性。官方对 `seed` 的措辞也是"best effort（尽力而为），不保证确定"。**所以测试模型输出不能断言"逐字相等"，只能断言语义性质**（见 `note/prompt_basic.md`）。
2. **有些模型不接受自定义 temperature**。参数是否被接受由模型决定，官方 API 参考没有做模型级豁免说明——所以别假设传了就有效，**发完请求要看响应和错误**，被拒时通常返回 HTTP 400。
3. **temperature 不影响是否被截断**。截断只看 token 上限，和 temperature 正交——高 temperature 反而更容易写出啰嗦的内容、更快撞上限。

---

## 三、max tokens：限的是"输出预算"

### 3.1 三个名字，一个意思

| 参数 | 状态 | 说明 |
|---|---|---|
| `max_tokens` | **已废弃** | 官方明确：deprecated，且**与 o 系列模型不兼容** |
| `max_completion_tokens` | 当前推荐 | 输出 token 的上限，**包含可见输出 token 和 reasoning token** |
| `max_output_tokens` | Responses API | 语义同上，覆盖 reasoning token、可见输出 token、以及不可见的格式 token |

**结论：Chat Completions 用 `max_completion_tokens`，不要再用 `max_tokens`。**
本项目的最小调用函数只暴露 `prompt`，压根不会发这个过时参数——避免把它悄悄发到线上、等到 o 系列模型报 400 才发现。

### 3.2 它和上下文窗口的关系

上下文窗口是"输入 + 输出共享"的总额度（见 `note/token_basic.md`）。
所以给输出留预算时，必须先减掉输入：

```
可用输出预算 ≤ 上下文窗口 − 输入 token − 安全余量
```

以 gpt-6-astra 为例（官方模型页）：上下文窗口 1,050,000，最大输入 922,000，最大输出 128,000。

### 3.3 实测：上限如何触发截断

本地 mock 服务端按 `max_completion_tokens` 截断并回填 `finish_reason` 后实测（这批数据记录于代码精简之前，mock 脚本已按需移除；表中行为与官方 `finish_reason` 定义一致）：

| max_completion_tokens | finish_reason | 输出 token 数 | 是否被截断 | 输出结尾片段 |
|---|---|---|---|---|
| 未设置 | `stop` | 58 | 否 | Chat Completions is the classic chat endpoint:… |
| 200 | `stop` | 58 | 否 | Chat Completions is the classic chat endpoint:… |
| 50 | `length` | 50 | 是 | … when the reply hits that ceiling the text |
| 20 | `length` | 20 | 是 | … messages, the model replies with choices, |
| 10 | `length` | 10 | 是 | … Completions is the classic chat endpoint: |

要点：

1. **不设置上限 ≠ 无上限**，只是让服务端用默认值；客户端自己算"剩余窗口"再决定给多少更可控。
2. **截断不报错**：HTTP 200、`choices` 完整、只是内容短了、`finish_reason=length`。
3. **截断发生在 token 边界**，中文可能被切在多字节字符中间（客户端要用 UTF-8 容错解码）。

### 3.4 推理模型上的特殊坑

reasoning 模型会先生成**不可见的 reasoning token** 再输出正文。官方说明里有两个关键事实：

- reasoning token **不可见，但占上下文窗口、并且按输出 token 计费**；
- 用 `max_output_tokens` 限制总生成量时，**可能在产生任何可见输出之前就撞上限**——此时你会"付了钱但拿到空回复"。

Responses API 对应的是 `status=incomplete` + `incomplete_details.reason=max_output_tokens`；
Chat Completions 对应的是 `finish_reason=length`。
官方建议刚开始用推理模型时，**至少为 reasoning + 输出预留 25,000 token**。

### 3.5 和成本的关系

按 gpt-6-astra 官方定价（输入 $10 / 缓存输入 $1 / 输出 $50 每 100 万 token）：

| 场景 | 输入 token | 输出 token | 预估成本 |
|---|---|---|---|
| 正常回复（输出 500） | 1,000 | 500 | $0.035000 |
| 命中缓存的同一 prompt（输出 500） | 1,000 | 500 | $0.026000 |
| 输出上限设为 10000 且真的写满 | 1,000 | 10,000 | $0.510000 |
| n=4，每个都写满 10000 | 1,000 | 40,000 | $2.010000 |

**输出单价是输入的 5 倍**，所以 `max_completion_tokens` 才是成本上限的主要杠杆：它直接乘上输出单价。
另外 `n>1` 时每个候选各自生成，官方明确"按所有候选生成的 token 总和计费"。

---

## 四、其它值得知道的参数

| 参数 | 作用 | 注意 |
|---|---|---|
| `n` | 生成几个候选 | 费用按所有候选的总 token 算 |
| `stop` | 最多 4 个停止序列 | o3 / o4-mini 等最新推理模型不支持 |
| `seed` | 尽力而为地确定性采样（Beta） | 不保证一致，用 `system_fingerprint` 监控后端变化 |
| `response_format` | `json_schema` / `json_object` | 服务端强校验的结构化输出，见 `note/prompt_basic.md` |
| `reasoning_effort` | 推理强度 | 取值 `none`/`minimal`/`low`/`medium`/`high`/`xhigh`/`max`，**因模型而异**；GPT-6 Astra 不支持 `none` |
| `verbosity` | 回答详略（low/medium/high） | 与 temperature 是两件事：一个控"随机度"，一个控"篇幅" |
| `stream` | SSE 流式返回 | 本笔记未覆盖，留到 Streaming 专题 |
| `tool_choice` | 强制/禁止/自动调用工具 | `none` / `auto` / `required` / 指定某个函数 |

> 补充：官方推理模型指南明确写了 **Chat Completions 不支持 GPT-6 Astra / GPT-6.1 Sol 的 function calling**，要用工具得走 Responses API。这直接影响"用 Chat Completions 做 Agent 工具调用"的可行性评估。

---

## 五、Java 视角的映射

| 普通程序 | LLM 调用 |
|---|---|
| 方法参数 | 请求体字段（`temperature`、`max_completion_tokens`） |
| 返回值类型 | `choices[].message.content` + `finish_reason`（**没有异常，但可能只返回一半**） |
| 异常处理 | HTTP 状态码 + `finish_reason` 双轨：前者是传输失败，后者是"业务上没写完" |
| 事务回滚 | 上下文窗口超限 → 需要主动截断/摘要历史 |
| 幂等键 | `seed`（尽力而为，不是幂等保证） |
| 成本护栏 | `n`、`max_completion_tokens` 控制单次请求的资源占用上限 |
| 日志里的敏感字段 | 密钥走 `Authorization` 头；**prompt 与回答本身也是敏感数据，要考虑脱敏与留存** |

一句话：**普通程序的失败会抛异常，LLM 的失败会"看起来成功地返回一个不完整的答案"**。
所以客户端必须把 `finish_reason` 和 `usage` 当成一等返回值，而不是只看 `content`。

---

## 六、工程清单（可直接抄进代码评审 checklist）

- [ ] 用 `max_completion_tokens`，不用 `max_tokens`
- [ ] 输入 + `max_completion_tokens` ≤ 上下文窗口（留余量）
- [ ] 每次调用后检查 `finish_reason`，`length` 要走"内容不完整"分支
- [ ] 成本按 `usage` 结算，不用本地估算替代
- [ ] `n > 1` 时记得费用是乘以候选数的
- [ ] 结构化输出优先用 `response_format`（服务端校验），不要靠 prompt 里的"请输出 JSON"
- [ ] 测试断言语义性质，不做逐字相等断言
- [ ] 需要可复现时用 `seed` + 记录 `system_fingerprint` + 固定模型版本
- [ ] 推理模型为 reasoning 预留足够 token（官方建议起步 25,000）
- [ ] 密钥只走请求头，不进日志

---

## 七、面试问答

### Q: Chat Completion 和老的 Completion 接口有什么区别？

老接口（`/v1/completions`）是把一段文本续写下去，模型只认"字符串"，没有对话概念。
Chat Completions 引入了 `messages`（带 role 的多轮对话）、`tools`（函数调用）、`response_format`（结构化输出）和 `finish_reason`（明确告诉你为什么停）。
老接口在现代模型上基本不可用，现在只该记 Chat Completions 和 Responses API 两条路。

### Q: 一次 chat completion 请求由什么组成？响应里最该看哪三个字段？

请求：`model`（必填）、`messages`（必填）、外加生成参数（`temperature`、`max_completion_tokens`、`n`、`stop`、`seed`、`response_format` 等）。
响应最该看三处：`choices[0].message.content`（正文）、`choices[0].finish_reason`（为什么停）、`usage`（真实 token 用量，计费依据）。
本项目 `agent_study/llm_client.py` 里的最小函数 `chat()` 只返回 `content` 文本；
要拿 `finish_reason` / `usage`，需要直接读响应 JSON（`data["choices"][0]["finish_reason"]`、`data["usage"]`）。

### Q: temperature 到底做了什么？

它把每个候选 token 的 logit 同除以 T 再做 softmax，也就是**改变采样分布的尖锐程度**。
T 小 → 分布尖，几乎总选最高分；T=0 → greedy，每步取 argmax；T 大 → 分布平，长尾 token 也有机会被选中。
它不改变模型的能力上限，只改变"在几种答案之间摇摆"。

### Q: 想让输出稳定可复现，temperature=0 就够了吗？

不够，但方向对。temperature=0 是 greedy，解决了"采样随机性"，但底层仍有浮点、批处理、模型版本、服务端调度等不确定性，
官方对 `seed` 的说法也是"尽力而为、不保证确定"。
而且 greedy 本身不等于"正确"——它只是每次都走同一条最可能的路，错了就稳定地错。
**所以生产上要的是"低 temperature + 结构化输出 + 断言性质的测试 + eval 集"，而不是"逐字比对"。**

### Q: max_tokens 和 max_completion_tokens 有什么区别？该用哪个？

`max_tokens` 已被官方标记 deprecated，并且与 o 系列模型不兼容；`max_completion_tokens` 是当前的参数，
它的上限**包含可见输出 token 和 reasoning token**。Responses API 里对应的叫 `max_output_tokens`（还额外覆盖不可见的格式 token）。
用 `max_completion_tokens`。

### Q: 设了 max_completion_tokens，回答会被截断吗？怎么知道？

会，而且**不会报错**。输出到上限就被切掉，`finish_reason` 变成 `length`。
所以客户端必须显式检查 `finish_reason`：`length` 意味着这是半截答案，要么放宽上限，要么在 prompt 里要求"先给结论再给理由"，
要么把任务拆小。对推理模型更要注意：可能撞上限时一个可见 token 都还没产出，等于白付钱。

### Q: 输入和输出加起来超了上下文窗口会怎样？

输入 + 输出共享同一个窗口预算（`max_completion_tokens` 是从同一预算里预留给输出的）。
超了之后各家行为不统一：有的直接报 context length exceeded，有的截断、有的按策略丢弃消息。
应用层的正确做法是自己先用 tiktoken 估算（见 `note/token_basic.md`），在接近上限时主动截断或摘要历史，别依赖服务端行为。

### Q: 为什么说 max_completion_tokens 是成本的主要杠杆？

因为 API 按 token 计费，而输出 token 的单价通常是输入的数倍（gpt-6-astra 是 $50 vs $10 每百万）。
`max_completion_tokens` 直接乘上输出单价构成单次请求的成本上限——设 10000 和设 500，差的不是"上限"而是**实付金额**。
再加上 `n>1` 时每个候选独立计费，两者相乘就是最坏情况的花费。

### Q: 除了 temperature 和 max tokens，还有哪些控制输出的手段？

按"控制什么"分四类：
**控格式**——`response_format`（`json_schema` / `json_object`，服务端硬校验）、`stop`（最多 4 个停止序列）；
**控数量与多样性**——`n`（候选数）、`seed`（尽力而为的确定性）；
**控推理投入**——`reasoning_effort`（none→max，因模型而异）；
**控篇幅**——`verbosity`（low/medium/high）。
注意"篇幅"和"随机度"是两件事：temperature 高的回答不一定更长，但更容易啰嗦、更容易撞上 token 上限。

### Q: Chat Completions 和 Responses API 该怎么选？

新项目优先 Responses：官方明确推荐，推理模型在它上面效果更好，且 GPT-6 Astra / GPT-6.1 Sol 的 function calling **只在 Responses 支持**。
Chat Completions 的价值在于兼容面广——老系统、国内网关、OpenAI 兼容代理基本都还是这个形态，本项目 `.env.prod` 用的 Kimi 也是。
工程上的现实答案通常是**读懂两种、自己封装成一层**，让上层业务不感知差异。

---

## 八、未覆盖 & 下一步

- **Streaming（SSE）**：`stream=true` 时的分块协议、`stream_options.include_usage`、中途断开与重连，留到 Streaming 专题。
- **重试与限流**：`stream`/非流式请求的超时、429 退避可以复用 `agent_study/async_client.py` 里已实现的退避逻辑。

---

## 参考来源

- Chat Completions 请求参数定义（由 OpenAPI spec 生成）：<https://github.com/openai/openai-python/blob/main/src/openai/types/chat/completion_create_params.py>
- Chat Completions 响应结构（含 `finish_reason` 取值）：<https://github.com/openai/openai-python/blob/main/src/openai/types/chat/chat_completion.py>
- `usage` 字段结构：<https://github.com/openai/openai-python/blob/main/src/openai/types/completion_usage.py>
- 推理模型指南（reasoning token、输出上限、模型支持差异）：<https://developers.openai.com/api/docs/guides/reasoning>
- GPT-6 Astra 模型页（上下文窗口、定价、端点支持）：<https://developers.openai.com/api/docs/models/gpt-6-astra>
