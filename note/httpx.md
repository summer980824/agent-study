# httpx 学习笔记

## 运行 demo

```bash
uv run python -m agent_study.http_demo
```

## `httpx.get` vs `httpx.Client`

| 用法 | 特性 |
| --- | --- |
| `httpx.get(...)` | 一次性函数,每次调用内部新建连接、发请求、关闭 |
| `httpx.Client()` | 长连接会话对象,复用 TCP/TLS 连接,可配置默认值 |

## 异常处理分工

### `client.py` 负责 —— 传输层 / 协议层错误

- 连接超时
- DNS 失败
- 401 / 403 / 429 之外的 4xx / 5xx 统一抛一个自定义异常

### 调用方负责 —— 业务语义错误

- 200响应成功
- 用户不存在 vs 网络错误该给用户看什么提示

## `Response.raise_for_status()`

- 状态码在 `200~299`:什么都不做
- 其他状态码:抛 `httpx.HTTPStatusError`


## 异步调用
async def：定义异步函数，调用时返回协程对象。
await：等待异步操作完成，同时允许事件循环调度其他任务。
async with：确保使用结束后关闭异步 HTTP Client。

### `asyncio.gather`
用于并发执行多个协程(或 Future/Task),在一线程内并发跑多个协程，等待它们全部完成后,按传入顺序收集返回值。结果列表的顺序与传入顺序一致,与各任务完成的先后无关。
默认行为下, 第一个异常会传播给调用方, 其他尚未完成的任务不会因此自动取消。
如果希望把异常当作结果收回来,并在所有任务完成后统一检查。用 return_exceptions=True

### 实际使用建议
要结果、要顺序:用 gather(3.11+ 优先用 TaskGroup)
要超时、要按完成顺序处理:用 asyncio.wait
想要拿到 Task 句柄做单独操作:先 create_task,再传给 gather
任意任务失败都不能拖垮别人:开 return_exceptions=True,拿到结果后再统一判断

## 问题：asyncio 适合解决什么类型的问题？为什么它不等于多线程或多进程并行？
参考回答：
1. asyncio 特别适合 I/O 密集型任务，例如 HTTP 请求、数据库访问和网络通信。
2. 当一个协程通过 await 等待 I/O 时，事件循环可以调度其他就绪的协程。
3. 这使程序能够更有效地利用等待网络响应的时间。
4. asyncio 通常在单线程事件循环中进行协作式调度，并不意味着多个 CPU 密集型任务同时执行。
5. CPU 密集型任务通常需要考虑多进程、线程池或原生计算库等方案。
对于 AI 工程，异步可以帮助服务高效地等待多个模型 API 或检索 API 的响应，但不会自动降低单次模型推理的计算时间。


## asyncio.Semaphore 控制并发上限
main()
 ├─ 创建 Semaphore(3)            计数器 = 3
 ├─ 创建 stats / started
 ├─ async with AsyncGitHubClient()    建立连接池
 ├─ gather 派发 6 个 fetch_user
 │    └─ fetch_user_1 抢到信号量 -> active=1
 │    └─ fetch_user_2 抢到信号量 -> active=2
 │    └─ fetch_user_3 抢到信号量 -> active=3
 │    └─ 4,5,6 挂起等待
 │    └─ 谁先 await 返回，谁先 release，下一个再进来
 ├─ gather 返回结果列表
 ├─ 退出 async with -> aclose() 归还连接
 ├─ 打印每用户结果 + 统计
 └─ assert max_active <= 3     安全网断言

 ## 问题：为什么 LLM API 调用需要控制并发？使用 asyncio.Semaphore 有什么作用？
为什么 LLM API（以及大多数外部 API）需要控制并发
根本原因是外部服务方有限流，并发一上去就会出现两类问题：

1. 触发服务端的速率限制
绝大多数 LLM 厂商都设了两层阈值：

RPM（Requests Per Minute）：每分钟请求数
TPM（Tokens Per Minute）：每分钟处理的 token 数（TPM 通常更先撞线）
超出后返回 HTTP 429 Too Many Requests。Anthropic、OpenAI 等都明确在文档里写"超过 rate limit 会拒绝请求"。OpenAI 的 tier 1 账户大概也就 500 RPM、TPM 视模型而定——并发一上来就 429，然后你得退避重试，反而更慢。

2. 资源/成本失控
LLM 调用是贵且慢的操作（一次对话几百 ms 到几十秒）。如果不加限制：

账单爆炸：你本来只想问 10 个问题，结果 asyncio.gather(*[call() for _ in range(1000)]) 一发出去，1000 个请求同时跑，可能 
5变成500。
响应时间尾部变长：服务方在高压下也会变慢，看似"并行更快"实际上后 50% 的请求等更久。
下游依赖连锁反应：自建网关、向量库、数据库也会跟着被冲垮。
3. 网络与公平队列
大量并发连接还可能触碰单主机连接数上限、CDN 的公平队列等。这不是 LLM 独有的，是任何"远端 HTTP 服务"都有的。

一句话总结
LLM API 要控制并发，是因为外部服务有 RPM/TPM 限流、不控制会让成本和错误率双双失控；asyncio.Semaphore 提供的是协程级别的反压阀——"你可以 fan out 一千个任务，但同一时刻最多只有 N 个真正在飞"，既保住上限，又不浪费事件循环。


## Coroutin/Task/Event Loop
1. Coroutine（协程对象）——"待执行的代码蓝图"
用 async def 定义的是一个协程函数；调用它返回的才是协程对象。
关键：调用协程函数不会执行任何代码。fetch_user(...) 只是返回一个"承诺"，里面没有任何字节被跑过。
2. Task（任务）——"已经被排上日程的协程"
Task 是 asyncio.Future 的子类，把一个 coroutine 包装起来并交给事件循环调度。
创建方式：
task = asyncio.create_task(coro)   # 排到事件循环上，立即可被调度
# 或者
await asyncio.gather(coro1, coro2) # gather 内部会帮你 ensure_future
Task 跟踪三件事：
状态：pending / running / done / cancelled
结果：task.result() 拿返回值，抛过的异常会重新 raise
取消：task.cancel() 能在协程下一次被驱动时往里塞 CancelledError
3. Event Loop（事件循环）——"调度员 + 运行时装配车间"
一个单线程、合作式的调度器，由 asyncio.run(main()) 在后台创建并运行。
它内部维护两类队列：
ready queue：可以被立即执行的回调（包括新创建的 Task）
waiting queue / I/O 多路复用：等待 I/O / sleep / Future 完成的 Task
核心动作只有两件事在循环：
从 ready queue 取出一个 Task，调用它底层 coroutine 的 send(None)，驱动它往前跑；
跑到 await 一个还没就绪的 Future，Task 就被挂起到 waiting queue；事件循环转去处理 I/O 就绪事件或定时器，唤醒对应的 Task，再 send(value) 继续。

三者关系一句话："async def"写出 coroutine 函数；调用得到 coroutine 对象；create_task/gather 把它变成 Task；事件循环才是真正驱动 Task 一步步往前走的那只手。