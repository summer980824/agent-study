# Weekly Review: D10-D13

## 1. 本周学会了什么

- Environment-based configuration
- HTTP / REST API
- Synchronous API Client
- Async HTTP requests

## 2. 本周完成的工程能力

- 配置与业务代码分离
- HTTP 请求与数据校验分离
- 统一处理常见网络异常
- 使用 AsyncClient 并发调用 API

## 3. 发现的问题

- asyncio.gather 的异常传播行为
- 模块导入时的网络请求副作用
- HTTP Client 的生命周期管理

## 4. 下周重点

- 控制并发请求数量
- 建立统一的结构化日志
- 使用 pytest 和 Mock 测试外部 API 调用

## 5. 面试问题

问题：一个 Python AI 服务需要同时调用模型 API、向量检索 API 和业务 REST API，你会如何设计调用层？
1. 传输层：每个上游一个 httpx.AsyncClient
不要三个上游共享同一个客户端。它们的 base_url、超时、headers、连接池大小都应该不同。
2. 韧性层：用装饰器组合，而不是塞进业务代码
模型 API：retry 在 429/5xx/超时，不重试 4xx（除了 408/429）。
向量 API：retry + 客户端并发信号量（asyncio.Semaphore(N)）控制 embedding 批大小。
业务 API：
HTTP 语义层：PUT 定义为幂等（多次执行效果一致），POST 默认不保证；
应用语义层：即使 PUT，如果服务端实现成"自增计数"也未必幂等；
5xx 不能直接判定"服务端未执行"：网络抖动、服务端崩溃都可能让请求已处理但响应丢失；
真正的判定标准：服务端是否对同一请求做了去重（基于 Idempotency-Key、请求指纹，或操作自然幂等）。
3. 适配器层：每个上游一个 Protocol + 实现
ModelClient 的 stream 返回 AsyncIterator，不是 await——这样上层能在 token到达时就往用户推送。
VectorClient 同时暴露 embed 和 search，因为检索前要先 embedding，组合成一个原子操作更易控。
BusinessClient 用 pydantic 校验响应，422/字段错误直接抛业务异常而不是原始 HTTP 异常。
4. 编排层：只做组合，不做 I/O 细节
5. 配置：在 config.py 里给每个上游独立分组，而不是一个扁平字典

容易踩的坑
1. 没有 timeout——任何一个上游抽风都会拖死整个请求。httpx.AsyncClient 不设 timeout 默认会在 5 秒网络无活动后触发超时，必须显式给。
2. 重试非幂等写——业务 API 的 POST 没有 idempotency key 时只重试网络层错误，不重试 5xx。
3. Embedding 没用信号量——向量服务 QPS 限制很严，一波并发上去直接 429，要用 asyncio.Semaphore 限并发。
4. 每个请求 new 一个 AsyncClient——TCP/TLS 握手和连接池失效，吞吐和延迟都崩。
5. 混用 sync/async——FastAPI 这种框架本身是 async，阻塞调用会卡死事件循环；如果非要 sync，用 asyncio.to_thread 隔离。
6. 错误吞掉——把 httpx.HTTPStatusError 翻译成业务异常时带上 status、body、request_id，否则线上排障只能看 log。
7. 观测缺失——给每个请求加 X-Request-ID（自己生成，向上游传），结构化日志里带上 upstream、latency_ms、status、tokens。