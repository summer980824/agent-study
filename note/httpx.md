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
- 重试策略(429、退避)

### 调用方负责 —— 业务语义错误

- 200响应成功
- 用户不存在 vs 网络错误该给用户看什么提示

## `Response.raise_for_status()`

- 状态码在 `200~299`:什么都不做
- 其他状态码:抛 `httpx.HTTPStatusError`
