# pytest 学习笔记

以 `tests/test_async_client.py` 为蓝本,总结这套测试用到的 pytest 用法。

## 运行测试

```bash
# 全跑
.venv/Scripts/python.exe -m pytest tests/test_async_client.py -v

# 跑某一个测试类
.venv/Scripts/python.exe -m pytest tests/test_async_client.py::TestAsyncGitHubClientRetry -v

# 跑某一个用例(支持子串匹配)
.venv/Scripts/python.exe -m pytest tests/test_async_client.py -v -k "retry_after"

# 首次失败即停
.venv/Scripts/python.exe -m pytest tests/test_async_client.py -x
```

`-v` 列出每个用例的完整 node id,排查失败时必备。

## 配置:`pyproject.toml`

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
```

这行只对 `pytest-asyncio` 有意义:`auto` 模式下,**所有 `async def test_*` 函数自动被视为异步测试**,不需要手动加 `@pytest.mark.asyncio`。

- `auto`:异步测试全自动跑,同步测试也照常
- `strict`:每个异步测试都要显式 `@pytest.mark.asyncio`,否则会被跳过/警告
- 选 `auto` 省事;选 `strict` 是为了让"哪些测试是异步的"在代码里一目了然

## 测试发现规则

- 文件名 `test_*.py` 或 `*_test.py`
- 类名 `Test*`(无 `__init__`)
- 函数/方法名 `test_*`
- 这套测试里用类分组:每个 `Test*` 类对应一个被测对象/场景,方法名描述场景而不是断言

```python
class TestAsyncGitHubClientRetry:
    async def test_no_retry_when_max_retries_zero(self): ...
    async def test_retry_then_success(self, monkeypatch): ...
```

## HTTP 桩:respx

`respx.mock` 通过 monkey-patch `httpx` 的 transport 层,让所有 `httpx.AsyncClient().get(...)` 走本地桩,不再发真实请求。

### 基本写法

```python
import respx
import httpx

@respx.mock                            # 装饰器形式:函数作用域内自动激活/关闭
async def test_x():
    respx.get("https://api.github.com/users/octocat").mock(
        return_value=httpx.Response(200, json={"login": "octocat"})
    )
    async with AsyncGitHubClient() as client:
        data = await client.get_json("/users/octocat")
```

`@respx.mock` 不带括号即装饰器形式;带括号 `respx.mock(...)` 可作上下文管理器或工厂,接受 `assert_all_called` 等参数。

### 多个连续响应用 `side_effect`

```python
respx.get(url).mock(
    side_effect=[
        httpx.Response(429, json={}),   # 第 1 次
        httpx.Response(429, json={}),   # 第 2 次
        httpx.Response(200, json={"ok": True}),  # 第 3 次
    ]
)
```

列表按调用顺序逐次返回;respx 在列表耗尽后再调用会抛 `StopIteration`,所以列表长度要和测试预期次数对得上。

### 抛异常

```python
respx.get(url).mock(side_effect=httpx.TimeoutException("timed out"))
respx.get(url).mock(side_effect=httpx.ConnectError("boom"))
```

### 检查调用记录

```python
route = respx.get(url).mock(return_value=...)

# ...跑测试...

assert route.called
assert route.call_count == 3
request = route.calls[0].request
assert request.headers["accept"] == "application/vnd.github+json"
assert request.url.path == "/users/octocat"
```

`route.calls` 是按调用顺序排列的 `CallList`,每条 `.request` / `.response` 都能拿到。

## 异步测试:pytest-asyncio

```python
import pytest

@respx.mock
async def test_something():
    respx.get("...").mock(return_value=httpx.Response(200, json={}))
    async with AsyncGitHubClient() as client:
        result = await client.get_json("/...")
```

- 因为 `asyncio_mode = "auto"`,上面没有 `@pytest.mark.asyncio` 也能跑
- `async def` 测试函数体里直接 `await`,不用 `asyncio.run()` 包裹
- 异步 fixture(`@pytest_asyncio.fixture`)在 auto 模式下也能直接用

## 断言异常

```python
with pytest.raises(AsyncGitHubNotFoundError) as exc:
    await client.get_json("/missing")

# exc.value 是抛出的异常实例
assert exc.value.path == "/missing"
assert "404" in str(exc.value)

# 用 match 校验消息(支持 regex)
with pytest.raises(AsyncGitHubAPIError, match="Expected a JSON object"):
    await client.get_json("/x")

# 同时校验类型和消息
with pytest.raises(AsyncGitHubAPIError) as exc:
    ...
assert not isinstance(exc.value, AsyncGitHubRateLimitError)
```

`pytest.raises` 既能在同步测试里用,也能在 `async with` 里用——只要它包住真正会抛异常的代码。

## monkeypatch fixture

`monkeypatch` 是函数级 fixture,自动在测试结束时还原所有改动——不用自己写 `try / finally` 恢复。

```python
async def test_x(monkeypatch):
    sleeps = []

    async def fake_sleep(delay):
        sleeps.append(delay)

    # 字符串路径:补丁 "agent_study.async_client" 模块下的 asyncio.sleep 属性
    monkeypatch.setattr(
        "agent_study.async_client.asyncio.sleep", fake_sleep
    )

    # ... 测试体里 asyncio.sleep 都会被 fake_sleep 替换 ...
```

为什么用字符串路径?`agent_study.async_client` 里写的是 `import asyncio` 然后 `await asyncio.sleep(delay)`——`asyncio.sleep` 是查 `asyncio` 模块上的属性。字符串路径让 pytest 解析路径后定位到模块对象,等价于 `monkeypatch.setattr(asyncio, "sleep", fake_sleep)`,teardown 时自动还原。

也可以直接 patch 普通属性/字典条目/环境变量:

```python
monkeypatch.setattr(obj, "attr", value)
monkeypatch.setitem(d, "key", value)
monkeypatch.setenv("VAR", "value")
monkeypatch.delattr(obj, "attr")
```

## 异步测试里的"假 sleep"

退避重试测试需要:

1. **快速**——不希望真睡 60 秒
2. **可断言**——验证实际计算的 sleep 时长

套路 = `monkeypatch` + 列表记录:

```python
def _install_fake_sleep(monkeypatch):
    sleeps = []
    async def fake_sleep(d):
        sleeps.append(d)
    monkeypatch.setattr("agent_study.async_client.asyncio.sleep", fake_sleep)
    return sleeps

async def test_backoff(monkeypatch):
    sleeps = _install_fake_sleep(monkeypatch)
    # ...触发重试...
    assert sleeps == [1.0, 2.0]   # 指数退避 attempt 0/1
```

## frozen dataclass 怎么测

```python
@dataclass(frozen=True)
class RateLimitInfo:
    limit: int | None
    ...

info = RateLimitInfo(60, 0, None, 60)
with pytest.raises((AttributeError, Exception)):
    info.limit = 10
```

frozen dataclass 改字段抛 `dataclasses.FrozenInstanceError`,它是 `AttributeError` 的子类。用元组 `(AttributeError, Exception)` 兼容两种风格,避免未来底层异常类变化时测试无故挂掉。

## 测试组织的小经验

| 该做 | 不该做 |
| --- | --- |
| 一个 `Test*` 类对应一类行为/对象 | 把所有用例堆在一个 `TestEverything` 里 |
| 方法名描述场景(`test_404_raises_not_found_with_path`) | 方法名描述断言(`test_assert_path_is_missing`) |
| 用 `match=` 校验错误消息片段 | 把整段消息硬编码到断言里 |
| 用 `route.call_count` 验证调用次数 | 维护一个全局计数器自己 `+= 1` |
| 桩只覆盖被测行为 | 桩顺手把无关依赖也一起 mock 掉 |

## 延伸:为什么测试 LLM 调用要 Mock

LLM 调用(OpenAI / Anthropic / 自建推理服务等)和测试 `AsyncGitHubClient` 一样,本质都是"发 HTTP 请求 → 等响应 → 解析结果",所以桩 HTTP 那一套(respx / `pytest-httpx` / `httpx.MockTransport`)完全适用。但 LLM 调用还有几条**特别需要 Mock**的理由。

### 1. 非确定性

LLM 默认带温度采样,同一段 prompt 每次可能给出不同输出。

```python
# 没 Mock —— 这次断言通过、下次可能挂
response = await client.chat([{"role": "user", "content": "1+1=?"}])
assert response == "2"
```

桩掉之后,响应可以是固定的字符串、固定的 JSON、甚至用 `side_effect` 模拟"第一条拒答、第二条回答对"这种场景。

### 2. 成本

主流 API 都是按 token 计费。一个测试套件跑几百次,一天下来 CI 跑几十轮,真金白银就没了——还把上游配额用光。Mock 让测试**零成本**。

### 3. 延迟

LLM 推理通常 200ms ~ 10s+。没有 Mock 的测试套件几分钟起步,反馈循环变得很慢;Mock 后和 `AsyncGitHubClient` 这套一样毫秒级。

### 4. 可用性 / 稳定性

- 上游限流(429)、临时宕机、断网 → 测试 flaky,排查时容易以为是自己的 bug
- 离线 / 飞机模式下开发 → 跑不了任何测试
- 跨地区网络抖动 → CI 偶发挂

Mock 让测试**完全离线、零网络依赖**。

### 5. 密钥 / 凭据

LLM 调用要 API key,不能进 git。CI 里通常用 secret,但:
- 本地开发可能没配
- 跑 PR 的 fork 仓库可能没权限读 secret
- 密钥轮换期间测试挂掉

Mock 让测试**不依赖任何密钥**。

### 6. 复现 bug

线上用户报了一个具体 case(比如某 prompt 让模型输出格式坏掉),要写一个回归测试。如果不 Mock,每次跑出来的内容都不一样,**根本写不出能复现这个 bug 的测试**。Mock 让你把"当时模型实际返回的那段字符串"原样保存成 fixture,反复回放。

### 7. 隐私 / 数据合规

把内部业务数据 / 用户输入发到上游 API 才能拿到响应,**测试阶段不该这么做**。Mock 让响应可控,数据不出本机。

### 怎么 Mock LLM 调用

最常见的两种姿势:

**a. 桩 HTTP 层(respx / `pytest-httpx`)** —— 适合官方 SDK,因为它们底层都是 `httpx`

```python
@respx.mock
async def test_chat_completion():
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "你好"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            },
        )
    )
    client = AsyncOpenAI()
    resp = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "say hi"}],
    )
    assert resp.choices[0].message.content == "你好"
```

这和 `async_client.py` 测试用的是**完全同一套机制**——只是 URL 换成了 LLM 厂商的 endpoint。

**b. 桩 SDK 的 client 对象** —— 适合不想处理 HTTP 细节,直接当单元测试写

```python
from unittest.mock import AsyncMock

def test_summarize():
    fake_client = AsyncMock()
    fake_client.chat.completions.create.return_value = make_response("摘要...")
    result = summarize(fake_client, text="...")
    assert result == "摘要..."
```

这种写法快,但要注意:**桩对象只能保证"调用了正确的 API",不保证"传对了参数 / 拼对了 URL"**——前者由单元测试覆盖,后者通常需要少量真实集成测试兜底(只在 CI 关键路径上跑,带密钥、本地跳过)。

### 不要 Mock 的地方

有些场景反而**应该**打到真实 LLM:

- **端到端 / 烟雾测试**:定期跑,确认密钥、模型版本、SDK 都还正常工作
- **评估(eval)**:用真实模型跑一组 benchmark,衡量 prompt/模型变更带来的效果差异——这就不是单元测试了,是另一类工具(`promptfoo` / `inspect` / 自建脚本)

这两类一般放慢管道、限制频率、和单元测试分开跑。

## 小结:这套测试用到的东西

- `pyproject.toml` 里的 `asyncio_mode = "auto"`
- `@respx.mock` 装饰器 + `respx.get(...).mock(return_value=...)`
- `respx.get(...).mock(side_effect=[...])` 做多次响应序列
- `respx.get(...).mock(side_effect=httpx.TimeoutException(...))` 触发传输层异常
- `pytest.raises(..., match=...)`, `pytest.raises(...) as exc`
- `monkeypatch.setattr` 替换 `asyncio.sleep` 加速重试测试
- `route.call_count` / `route.calls[i].request.headers` 校验调用细节

只要这几块组合,异步 HTTP 客户端的测试基本能覆盖全。