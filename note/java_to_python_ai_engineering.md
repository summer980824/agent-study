# Java 工程师转 Python AI 工程：常见问题自测

> 每题先给结论，再展开关键差异点。

## Java 与 Python 的类型系统和开发体验差异。

结论：Python 是动态类型 + 鸭子类型，Java 是静态强类型。

- Python 的 `dict` / `list` 原生对应 JSON 的 object / array，配合 `json` 模块（`json.dumps` / `json.loads`）即可双向转换；Java 需要 Jackson / Gson 这类库。
- Python 函数参数不需要声明类型也能跑；类型提示（`def f(x: int) -> str`）是给编辑器和 `mypy` 用的，运行时不强制。
- `None` 对应 Java 的 `null`。Python 有 `Optional[X]` / `X | None` 类型提示，`mypy` 等静态检查工具**能**检查它，但运行时不会强制执行——忘记判空要靠类型检查或测试提前发现，否则到运行时才炸（`AttributeError` / `TypeError`）。
- 开发体验上：Python 写脚本快、REPL/notebook 迭代快；代价是重构依赖测试和类型检查兜底，而不是编译器。

## Maven/Gradle 与 uv 的依赖管理差异。

结论："解析更快、默认锁文件"，语义上更像 Cargo。

- uv 与 Maven 一样会解析**兼容的版本区间**，但它把解析结果写进 `uv.lock` 锁文件，`uv sync` 严格按锁文件安装——差异在**默认机制**：uv 锁文件是一等公民、开箱即用；Maven 也支持锁定（`dependencyManagement` 钉版本、`flatten-maven-plugin`、Enforcer 规则约束版本区间），只是要手工搭建，默认的 `mvn install` 按解析结果直接装。可复现性不是能力差距，是约定和成本的差距。
- 速度差距来自实现：uv 用 Rust 写的、全局缓存，解析和安装比 Maven/Gradle 快一两个数量级。
- 对应关系：`pom.xml` ≈ `pyproject.toml`，`mvn install` ≈ `uv sync`，本地仓库 `~/.m2` ≈ uv 的全局缓存。
- "旧版本兼容、build 或 runtime 报错"不是 Maven 的特性，是**语义化版本 + 依赖传递**带来的普遍问题，Python 生态同样有（所以才需要 lock 文件）。

## Spring Boot 配置管理与 Python Config 的对比。

结论：单看配置注入，pydantic-settings 只是"读环境变量 + 校验"的工具，和 Spring Boot 不是一个量级。

Spring Boot 提供的是一整套：

- `@ConfigurationProperties`：把配置文件绑定成强类型 Bean。
- profile（dev/test/prod）按环境切换配置。
- 配置中心（Spring Cloud Config / Nacos）集中管理多实例配置。
- Actuator 暴露配置快照、健康检查，配合 DI 容器，配置本身就是被管理的对象生命周期的一部分。

Python 侧的对应物要拼装：

- 配置加载与校验：`pydantic-settings`（本项目 `config.py` 用它读 `.env` / `.env.dev`，按 `APP_ENV` 切换）。
- 依赖注入：Python 没有内置 DI 容器，FastAPI 的 `Depends` 是函数级注入，量级远小于 Spring 容器；小项目直接"显式传参"（本项目 CLI → client 就是刻意不用容器）。
- 结论：Python AI 应用通常是无状态脚本/小服务，配置面小，环境变量 + pydantic 校验就够；Java 经验中"配置即契约、必须可校验可审计"的思路仍然值得带过来。

## Java 异常体系与 Python 异常处理的对比。

结论：Python 确实没有 checked exception，所有异常都像 RuntimeException 一样不强制捕获——但不止这一点。

- 对应关系：`try/except` ≈ `try/catch`，`raise` ≈ `throw`，`finally` 一样有；另外 Python 多了 `else`（无异常时执行）和 `with`（上下文管理，≈ try-with-resources）。
- Java 的编译期强制处理（checked exception）在 Python 里不存在，异常是否处理全靠约定和代码评审——所以 Python 项目更要靠测试覆盖错误分支。
- 风格差异：Python 惯用 **EAFP**（先做了再捕异常，如 `dict[key]` 失败捕 `KeyError`），Java 惯用 **LBYL**（先 `containsKey` 判断）。`StopIteration`、`ValueError` 等都靠 except 分流。
- 本项目 cli.py 的分层处理是典型例子：`ValidationError`（配置）→ `LLMResponseError`（业务校验）→ `httpx.HTTPError`（网络）→ 各自翻译成不同的退出码和用户提示，相当于 Java 里的全局 `@ExceptionHandler`。

## JUnit/Mockito 与 pytest/Mock 的对比。

结论：pytest 写起来更省样板，但 mock 思路一致。

- 断言：JUnit 5 用 `assertEquals(...)` 一堆方法；pytest 直接写 `assert x == y`，失败时自动拆解表达式给出上下文。
- 参数化：`@ParameterizedTest` ≈ `@pytest.mark.parametrize`，pytest 版本更轻。
- Mock：Mockito 的 `when/thenReturn` ≈ `unittest.mock` 的 `patch` / `MagicMock`；HTTP 层 Java 用 WireMock，Python 用 `respx`（本项目测试拦截 httpx 就靠它）。
- fixture：pytest 的 fixture（conftest.py 共享）比 JUnit 的 `@BeforeEach` 灵活得多，可以做作用域管理（session 级只建一次的 client）。
- 异步测试：pytest 加 `pytest-asyncio` 即可，JUnit 需要额外引擎。
- 值得保留的 Java 经验：测试金字塔、边界值用例设计、mock 只打边界不 mock 被测逻辑本身。

## 为什么 AI 应用开发更适合快速使用 Python 生态。

结论：整个 AI 技术栈就是长在 Python 上的。

- 主流框架和模型代码首发都是 Python：PyTorch、transformers、LangChain、LlamaIndex，文档和示例全是 Python。
- 数据形态匹配：AI 工作流本质是"张量 + JSON 消息 + notebook 迭代"，Python 的 dict/notebook 天然贴合，Java 的类型 ceremony 在这里是纯负担。
- 胶水能力：Python 是"第二好的语言"——数值计算底层是 C/C++/CUDA（numpy、torch），Python 只负责编排，所以慢的不是热点路径。
- 调试方式：AI 开发大量依赖"改提示词 → 跑一次看输出"，脚本化 + notebook 的反馈速度远超编译部署周期。
- 反过来看 Python 的短板，也要看场景：GIL 主要限制传统 CPython 里 **CPU 密集型的多线程**（这时要 multiprocessing 或把热点下沉到 C/CUDA）；而**异步 I/O 并发**（asyncio + httpx）不受 GIL 影响，几千个并发网络请求没有问题——调 API、做编排正好是 I/O 密集型场景。运行时才发现类型错误则靠类型检查工具和测试兜底。

## 哪些企业级能力仍然值得复用 Java 的工程经验。

结论：模型可以换，工程能力不会过时——Java 经验恰恰是 Python AI 项目的稀缺品。

- **错误处理与降级**：超时、重试（带退避和上限）、熔断、429 处理——AI API 是外部不可控服务，比传统 RPC 更需要这些。
- **可观测性**：结构化日志 + request_id 串联（本项目 logging_config 就是）、metrics、tracing。
- **配置与密钥管理**：环境分层、密钥不进代码库、配置可审计。
- **测试策略**：契约测试、mock 边界、CI 流水线，保证重构时不破坏行为。
- **并发与资源管理**：连接池复用（共享 httpx.Client）、信号量限流、异步编排（asyncio ≈ CompletableFuture）。
- **领域建模**：清晰的模块边界和单向依赖（CLI → client → API），Python 没有编译器兜底，更要靠架构约束。
- 一句话：Python 负责把 AI 能力快速串起来，Java 的经验保证它在生产环境里可靠。
