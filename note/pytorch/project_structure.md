# PyTorch 项目结构：从 Notebook 到可复用模块（Day47）

对应 notebook：`notebooks/day47_pytorch_project_structure.ipynb`
前置：`notebooks/day46_mini_neural_network.ipynb`（所有代码写在一个 notebook 里）

Day47 把 Day46 的实验拆进了 `agent_study/pytorch/` 包，notebook 只负责"调用 + 展示结果"。本文记录这么拆的原因和背后的工程约定。

## 1. 为什么把 data、model、train 分开？三个模块各负责什么

```
agent_study/pytorch/
├── data.py    # build_toy_dataset()：构造 vocab、X_train/y_train、X_test/y_test
├── model.py   # TinyTextClassifier：Embedding → mean pooling → Linear
└── train.py   # train_model()：训练循环，返回 loss_history
```

分开的根本原因：**这三样东西的变化频率和复用场景完全不同**。

| 模块 | 职责 | 什么时候会改它 |
|---|---|---|
| `data.py` | 数据从哪来、长什么样（词表、张量、标签） | 换数据集、改预处理、加 batching |
| `model.py` | 网络结构：输入怎么变成 logits | 换结构（加隐藏层、换 Attention）、调超参 |
| `train.py` | 训练过程：loss、optimizer、循环、记录 history | 换优化器、加 eval/early stopping、加日志 |

如果三者混在一起（像 Day46 那样写在一个 notebook cell 序列里）：

- 想**只换模型**对比实验时，得复制整段代码，改一处、其余原样拷贝——几轮之后没人说得清两份代码差在哪。
- 想**换个数据**试同一个模型，同样要整段复制。
- 训练循环里的 bug 会散落在每个副本里，修一处漏一处。

拆开之后形成清晰的单向依赖：`train` 依赖"任意 model + 任意 data"的接口（`model(X)`、`loss.backward()`），不关心具体是文本分类还是回归；`model` 不关心数据从哪来；`data` 不关心谁来消费。**实验 = 三个模块的一种组合**，对比实验只改一个变量，这正是"做实验"和"抄代码"的区别。

这也对应 Java 里熟悉的原则：单一职责 + 面向接口编程。`train_model(model, X, y)` 的签名就是一个隐式接口——任何实现了 `forward` 的 `nn.Module` 都能传进来。

## 2. Python import 路径如何组织

本项目里 import 能工作，靠三层机制：

```
项目根目录 d:\workspace\agent-study\
├── pyproject.toml          # name = "agent-study"，[tool.uv] package = true
├── agent_study/            # 包目录（与项目同名，下划线替代连字符）
│   ├── __init__.py         # 有这个文件，agent_study 才是 Python 包
│   └── pytorch/
│       ├── __init__.py
│       ├── data.py         # → agent_study.pytorch.data
│       ├── model.py        # → agent_study.pytorch.model
│       └── train.py        # → agent_study.pytorch.train
├── notebooks/              # 实验笔记本，import agent_study.xxx
└── tests/                  # pytest，同样 import agent_study.xxx
```

1. **包 = 目录 + `__init__.py`**。`agent_study/pytorch/model.py` 这个文件路径，对应 import 路径 `agent_study.pytorch.model`。`__init__.py` 可以是空文件，它的存在标记"这是一个包"。
2. **可编辑安装（editable install）**。`pyproject.toml` 里 `[tool.uv] package = true` + hatchling 构建后端，使项目以 editable 模式装进 `.venv`：site-packages 里只放一个指向源码目录的链接，**改源码立即生效，不用重装**。所以从任何目录（包括 `notebooks/`）启动 Python 都能 `import agent_study`。
3. **pytest 的 `pythonpath = ["."]`**。`pyproject.toml` 的 `[tool.pytest.ini_options]` 把项目根目录加进 `sys.path`，保证即使不走安装，测试也能 import 到包。

类比 Java：包目录 ≈ `src/main/java/com/example/...` 的包结构；`__init__.py` ≈ 标记源码根的约定；editable install ≈ 本地 `mvn install` 后其他模块依赖 SNAPSHOT——改代码即生效，无需重新打包。

关键规则：**import 的是"安装进环境的包"，不是"相对于当前文件的目录"**。notebook 能 `from agent_study.pytorch.data import build_toy_dataset`，不是因为 notebook 和包有相对位置关系，而是因为包已装进 venv。

## 3. Notebook 与可复用 Python 模块的职责区别

| | Notebook | Python 模块（.py） |
|---|---|---|
| 职责 | **探索与叙事**：试想法、看中间结果、画图、写结论 | **沉淀与复用**：被多处调用的稳定逻辑 |
| 执行方式 | 单元格可乱序、重复执行，状态留在 kernel 内存里 | `import` 时自上而下执行一次，无隐藏状态 |
| 可测试性 | 几乎无法被 pytest 引用 | 可以被测试、被其他模块 import |
| 版本控制 | JSON 格式，diff 里混着输出和执行计数，难 review | 纯文本，diff 干净 |
| 复用方式 | 复制粘贴 cell | `import` |

实践中的分工（Day47 就是范例）：

- **模块负责"怎么做"**：`build_toy_dataset` / `TinyTextClassifier` / `train_model` 这些确定要反复用的逻辑，写进 `agent_study/pytorch/`。
- **Notebook 负责"做什么实验、结果如何"**：选择参数、调用模块、打印 shape、画 loss 曲线、写 markdown 分析。day47 的 notebook 里没有任何一段逻辑是"只能在这里跑"的。

判断标准：**一段代码第二次被需要时，就应该从 notebook 搬进模块**。留在 notebook 里的应该是"这一次实验特有的"东西——参数取值、可视化、结论文案。

## 4. 为什么研究代码也需要工程化？直接把实验都写在 Notebook 里不行吗

短期看不行，长期看是必需。notebook-only 的实验方式有几个必然暴露的问题：

1. **状态不可复现**。Notebook 的 kernel 内存保留了所有历史执行结果：先跑 cell 5 再跑 cell 2，变量可能是旧值。"在我机器上从头 Run All 能跑通"经常不成立。模块 + 函数没有隐藏状态，每次调用行为一致（给定同样的 seed）。
2. **无法对比实验**。研究的核心动作是控制变量："同一个模型换数据""同一份数据换结构"。逻辑散在 cell 里时，每次对比都靠复制粘贴，实验间的真实差异很快被淹没。拆成模块后，对比实验 = 换 import / 换参数，差异清晰可见（Day46 vs Day47 的 loss 曲线逐位一致，正是可复现性的验证）。
3. **bug 无法系统性修复**。如果训练循环被复制在 5 个 notebook 里，要修 5 处且必然有遗漏。在模块里只修一处，所有调用方自动受益。
4. **无法测试和重构**。没有模块就没有 `tests/`，改一行代码不知道弄坏了什么。`train_model` 进了模块，就可以写测试锁定行为（loss 单调下降、返回长度等于 epochs），之后重构才有底气。
5. **无法被下游复用**。研究最终要走向应用：CLI、API 服务、其他脚本。它们没法 import 一个 notebook，但可以 import 模块。工程化不是研究结束后的"收尾工作"，而是研究成果能被使用的前提。

一句话总结：**notebook 是草稿纸，模块是正式文档**。草稿纸上推导可以乱，但结论必须誊写进正式文档——否则三天后的自己和协作者都读不懂、用不了、改不动。工程化不是额外的负担，是把"一次性实验"变成"可积累资产"的唯一方式。
