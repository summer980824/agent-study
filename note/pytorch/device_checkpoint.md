# Device 与 Checkpoint(设备放置与模型保存)

> 对应 notebook:`notebooks/` 训练循环与模型保存相关章节

## 1. 为什么模型和输入 Tensor 必须位于相同 Device?

### 根本原因:kernel 启动要求指针同源

PyTorch 的算子最终调用的是设备特定的 kernel:

- **CPU kernel**:接收 host 内存指针,由 CPU 线程执行
- **CUDA kernel**:接收显存(device memory)指针,由 GPU SM 执行,通过 PCIe/NVLink 访问 host 内存既不被允许、性能也不可接受

当 `Linear` 层的权重在 GPU、输入在 CPU 时,`matmul` 拿到两个属于不同内存空间的指针,无法选择任何一个 kernel 完成计算,只能报错:

```python
model = model.to("cuda")
x = torch.randn(4, 10)          # 在 CPU
model(x)
# RuntimeError: Expected all tensors to be on the same device,
# but found at least two devices, cuda:0 and cpu!
```

### 反直觉点:跨设备拷贝不是静默自动的

NumPy 风格的经验会让人觉得"框架会自动转换"。PyTorch 选择**显式报错**而不是隐式搬运,因为:

- 隐式搬运隐藏了性能陷阱:每次 forward 都同步拷贝 `H2D`,训练吞吐会灾难性下降,用户却看不到原因
- 显式报错让 device 放置成为代码中的一等公民,逼你在数据加载阶段(`DataLoader` 的 `pin_memory` + `non_blocking` 异步拷贝)就处理好

### 实操规则

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)                 # 模型参数搬走(就地修改 Module)

for x, y in dataloader:
    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    logits = model(x)                    # 双方同 device,才能计算
```

注意:`model.to(device)` 是 in-place 的(Module 内部替换参数张量),而 `tensor.to(device)` 返回新张量,必须重新赋值 —— 这是另一个常见 bug 来源。

## 2. state_dict 和整个模型对象有什么区别?

### 定义对比

| | `model.state_dict()` | 整个模型对象(pickle) |
|---|---|---|
| 内容 | 纯数据:`OrderedDict[str, Tensor]`,只有参数和 buffer | 完整 Python 对象:类定义引用 + 权重 + 超参数 + 训练状态字段 |
| 依赖 | 与代码结构解耦,加载时按 key 匹配 | 依赖**原始类定义可导入**(pickle 只存类路径,不存代码) |
| 安全性 | 只含张量,不可执行,加载相对安全 | 反序列化可执行任意代码,不可信 checkpoint 有安全风险 |
| 可读性/可移植性 | 可用 `torch.load` 查看、跨框架/跨代码版本灵活搬运 | 黑盒二进制,代码重构后无法加载 |
| 文件大小 | 只有权重 | 略大(多 pickle 开销),量级相同 |

### 保存/加载代码形态

```python
# state_dict 方式(推荐)
torch.save(model.state_dict(), "ckpt.pt")
model.load_state_dict(torch.load("ckpt.pt"))   # 需要先有模型实例

# 整体 pickle 方式
torch.save(model, "model.pt")
model = torch.load("model.pt")                  # 直接拿到对象,但要求类可导入
```

### 关键差异:结构和代码的耦合

`state_dict` 是**结构无关的数据快照** —— 保存时不需要知道模型怎么实现,加载时才要求 key 对得上。这使得:

- 改网络结构后可以做部分加载:`load_state_dict(sd, strict=False)`
- 微调场景可以取部分权重(如只加载 backbone)
- 跨代码版本迁移时,可以用脚本改写 state_dict 的 key

而 pickle 整个模型,本质上是把"指向你硬盘上某个类定义的引用"写进文件。重构一下目录(`git mv model.py net.py`),旧文件就永远加载不了了。

**工程结论**:权重用 `state_dict` 存;只有完全确定部署环境、追求"一行加载"的离线 demo 才考虑 pickle 整个模型。

## 3. 为什么恢复训练时需要保存 optimizer state?

### optimizer 内部维护了逐参数的动量缓存

以带动量的 SGD 为例,更新规则是:

$$v_{t+1} = \mu v_t + \nabla_\theta L, \quad \theta_{t+1} = \theta_t - \eta \, v_{t+1}$$

每步更新都依赖历史累积量 $v_t$。Adam 更复杂,每个参数维护**两个状态**:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t, \quad v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$$

这些 $m$、$v$ 都按参数形状存着,占用与模型同量级的显存/内存。

### 不保存会发生什么:隐式的学习率重置

只恢复模型权重、重新构造 optimizer 时,所有动量状态清零。效果相当于:

- **warmup 被打断**:训练初期本来靠小更新稳定方向,动量清零后几步内更新方向变得噪声主导
- **已收敛区域的"震荡回弹"**:SGD 的动量本身有类似加速的作用,清零后等效于突然降低了等效学习率;而 Adam 的 $v_t$ 清零意味着第二阶矩重新估计,每个参数的"有效步长"瞬间放大数倍,容易把模型从 loss 谷底弹出去

直观类比:模型权重是"位置",optimizer state 是"速度"。只存位置恢复,等于让全速运动的物体瞬间静止后重新随机起步。

### 什么时候可以不存

- **只推理、不继续训练** —— 梯度相关信息无用
- **全新 fine-tune,刻意要重置优化状态**(换优化器、换学习率策略时反而经常这么做)

### 保存与恢复

```python
ckpt = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),   # 含 momentum / Adam m、v
    "epoch": epoch,
    "lr_scheduler": scheduler.state_dict(),
}
torch.save(ckpt, "ckpt.pt")

# 恢复
model.load_state_dict(ckpt["model"])
optimizer.load_state_dict(ckpt["optimizer"])  # 参数形状必须一致,否则报错
```

注意:`load_state_dict` 要求 optimizer 关联的参数组与保存时**完全一致**(包括 param_groups 里的 lr、weight_decay 等配置也会被还原)。

## 4. 训练 checkpoint 通常需要保存什么?为什么只保存 model.state_dict() 有时不够?

### 完整 checkpoint 的清单

```python
ckpt = {
    # 核心:权重
    "model": model.state_dict(),

    # 恢复训练:优化状态
    "optimizer": optimizer.state_dict(),
    "lr_scheduler": scheduler.state_dict(),   # StepLR / CosineAnnealing 的内部步数

    # 训练进度与决策依据
    "epoch": epoch,
    "global_step": global_step,
    "best_val_loss": best_val_loss,           # early stopping / 模型选择依据
    "val_metric_history": [...],              # 画曲线、判断过拟合

    # 可复现性
    "rng_state": {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    },

    # 超参数快照(与权重配套,避免"权重找不回当时的配置")
    "config": {"lr": 1e-3, "batch_size": 32, "arch": "..."},
}
```

### 为什么只存 model.state_dict() 不够 —— 分场景

**场景一:中断后继续训练(最常见)**

缺 optimizer/scheduler state → 优化状态清零,等效扰动训练轨迹(见第 3 节);缺 epoch/step → 学习率调度错乱(如 Cosine 已衰减到 0.1x,重启后却按初始 lr 跑);缺 rng state → 数据增强、Dropout 的随机序列与之前重叠,实验不可复现。

**场景二:从 checkpoint 选最优模型**

只存权重,无法知道它是第几轮、验证集指标多少。`best_val_loss` 这类元数据是模型选择的依据,缺失时只能靠文件名约定(脆弱)或重新验证(昂贵)。

**场景三:排查与复现**

训练曲线、超参数快照看似冗余,但两周后想回答"这个模型当时 lr 是多少、第几轮 early stop 的"时,没有它们就只能翻日志碰运气。

### 反面约束:不是越多越好

- 大模型 Adam 状态下 optimizer state 体积可达权重的 **2 倍**($m$、$v$ 各一份),定期落盘 I/O 开销可观 —— 工程上常区分 `last.ckpt`(完整,用于断点续训)和 `best.ckpt`(可只存模型权重,用于部署/评测)
- rng state 里 `cuda.get_rng_state_all()` 在多卡时体积不小,只在对随机性敏感(对比实验)时才值得存
- 校验和/原子写入(`tmp` 文件 + rename)避免写到一半进程被杀产生半个 checkpoint

**一句话总结**:`model.state_dict()` 回答"模型现在长什么样",而训练 checkpoint 还要回答"怎么走到这里的" —— 前者只需权重,后者需要优化状态、进度、随机性与配置共同构成训练轨迹的完整切片。
