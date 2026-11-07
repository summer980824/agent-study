# Autograd 
## 机制简介
Autograd（Automatic Differentiation，自动微分）是 PyTorch 的核心：你前向算一遍，它自动记住计算过程，反向一次就能求出所有偏导数。程序员不再需要手写链式法则。

三个关键词：
- **计算图（Computation Graph）**：前向计算时，PyTorch 悄悄记录"哪个张量由哪些运算得来"，形成一张有向图。`requires_grad=True` 的张量是图的**叶子节点**（通常是参数）。
- **前向传播（Forward）**：从输入算出输出（比如 loss），顺便建图。
- **反向传播（Backward）**：调用 `loss.backward()`，从输出往输入走一遍，把每个叶子的梯度算出来写进 `.grad`。


## gradient（梯度）
数学定义：梯度是各偏导数组成的向量 `∇f = [∂f/∂x₁, ∂f/∂x₂, ...]`。

物理意义：x.grad 表示 x 每增大一个单位，y 会增大多少（当前点上的瞬时变化率）；梯度的反方向是 loss 下降最快的方向，这就是梯度下降能工作的原理。

一个常见坑：grad 默认是**累加**的——多次调用 backward() 时梯度会相加，训练循环里通常每个迭代前要做 x.grad.zero_()（或 optimizer.zero_grad()）清零。

`no_grad()`：停止计算梯度——推理/评估时包一层，既不建图也不存梯度，省显存、提速：
```python
with torch.no_grad():
    y_pred = model(x)
```
另外 `torch.inference_mode()` 是更彻底的优化版（连版本计数器都不记），纯推理场景优先用它。

`detach()`：从计算图上"剪下来"，得到一张不追踪梯度的张量（常用于"拿数值不参与反向"的场景）。


## MSE Mean Squared Error
计算所有预测值与实际值的误差的平方，再算出平均值
loss = ((y_pred - y_true) ** 2).mean()

为什么用平方而不是绝对值？
- 平方处处可导，且导数是线性的（误差越大梯度越大），优化更平滑；
- 会"重罚"大误差，让模型优先消灭大偏差；
- 缺点是对离群点敏感。

```python
loss = ((y_pred - y_true) ** 2).mean()
loss.backward()          # 反向传播，梯度自动累加到 w.grad / b.grad
```


## 线性回归
给定目标数据集和训练参数，计算出预测数据集与目标数据集的各参数的loss梯度，然后通过梯度下降循环最终得到结果。

训练循环的基本骨架：
```python
# 参数（叶子节点，需要梯度）
w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

for epoch in range(100):
    # 1. 前向：预测 + loss
    y_pred = w * x + b
    loss = ((y_pred - y_true) ** 2).mean()

    # 2. 清零梯度（否则会累加）
    w.grad = None   # 等价于 w.grad.zero_()

    # 3. 反向：算出 ∂loss/∂w 和 ∂loss/∂b
    loss.backward()

    # 4. 梯度下降：沿负梯度方向迈一小步
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
```
真实项目中第 2、4 步由 `optimizer`（如 `torch.optim.SGD`）代劳，但原理完全一样。

手动验证梯度（链式法则）：对 `loss = mean((w·x + b - y)²)`，单个样本有 `∂loss/∂w = 2(y_pred - y_true)·x`，`∂loss/∂b = 2(y_pred - y_true)`。backward() 算的就是这些东西的均值——可以拿这个结果对照 `.grad` 检查理解是否正确。


## 反向传播（Backpropagation）到底在计算什么？

一句话：**反向传播计算的是 loss 对每一个参数的偏导数 `∂loss/∂参数`**，并且只用一次反向遍历就全部算完。

### 它不是什么
- 不是数值微分（numerical differentiation）：不是对参数做 ε 扰动再算差商——那样每算一个参数的梯度都要重跑一遍前向，参数量大时完全不可行。
- 也不是符号微分：不在纸上推导出解析公式，而是按"函数值 + 局部导数"的规则在图上机械地算。

它是**自动微分中的"反向模式"（reverse-mode AD）**：利用链式法则，把复合函数的求导拆成图上逐节点的小步骤。

### 逐节点看：局部梯度 × 上游梯度
计算图上的每个节点都只知道两件事：
1. 它的**局部梯度**：输出对自己输入的偏导数（比如 `*` 节点的局部导数就是乘数本身）。
2. 从 loss 传到自己输出端的**上游梯度**。

节点的输入收到的梯度 = 局部梯度 × 上游梯度。这正是一条链式法则的应用：
```
∂loss/∂x = (∂loss/∂y) · (∂y/∂x)
   上游        局部梯度
```

### 为什么反向一次就够了
链式法则在**多变量**下还有一个事实：如果某中间变量 `z` 同时影响多条通往 loss 的路径，那么 `∂loss/∂z` 等于**各条路径的贡献之和**。反向遍历时每经过一个节点就把贡献累加进去（这也是 `.grad` 累加语义的来源），所以从 loss 倒着走一遍，沿途每个节点的梯度都齐了。

### 例子
```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x + 1     # 计算图：x → x², 3x → 相加 → y
loss = y                   # 这里 y 就是 loss

loss.backward()
print(x.grad)              # tensor(7.)
```
反向过程：`∂y/∂y = 1` 出发，经过加法节点（导数恒为 1，原样传递），到 `3x` 节点得到 3，到 `x²` 节点得到 `2x = 4`，两条路径的贡献相加 `3 + 4 = 7`，写入 `x.grad`。

### 为什么它快
反向传播的计算量和前向传播是**同一量级**（大约 2~3 倍前向开销），却能给出**所有**参数的梯度。对动辄百万、十亿参数的神经网络，这是唯一可行的方案——前向算 loss，反向一次拿全部梯度，然后优化器用它更新参数，如此循环。
