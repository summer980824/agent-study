# Attention PyTorch Bridge 实验笔记

对应实验：`notebooks/day48_attention_pytorch_bridge.ipynb`

核心实现：`agent_study/pytorch/attention.py`

## Q1: 这个实验要验证什么？

实验用三个 PyTorch 算子实现最简注意力机制：

```python
def simple_attention(query, key, value):
    scores = torch.matmul(query, key.transpose(-2, -1))
    weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(weights, value)
    return output, weights
```

对应公式：

$$\operatorname{Attention}(Q,K,V)=\operatorname{softmax}(QK^T)V$$

三步分别回答三个问题：

1. `Q @ K.T`：每个 Query 和每个 Key 有多匹配？
2. `softmax`：当前 Query 应该把多少注意力分配给每个位置？
3. `weights @ V`：按照这些权重，从每个位置取多少 Value 信息？

一句话概括：**Q 和 K 决定“看谁”，V 决定“拿到什么”。**

## Q2: 输入数据的形状和含义是什么？

实验构造了 3 个位置，每个位置都有一个 4 维 Query、一个 4 维 Key 和一个 2 维 Value：

```python
query = torch.tensor([
    [1., 0., 1., 0.],
    [0., 1., 0., 1.],
    [1., 1., 0., 0.],
])

key = torch.tensor([
    [1., 0., 0., 1.],
    [0., 1., 1., 0.],
    [1., 1., 0., 0.],
])

value = torch.tensor([
    [1., 0.],
    [0., 2.],
    [2., 1.],
])
```

形状如下：

```text
query: [seq_len, d_k] = [3, 4]
key:   [seq_len, d_k] = [3, 4]
value: [seq_len, d_v] = [3, 2]
```

这里 `d_k = 4`，`d_v = 2`。Query 和 Key 必须具有相同的特征维度，才能计算点积；Value 的维度可以不同，因为它决定最终输出的特征维度。

## Q3: 第一步如何计算注意力分数？

```python
scores = torch.matmul(query, key.transpose(-2, -1))
```

形状推导：

```text
[3, 4] @ [4, 3] -> [3, 3]
```

`key.transpose(-2, -1)` 把“每行一个 Key”改成“每列一个 Key”，于是一次矩阵乘法就能算出所有 Query 与所有 Key 的点积：

```python
scores[i, j] = query[i] @ key[j]
```

本实验得到：

```python
scores = tensor([
    [1., 1., 1.],
    [1., 1., 1.],
    [1., 1., 2.],
])
```

前两个 Query 与三个 Key 的点积都为 1。第三个 Query 与第三个 Key 完全相同，所以点积最大：

```text
[1, 1, 0, 0] · [1, 1, 0, 0] = 2
```

注意：点积大只表示在这个人为构造的例子中匹配程度高。真实模型中的 Q、K 由可学习投影产生，匹配语义通过训练形成，并不是人工规定的。

## Q4: 第二步如何把分数变成注意力权重？

```python
weights = torch.softmax(scores, dim=-1)
```

`dim=-1` 表示对每一行独立做 softmax，即每个 Query 都分别对所有 Key 归一化：

$$\operatorname{softmax}(s_{ij})=\frac{e^{s_{ij}}}{\sum_k e^{s_{ik}}}$$

得到：

```python
weights = tensor([
    [0.3333, 0.3333, 0.3333],
    [0.3333, 0.3333, 0.3333],
    [0.2119, 0.2119, 0.5761],
])
```

前两行分数相同，因此权重都是 $1/3$。第三行为：

$$\operatorname{softmax}([1,1,2])
=\left[\frac{e}{2e+e^2},\frac{e}{2e+e^2},\frac{e^2}{2e+e^2}\right]
\approx[0.2119,0.2119,0.5761]$$

因此，`weights[i, j]` 表示第 `i` 个 Query 对第 `j` 个 Value 的分配比例。每一行的权重之和都为 1，但不同位置之间不需要相等。

## Q5: 第三步如何得到输出？

```python
output = torch.matmul(weights, value)
```

形状推导：

```text
[3, 3] @ [3, 2] -> [3, 2]
```

输出的每一行都是所有 Value 的加权求和：

```python
output[i] = sum(weights[i, j] * value[j] for j in range(3))
```

前两个 Query 使用均匀权重，所以输出都是三个 Value 的平均值：

$$\frac{[1,0]+[0,2]+[2,1]}{3}=[1,1]$$

第三个 Query 更关注第三个位置：

$$0.2119[1,0]+0.2119[0,2]+0.5761[2,1]
\approx[1.3642,1.0]$$

最终输出：

```python
output = tensor([
    [1.0000, 1.0000],
    [1.0000, 1.0000],
    [1.3642, 1.0000],
])
```

这说明 Attention 不是直接选出某一个 Value，而是根据权重对所有 Value 做**可微的加权混合**。权重越集中，输出越接近某个位置的 Value；权重越均匀，输出越接近平均信息。

## Q6: 最后的断言验证了什么？

```python
assert weights.shape == (3, 3)
assert output.shape == (3, 2)
assert torch.allclose(weights.sum(dim=-1), torch.ones(3))
```

分别检查：

1. 每个 Query 对三个 Key 都有一个权重，所以权重形状是 `[3, 3]`。
2. 每个 Query 输出一个 2 维向量，所以输出形状是 `[3, 2]`。
3. 每个 Query 的权重之和为 1，验证 softmax 的归一化方向正确。

`torch.allclose` 用来容忍浮点计算误差，不应直接用 `==` 比较浮点计算结果。

## Q7: Transformer 为什么可以完全建立在这些基础算子之上？

严格来说，完整 Transformer 不只使用本实验的 `matmul`、`transpose` 和 `softmax`，还使用线性层、逐元素加法、归一化、激活函数、掩码、张量变形和 embedding 查表等基础算子。但这些算子已经足以组合出 Transformer 的核心网络，不需要为每种语言现象编写一条专门规则。

### 1. Transformer 的核心计算本来就能分解成基础张量运算

一个典型 Transformer Block 可以写成：

```text
输入 x
  ├─ LayerNorm
  ├─ 线性投影生成 Q、K、V
  ├─ Q @ K.T，计算所有位置之间的分数
  ├─ 缩放、mask、softmax，得到注意力权重
  ├─ weights @ V，完成跨位置的信息混合
  ├─ 输出投影和残差连接
  ├─ LayerNorm
  ├─ 前馈网络：线性层 -> 激活函数 -> 线性层
  └─ 残差连接
```

其中：

- **线性投影**：矩阵乘法加 bias，把同一个 token 表示映射成不同角色的 Q、K、V。
- **Attention**：点积、softmax 和加权求和，让 token 按内容交换信息。
- **前馈网络（FFN）**：矩阵乘法和非线性激活，对每个位置分别做特征变换。
- **残差连接**：张量加法，保留原始信息并改善深层网络的梯度传播。
- **LayerNorm**：均值、方差、除法和仿射变换，使数值范围更稳定。
- **多头机制**：`reshape`、`transpose` 和拼接，把同一计算拆到多个表示子空间并行执行。
- **位置表示**：位置向量加法，或 RoPE 中的三角函数和逐元素运算，为模型补充顺序信息。
- **mask**：布尔索引或加法掩码，把不允许关注的位置分数设为负无穷，例如防止 Decoder 看到未来 token。

复杂模型不是由一个神秘的“理解语言算子”构成，而是由许多简单、可组合的数学运算构成。

### 2. 矩阵乘法能同时表达投影、匹配和信息混合

Transformer 中最重要的三类计算都可以写成矩阵乘法：

```python
Q = x @ W_q.T                 # 特征投影
scores = Q @ K.T              # 所有 Query-Key 对的匹配
output = attention_weights @ V # 所有 Value 的加权求和
```

矩阵乘法既简洁，又适合硬件并行。GPU 的 Tensor Core 以及 cuBLAS 等底层库都对稠密矩阵乘法做了高度优化。一次算子调用可以完成大量乘加，避免用 Python 循环逐 token、逐位置计算。

因此，基础算子不仅“足够表达”Transformer，也让 Transformer 能高效运行。

### 3. 张量并行消除了训练时的逐步循环依赖

RNN 通常要先算完第 $t-1$ 个时刻，才能计算第 $t$ 个时刻。标准 Attention 则可以先一次性算出整个 `Q @ K.T`，让所有位置同时参与计算。

这使 Transformer 在训练时可以充分利用整段序列的并行性。自回归 Decoder 并不是不能并行训练，而是通过 causal mask 禁止每个位置读取未来信息，同时仍然一次算出所有位置的表示。

代价是标准 Attention 需要构造 $n\times n$ 的分数矩阵，计算量通常为 $O(n^2d)$，注意力矩阵显存开销为 $O(n^2)$（具体实现可通过分块等方式降低显存占用）。所以“高度并行”不等于“没有成本”。

### 4. 非线性和层叠让简单算子具有复杂表达能力

如果整个网络只有矩阵乘法和加法，多层线性变换仍可合并成一个线性变换。Transformer 的表达能力还来自：

- softmax 提供的、由输入动态决定的非线性权重；
- GELU 等前馈网络激活函数；
- 多层 Attention 与 FFN 的交替堆叠；
- 多头机制对不同关系进行并行建模。

Attention 负责**跨位置混合信息**，FFN 负责**位置内部的特征变换**。两者反复组合后，模型可以逐层构造越来越抽象的表示。

### 5. 所有核心算子都可微，参数可以通过训练自动学习

基础算子只规定计算结构，并不直接包含语言知识。真正的知识存储在线性投影、FFN、embedding 等参数中。

由于矩阵乘法、softmax、LayerNorm 和常见激活函数都可微，PyTorch autograd 可以从最终损失一路计算梯度，更新这些参数。训练过程会自动学到：

- 哪些特征应进入 Q、K、V；
- 什么模式应当相互匹配；
- 每层应该提取和保留什么信息；
- 不同头应该关注哪些关系。

mask 等操作本身不需要学习参数，但可以把梯度限制在允许的路径上。因此，**简单可微算子 + 大量可学习参数 + 训练数据**，共同产生了复杂行为。

### 6. “建立在基础算子之上”不等于“整个 AI 系统只有这些算子”

需要区分核心网络和完整系统：

- Transformer 核心前向与反向计算，可以由上述基础算子组合完成。
- 分词、数据清洗、采样、搜索、分布式通信和优化器等，还需要其他算法与工程组件。
- 基础算子本身也不是凭空工作，它们依赖 PyTorch、CUDA、GPU 和底层数值库。

所以准确结论是：**Transformer 把序列建模问题转化成了可微、可组合、适合并行硬件的基础张量运算；复杂智能来自这些运算的参数化组合和大规模训练，而不是来自某个单独的复杂算子。**

## Q8: 这个简化实验距离标准 Transformer Attention 还有哪些差别？

本实验为了突出核心数据流，省略了：

1. **缩放**：标准形式为 $\operatorname{softmax}(QK^T/\sqrt{d_k})V$，避免维度较大时点积过大，使 softmax 饱和、梯度变小。
2. **可学习投影**：真实模型使用 $W_Q,W_K,W_V$ 从输入生成 Q、K、V，而不是直接手工指定。
3. **多头注意力**：真实模型在多个表示子空间中并行计算注意力，再拼接并投影。
4. **mask**：根据任务屏蔽 padding 或未来位置。
5. **完整 Block**：还包括输出投影、残差连接、LayerNorm、FFN 等组件。

这个实验最重要的收获是：先理解 `Q @ K.T -> softmax -> @ V` 的数据流，再把缩放、投影、多头和 mask 逐层加回去，就能从简化版过渡到标准 Transformer。