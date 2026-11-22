## Q1: 什么是 Query、Key、Value？

三者都来自同一条输入序列，是同一个 token 向量经过**三个不同的可学习线性投影**得到的三个角色不同的向量：

```python
# 输入 token 表示: [seq_len, d_model]
W_q = nn.Linear(d_model, d_model)  # Query 投影
W_k = nn.Linear(d_model, d_model)  # Key 投影
W_v = nn.Linear(d_model, d_model)  # Value 投影

Q = W_q(x)  # [seq_len, d_model] —— "我在找什么"
K = W_k(x)  # [seq_len, d_model] —— "我有什么特征可被检索"
V = W_v(x)  # [seq_len, d_model] —— "如果选中我，实际提供什么信息"
```

用**信息检索**类比最直观：

| 角色 | 类比 | 在注意力中的语义 |
|---|---|---|
| Query | 搜索框里输入的关键词 | 当前 token 发出的"提问"：我需要关注什么样的信息？ |
| Key | 每个文档的索引/标签 | 每个 token 挂出的"招牌"：我是什么类型的内容，可以回答什么问题？ |
| Value | 文档的正文内容 | 每个 token 携带的"实际信息"：一旦匹配上了，就把这份内容传递过去 |

注意力机制的完整流程：

1. 每个 token 同时产生 Q、K、V 三个向量。
2. 用我的 **Q** 去和所有 token 的 **K** 做点积，衡量"我的提问和你的招牌有多匹配"——这就是注意力分数。
3. 分数经过 softmax 变成权重（加起来等于 1）。
4. 用权重对所有 token 的 **V** 做加权求和，得到新的表示。

关键直觉：**Q 和 K 决定"看谁"，V 决定"拿到什么"。** 同一个词 "bank"，在 "river bank" 和 "bank account" 里会产生不同的 Q，从而注意到不同的上下文词，得到不同的含义。

## Q2: 矩阵转置 K.T 的作用？

`K.T`（即 `K.transpose(-2, -1)`）把 Key 矩阵从 `[seq_len, d_k]` 转置成 `[d_k, seq_len]`，目的是让 `Q @ K.T` 能算出**所有 token 两两之间的注意力分数**。

```python
Q = torch.randn(seq_len, d_k)
K = torch.randn(seq_len, d_k)

scores = Q @ K.T          # [seq_len, d_k] @ [d_k, seq_len] = [seq_len, seq_len]
scores = scores / math.sqrt(d_k)   # 缩放，防止点积随 d_k 增大而过大导致 softmax 饱和
```

`scores[i, j]` 就是第 `i` 个 token 的 Query 与第 `j` 个 token 的 Key 的点积，即"token i 对 token j 的关注程度"。

**为什么要转置？** 因为矩阵乘法的匹配规则是"前一个矩阵的列数 = 后一个矩阵的行数"：

- Q 的每一行是一个 token 的 Query 向量，shape `[seq_len, d_k]`。
- K 的自然形状也是 `[seq_len, d_k]`（每行一个 token 的 Key 向量）。
- 直接 `Q @ K` 维度对不上（`d_k ≠ seq_len`），而且即使维度凑巧相等，矩阵乘法算的也是"Q 的行空间组合"，不是我们要的"每个 Q 向量和每个 K 向量分别点积"。
- 转置后 K.T 的**每一列**是一个 token 的 Key 向量，`Q @ K.T` 的结果 `[i, j]` 恰好是 Q 的第 i 行与 K.T 的第 j 列——也就是第 i 个 Q 和第 j 个 K 的点积。

一句话：**转置把 K 从"每行一个 Key"变成"每列一个 Key"，使得一次矩阵乘法同时算出所有 (Query, Key) 对的点积。** 转置本身不复制数据（只是一个 stride 视图），计算开销几乎为零。

实际 batch 场景下的完整形状推导：

```python
# [batch, seq_len, d_model] @ [d_model, d_model] → [batch, seq_len, d_model]
Q, K, V = (proj(x) for proj in (W_q, W_k, W_v))

scores = Q @ K.transpose(-2, -1)   # [batch, seq_len, d_k] @ [batch, d_k, seq_len]
                                   # → [batch, seq_len, seq_len]
attn = softmax(scores, dim=-1)
out = attn @ V                     # [batch, seq_len, seq_len] @ [batch, seq_len, d_v]
                                   # → [batch, seq_len, d_v]
```

注意 `K.T` 只对最后两维转置，`batch` 维保持不动。

## Q3: 为什么 Transformer Attention 本质上大量使用矩阵乘法？

因为注意力的每一步，从数学定义上就是向量的**点积**和**加权求和**，而这两种运算恰好都是矩阵乘法：

**1. 线性投影（生成 Q/K/V）本身就是矩阵乘法**

```python
Q = x @ W_q.T  # [seq_len, d_model] @ [d_model, d_model]
```

每个 token 的向量左乘一个权重矩阵，天然就是矩阵乘。

**2. 计算注意力分数 = 一堆点积 = 一次矩阵乘法**

Q1 中已经看到，`Q @ K.T` 一个操作就算出了 `seq_len × seq_len` 个点积。如果不用矩阵乘法，就要写双重循环：

```python
# 矩阵乘法（实际写法，一次完成）
scores = Q @ K.T

# 等价的双重循环（仅用于理解语义，实际不要这么写）
scores = torch.zeros(seq_len, seq_len)
for i in range(seq_len):
    for j in range(seq_len):
        scores[i, j] = Q[i] @ K[j]   # 每次循环 = 一次点积
```

矩阵乘法把这 `O(n²)` 次点积合并成一个算子。

**3. 加权求和 V 也是矩阵乘法**

softmax 后的注意力矩阵 `[seq_len, seq_len]` 左乘 V，每一行是"对所有 V 向量的加权求和"——这正是矩阵乘法的行 × 列定义。

**4. 矩阵乘法是 GPU 上最成熟、最快的算子**

这是从"数学上方便"到"工程上必须"的关键一步：

- GPU 的硬件（Tensor Core）就是为大规模矩阵乘法优化的，吞吐远高于一般逐元素操作。
- 稠密矩阵乘法计算密度高（每次访存能做大量乘加），能充分利用内存带宽。
- cuBLAS、cuDNN 等库对矩阵乘法做了极致优化，PyTorch 的 `@`、`matmul`、`einsum` 底层都走这些库。
- 反向传播时，矩阵乘法的梯度（对 `A @ B` 求导仍是矩阵乘法）同样高效。

**总结：** 注意力的定义（投影 → 点积 → 加权求和）在数学上完全由矩阵乘法构成；写成矩阵形式后，原本需要循环的 `O(n²)` 次点积变成单个算子调用，可以直接落在 GPU 的 Tensor Core 上高速执行。可以说 **Attention 的算法设计和 GPU 的硬件特性是相互成就的**：它选了硬件最擅长的运算来表达注意力语义。

一个直观的类比：矩阵乘法之于 Transformer，就像卷积之于 CNN——都是"把数学定义直接映射到硬件最擅长的算子"的设计。
