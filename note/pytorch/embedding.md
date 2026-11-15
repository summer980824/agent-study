## Q1: nn.Embedding 的参数和输出 Shape

```python
nn.Embedding(num_embeddings, embedding_dim, padding_idx=None)
```

- `num_embeddings`：词表大小，即权重矩阵的行数（多少个词）。
- `embedding_dim`：每个词向量的维度，即权重矩阵的列数。
- `padding_idx`（可选）：指定某个 ID（通常是 `<pad>`=0）的向量初始化为零，且反向传播时该行的梯度被置零，避免填充位污染训练。注意"梯度为零"不等于"永不变化"：若优化器带 weight_decay，该行仍可能被权重衰减改动。

内部权重 `embedding.weight` 的 Shape 为 `[num_embeddings, embedding_dim]`，是可训练的 `nn.Parameter`，默认按标准正态分布 N(0, 1) 随机初始化。

**输出 Shape**：输入是任意形状的 LongTensor（每个元素是一个词 ID），输出在输入形状后面拼接 embedding_dim。

| 输入 | 含义 | 输出 Shape |
|---|---|---|
| `[seq_len]` 一维 | 一句话 | `[seq_len, embedding_dim]` |
| `[batch, seq_len]` 二维 | 一批句子 | `[batch, seq_len, embedding_dim]` |

例（day38）：`num_embeddings=5, embedding_dim=4`，输入 `[4]` → 输出 `[4, 4]`。

## Q2: Tokenizer 与 Embedding 的职责区别

- **Tokenizer**：文本 → 数字。负责分词、维护词表，把原始字符串（如 `"i like pytorch"`）映射成 ID 序列（如 `[1, 2, 3]`）。它只做查表替换，输出的是整数，本身没有可训练参数（BPE 等词表虽从语料统计构建，但不参与梯度训练）。
- **Embedding**：数字 → 向量。负责把 Tokenizer 产出的 ID 序列映射成稠密浮点向量，供神经网络后续层计算。它是模型的一部分，权重随训练更新。

一句话：**Tokenizer 决定"哪个词是哪个编号"，Embedding 决定"这个编号代表什么语义"。** 前者是固定规则，后者是可学习参数。

## Q3: 为什么 Embedding 本质上是 Lookup Table

因为 `nn.Embedding` 的内部实现就是一个形状 `[词表大小, 嵌入维度]` 的权重矩阵，前向计算没有任何数学运算，只是按输入的 ID 取对应的行：

```python
embedding(token_ids) == embedding.weight[token_ids]  # assert 成立
```

输入 `[1, 2, 3, 1]` 就是依次取出第 1、2、3、1 行。它既不乘也不加，纯查表。"可训练的查找表"就是它区别于普通字典的地方：表的内容（weight）通过反向传播学习。

## Q4: Lookup 与 One-hot 矩阵乘法的数学等价关系

设词表大小为 V，嵌入维度为 d，权重矩阵 W 形状 `[V, d]`，某个词的 ID 为 i。

- **One-hot 表示**：e_i 是一个长度为 V 的向量，只有第 i 位为 1，其余为 0。
- **矩阵乘法**：`e_i @ W` 的结果 = 0 乘各行 + 1 乘第 i 行 = **W 的第 i 行**。

逐分量看：`(e_i @ W)_j = Σ_k e_i[k] · W[k, j] = W[i, j]`，因为求和里只有 k=i 那一项非零。

所以 `one_hot @ embedding.weight` 与 `embedding.weight[ids]`（即 Embedding 查表）结果完全一致（day38 中 `assert torch.allclose(...)` 验证了这一点）。Embedding 相当于把"构造 one-hot + 矩阵乘法"这两步合成了一次高效的按索引取行。

## Q5: Embedding Weight 如何通过 Backpropagation 更新

`embedding.weight` 是 `nn.Parameter`（`requires_grad=True`），因此：

1. **前向**：查表取出某些行参与后续计算，输出带上 `grad_fn=EmbeddingBackward0`（autograd 记录了"这次前向读了哪些行"）。
2. **反向**：梯度从损失经后续层传回到 Embedding 输出，再经 `EmbeddingBackward` 传播到 weight。由于前向只是取行，反向等价于"把输出梯度放到对应行、其余行补 0"——即 weight 的梯度只有被本次输入引用过的行非零（仍是普通 dense 张量，未引用行全为零，并非 sparse tensor 类型）。
3. **更新**：对不带 momentum 和 weight_decay 的普通 SGD，按 `weight -= lr * grad` 更新时只有被引用行被改变，未引用行保持不变。但注意：**Adam、带动量的 SGD 或 weight_decay 下，未引用行也可能变化**——历史动量/一阶矩的非零残留会继续推动更新，权重衰减则作用于整张表。

这就是"语义相近的词向量也相近"的来源：同一个词在每次出现时都共享同一行向量，模型为了降低损失会不断调整它，使其在上下文中"好用"，相似用法下的词会被推向相似的向量。

## Q6: embedding_dim 的值由什么决定

`embedding_dim` 不是由数据计算出来的，而是人为设定的**超参数**，靠经验规律和实验效果权衡。决定因素：

1. **词表大小**：经验上维度与词表规模正相关。常用参考：小词表几十维；Word2Vec 约 300 维（百万级词表）；BERT/GPT 等 768–4096 维（词表几万到十几万）。有人用 `log2(vocab_size)` 做启发式参考（类比二进制编码所需位数），但它不是真实下界——维度更小的 embedding 也能正常工作。
2. **任务与数据量**：数据少时维度太大会过拟合（每个词出现次数少，学不充分），宜小；数据多、任务复杂可以更大以换取表达力。
3. **下游架构约束**：Embedding 输出通常直接接后续层，维度常和模型 hidden size 对齐（如 BERT 为 768），省去投影层；同时受参数预算限制——embedding 层常是全模型参数大头，维度翻倍参数量即翻倍。
4. **实验调优**：无解析解。以同类任务的取值做起点，观察验证集表现——太小欠拟合（词间区分不开）就调大，太大过拟合就调小；训练后可检查向量质量（如相近词的余弦相似度是否高）。

**一句话**：embedding_dim 是"表达能力"与"参数量/过拟合风险"的权衡，由词表规模、数据量、下游架构共同约束，最终靠实验选定。

## Q7: PyTorch 中的 nn.Embedding 本质上是什么

一个**包装了可训练权重矩阵（`nn.Parameter`）的 `nn.Module`，前向计算只做按索引取行（index_select）**。

拆开看：

- **数据上**：内部只有 `self.weight` 一个参数，Shape `[num_embeddings, embedding_dim]`，就是"词 → 向量"的查找表（见 Q3）。
- **行为上**：`forward` 没有矩阵乘法或激活函数，把输入 ID 张量当作行索引，从 weight 中取出对应行返回，因此输入任意形状 `[..., ]` 的 LongTensor，输出为 `[..., embedding_dim]`。
- **训练上**：因为是 `nn.Parameter`（`requires_grad=True`），自动接入 autograd（输出带 `EmbeddingBackward0`）并注册到优化器中，随反向传播更新（见 Q5）。
- **工程上**：额外提供了 `padding_idx`（某行固定为零且不更新）、`max_norm`（向量范数裁剪）、`from_pretrained`（用预训练向量初始化）等便利功能。

一句话：**nn.Embedding = 一张会随梯度下降不断修改内容的查找表，是线性层（one-hot @ W）的高效等价实现。**
