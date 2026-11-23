# Mini Neural Network：迷你文本情感分类（Day46）

对应 notebook：`notebooks/day46_mini_neural_network.ipynb`
任务：输入 `[情感词, 名词]` 两个 token，判断评论是正面（1）还是负面（0）。
模型：`Embedding(11, 8) → mean pooling → Linear(8, 2)`，无隐藏层，本质是"词嵌入 + 逻辑回归"。

## 1. 从 Token IDs 到 Logits 的 Shape 变化

以训练集为例（batch=8，每条 2 个 token，embedding_dim=8，num_classes=2）：

| 阶段 | 张量 | Shape | 含义 |
|---|---|---|---|
| 输入 | `X_train` | `[8, 2]` | token id，Long 类型 |
| Embedding | `embedded` | `[8, 2, 8]` | 每个 id 查表换成 8 维向量 |
| Mean Pooling | `pooled` | `[8, 8]` | 对序列维取平均，消去词数维 |
| Linear | `logits` | `[8, 2]` | 每个样本 2 个类别的原始打分 |

记忆规则：
- `nn.Embedding` 在最后一维**追加** embedding_dim：`[B, L] → [B, L, D]`。
- `mean(dim=1)` **消去**序列维：`[B, L, D] → [B, D]`。
- `nn.Linear` 只改**最后一维**：`[B, D] → [B, C]`。

测试集形状同理：`[2, 2] → [2, 2, 8] → [2, 8] → [2, 2]`。

## 2. Mean Pooling 具体做了什么，为什么要用

`pooled = embedded.mean(dim=1)`：把一个样本所有词的向量**逐维度取平均**，得到一句的向量。

```
词1向量 [e11, e12, ..., e18]
词2向量 [e21, e22, ..., e28]   →   句向量 [(e11+e21)/2, ..., (e18+e28)/2]
```

为什么用：
1. **变长序列 → 固定长度**：不管一句话有几个词，输出都是 `[B, D]`，才能接后面的 Linear 分类器。
2. **融合整句信息**：所有词都参与句向量，而不是只取第一个词或最后一个词。
3. **零参数、计算便宜**：不引入任何需要学习的参数。

代价：**丢失词序**。`good bad` 和 `bad good` 平均后完全一样。本例中第一个情感词已决定标签，所以够用；对词序敏感的任务需要 RNN / CNN / Attention 等结构。

## 3. 为什么 CrossEntropyLoss 输入 logits 而不是 Softmax 概率

`nn.CrossEntropyLoss()` 内部已经合并了 `LogSoftmax + NLLLoss`，直接吃原始 logits：

```python
loss = loss_fn(logits, y_train)                      # ✅ 正确
loss = loss_fn(torch.softmax(logits, dim=1), y_train) # ❌ 不要手动 Softmax
```

原因：
1. **数值稳定**：内部用 log-sum-exp 技巧直接从 logits 算 log 概率。若先手动 Softmax，极小概率可能下溢成 0，再取 log 就是 `-inf`。
2. **梯度简洁**：合并算子的梯度为 `softmax(logits) - one_hot(target)`，形式简单、反向传播高效。
3. **分工明确**：训练时只需要 logits；Softmax 只在**推理时**想查看概率才用，预测类别直接 `logits.argmax(dim=1)` 即可（Softmax 是单调变换，不改变 argmax 结果）。

详见 `loss_function.md`。

## 4. Initial Loss、Final Loss、Train/Test Accuracy

| 指标 | 数值 | 解读 |
|---|---|---|
| Initial Loss | 0.6681 | 接近 `ln 2 ≈ 0.693`：随机初始化下两类预测接近均匀分布（各 50%），符合预期 |
| Final Loss | 1.49e-07 | 几乎为 0，模型对训练集不仅全对，而且置信度极高（完全拟合） |
| Train Accuracy | 1.0 | 8 条训练样本全部预测正确 |
| Test Accuracy | 1.0 | 2 条测试样本全部预测正确 |

注意：测试集只有 2 条，且模式与训练集完全一致（第一个情感词决定标签，名词无影响）。100% 的测试准确率**不能说明泛化能力强**，这个实验的价值在于跑通完整训练流程，而不是模型能力。

## 5. 从输入到 Loss，一个简单的 PyTorch 神经网络经历了什么

以前向传播为例，数据依次经过：

1. **查表（Embedding）**：离散 token id → 连续词向量，得到 `[B, L, D]`。
2. **聚合（Mean Pooling）**：词向量序列 → 固定长度句向量，得到 `[B, D]`。
3. **打分（Linear）**：句向量 → 各类别 logits，得到 `[B, C]`。
4. **算损失（CrossEntropyLoss）**：logits 与真实标签对比，输出一个**标量** loss：

   ```
   loss = -(1/N) Σ log( softmax(logits_i)[y_i] )
   ```

   即：对每个样本取正确类别的负对数概率，再对 batch 求平均。预测越不准、越不自信，loss 越大。

前向过程中 autograd 悄悄记录了计算图，随后才是训练的后半段：

```
loss.backward()      # 沿计算图反向传播，算出每个参数的梯度 param.grad
optimizer.step()     # Adam 根据梯度更新参数
optimizer.zero_grad() # 清零梯度，准备下一轮（在 backward 之前调用即可）
```

循环以上步骤，参数不断调整，loss 持续下降——这就是"神经网络在学习"的具体含义。
