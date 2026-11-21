# Week Review: D36-D41

## 1. What I Learned

- Traing Loop:
- DataSet Dataloader:
- Embedding:
- Activation Dropout:
- Layernorm:
- Device Checkpoint:

## 2. What I Can Implement Independently

- [ ] Create a training loop 
- [ ] Define a DataSet and build a Dataloader with it
- [ ] Build Token embedding
- [ ] Experiment on ReLU, GELU and Dropout
- [ ] Experiment on LayerNorm and BatchNorm
- [ ] Use CUDA, use checkpointing

## 3. Knowledge Gaps

- Topic 1:	RELU和GELU具体训练什么用？
ReLU 和 GELU 都是激活函数，作用完全一样：给网络引入非线性。如果没有它们，多层神经网络叠再多层也等价于单层线性变换，什么也学不了。
- Topic 2:  Transformer用的是LayerNorm, 为什么还需要学习BatchNorm?
LN 并不是"更好的归一化"，而是"更适合 Transformer 场景的归一化"。BN 在 CNN 视觉任务里依然占主导地位。

## 4. Experience notebooks/day42_review_d36_d41.ipynb

**实验目的**：不做真实训练（无 loss / optimizer），把 D36–D41 学过的组件串成一个最小文本分类器 `MiniClassifier`，验证各组件拼在一起时前向传播的 shape 能正确衔接。

**模型结构**（每个组件对应一天的笔记）：

```python
Embedding → LayerNorm → mean(dim=1) 池化 → Dropout(0.2) → Linear(16, 2) 分类头
```

| 组件 | 来源 | 作用 |
|---|---|---|
| `nn.Embedding(100, 16)` | D38 | token id 查表 → 16 维向量 |
| `nn.LayerNorm(16)` | D40 | 每个 token 向量在特征维上归一化 |
| `x.mean(dim=1)` | D36/37 | 变长序列压成固定句子向量（mean pooling） |
| `nn.Dropout(0.2)` | D39 | 分类前随机丢弃 20%，防过拟合 |
| `nn.Linear(16, 2)` | D32 | 输出 2 类 logits |

**输入**：`torch.randint(0, 100, (8, 5))` —— 8 个样本，每个 5 个 token 的随机 id。

**前向 shape 变化链**：

```
(8, 5)          # token ids
→ (8, 5, 16)    # embedding：每个 token 变 16 维向量
→ (8, 5, 16)    # LayerNorm：逐样本逐 token 归一化
→ (8, 16)       # mean(dim=1)：5 个 token 平均成 1 个句子向量
→ (8, 16)       # dropout
→ (8, 2)        # 分类头：2 类 logits
```

**实验结果**：

- `assert logits.shape == (8, 2)` 通过，输出 `torch.Size([8, 2])`。
- 关键观察：D32 中 `nn.Linear(1, 1)` 最后一维 1→1 形状不变（只做 `y = w·x + b`）；本实验 `Linear(16, 2)` 把最后一维从 16 压到 2，shape 才发生变化——Linear 只改最后一维，其余维度原样保留。

**结论**：五个拼图（embedding 查表、LayerNorm 稳定分布、mean pooling 处理变长序列、dropout 正则化、Linear 输出分类）能无缝拼成完整前向流程；接上 `CrossEntropyLoss` 和 optimizer（D34/D36）即可进入真实训练。



