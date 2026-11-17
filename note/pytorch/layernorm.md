## Q1: LayerNorm 和 BatchNorm 有什么区别？

两者都是归一化层，但归一化的维度不同：

| 维度 | BatchNorm | LayerNorm |
|---|---|---|
| 计算统计量用的维度 | 在 **batch 维**上：同一个通道，跨整个 batch 计算均值和方差 | 在 **特征维**上：同一条样本，跨 hidden_size 计算均值和方差 |
| 依赖 batch | 是，统计量随 batch 内容变化 | 否，每条样本独立计算 |
| 序列数据 | 不适用（batch 内样本长度不同时很难做，且测试时要记住训练期的滑动统计量） | 天然适用，每个 token 独立 |
| train/eval 行为 | 两种行为不同（训练用 batch 统计量，测试用滑动平均） | 两种行为相同 |

具体看 Transformer 输入 `[batch, seq_len, hidden_size]`：

```python
# BatchNorm 希望输入 [N, C, *]，对 C 维跨 batch 算统计量
nn.BatchNorm1d(hidden_size)  # 需要把 [batch, seq_len, hidden] 转置成 [batch, hidden, seq_len]

# LayerNorm 直接对最后一维归一化
nn.LayerNorm(hidden_size)    # 对 hidden_size 维，每个 token 独立
```

一句话：**BatchNorm 归一化的是"同一特征在不同样本间的分布"，LayerNorm 归一化的是"同一样本内部不同特征之间的分布"。**

## Q2: Transformer 的输入通常是 [batch, seq_len, hidden_size]，为什么 nn.LayerNorm(hidden_size) 很自然？

`nn.LayerNorm(hidden_size)` 对输入的**最后一维**做归一化。对 `[batch, seq_len, hidden_size]` 应用它，相当于对 `(batch × seq_len)` 个 token 向量，每个都独立地在自己的 `hidden_size` 维上减均值、除标准差：

```python
ln = nn.LayerNorm(hidden_size)
x = torch.randn(batch, seq_len, hidden_size)
y = ln(x)  # 输出 shape 不变，每个 token 向量被独立归一化
# 数学上等价于对每个 x[i, j, :] 分别做 (x - mean) / sqrt(var + eps)
```

之所以自然，原因有三个：

1. **shape 匹配**：归一化维度恰好就是特征维 `hidden_size`，不需要任何转置或 reshape。
2. **每个 token 一个向量**：Transformer 的注意力、FFN 都是逐 token 操作的，LayerNorm 也是逐 token 归一化，语义一致——"把每个 token 的特征向量调整到零均值、单位方差"。
3. **与 batch 和 seq_len 无关**：不管 batch 里有多少条样本、每条序列有多长，归一化行为完全一样，不会引入 BatchNorm 那种跨样本的耦合。

## Q3: 如果 BatchNorm 的 batch size 很小，可能出现什么问题？

BatchNorm 的均值和方差是用当前 batch 的数据估计的。batch size 很小时：

1. **统计量噪声大**：均值/方差估计不准（高方差），每次迭代的归一化结果波动剧烈，训练不稳定、 loss 抖动。
2. **train/eval 不一致被放大**：训练用 batch 统计量（噪声大），测试用滑动平均（相对准确），两者的系统性偏差在小 batch 下更严重，导致 eval 性能明显掉。
3. **极端情况 batch size = 1**：能否计算取决于每个通道有多少个统计元素。
   - 输入 `[1, C]`（无空间维度）：每个通道只有 1 个元素，方差无定义，训练直接报错：

     ```python
     bn = nn.BatchNorm1d(4)
     bn(torch.randn(1, 4))  # train 模式下
     # ValueError: Expected more than 1 value per channel when training,
     #             got input size torch.Size([1, 4])
     ```

   - 输入 `[1, C, L]`（有空间维度，`L > 1`，如 Conv 特征图）：统计量在 batch 和空间位置维度（N、L）上联合计算，通道内仍有 L 个元素，训练可以正常进行，只是统计量完全来自单条样本，质量受影响。

   以上两种情况的实验见 `notebooks/day40_layernorm.ipynb` 倒数第二个 cell。
4. **分布式训练更糟**：每个 GPU 算自己小 batch 的统计量（而不是全局统计量），进一步加剧噪声。

实践结论：BatchNorm 在 batch size ≥ 32 左右才比较稳定，小 batch 场景（如检测、分割、大模型训练）通常改用 LayerNorm、GroupNorm 等不依赖 batch 的归一化。

## Q4: BatchNorm 的 train / eval 行为

两种模式行为不同，用 `model.train()` / `model.eval()` 切换：

**train 模式**：
- 用**当前 batch** 的数据计算均值和方差；
- 同时把 batch 统计量累积进滑动平均（running mean / running var），公式为 `running = (1 - momentum) * running + momentum * batch_stat`，`momentum` 默认 0.1。

**eval 模式**：
- 不再计算 batch 统计量，直接用训练期累积下来的 running mean / var 做归一化；
- 行为变成确定性的，不再受输入 batch 内容影响。

```python
bn = nn.BatchNorm1d(8)
x = torch.randn(4, 8)

bn.train()
y_train = bn(x)          # 用 x 自己的均值方差
print(bn.running_mean)   # 已被更新

bn.eval()
with torch.no_grad():
    y_eval = bn(x)       # 用 running_mean / running_var
```

注意点：

- 忘记调 `eval()` 是常见 bug：测试时仍在更新 running 统计量，且归一化结果随 batch 内容漂移，指标不可信。
- `bn(x)` 在 eval 下的结果不等于用当前 x 重新算统计量的结果。
- LayerNorm 没有 running 统计量，`train()` 和 `eval()` 下行为完全相同，这也是它省心的地方。

### 实验

见 `notebooks/day40_layernorm.ipynb` 最后两个 cell（`track_running_stats=True` 默认值），实验输出验证了三个要点：

1. **train 的归一化和 running 更新是两件事**：前向输出始终用当前 batch 统计量；running 只是顺带用 `momentum`（默认 0.1）做滑动平均，为 eval 做准备。例如两个 batch 均值分别为 `[2, 3, 4]`、`[6, 7, 8]`，running_mean 依次更新为 `[0.2, 0.3, 0.4]` → `[0.78, 0.97, 1.16]`。
2. **eval 用的是整个训练历史的加权平均**，不是任何单个 batch 的统计量——所以输入 `[10, 11, 12]` 这种远离训练分布的数据，输出是 `[8.45, 9.19, 9.94]` 而不是 0 附近；且 eval 输出与 `(x - running_mean) / sqrt(running_var + eps)` 手动计算完全一致。
3. **eval 下重复调用结果完全一致**（确定性），不受输入内容影响。

## Q5: Transformer 为什么常用 LayerNorm

1. **不依赖 batch 大小**：见 Q3，大模型训练显存紧张，batch size 往往不大，BatchNorm 的统计量噪声会成为问题；LayerNorm 逐样本计算，batch size 再小也稳定。
2. **天然适配变长序列**：不同样本的 seq_len 不同，padding 位置会污染 BatchNorm 的统计量；LayerNorm 每个 token 独立计算，padding 只会影响 padding 自己（配合 `padding_idx` 和 attention mask 即可）。
3. **train/eval 行为一致**：没有 running 统计量，训练和推理严格一致，也不会出现忘记切 `eval()` 这类 bug；对推理友好，结果确定性。
4. **逐 token 语义与 Transformer 匹配**：Transformer 的所有计算（attention、FFN）都在特征维上进行，LayerNorm 恰好归一化特征维，几何意义清晰——控制每个 token 表示的尺度，稳定前向传播的数值范围，缓解梯度问题，让深层网络可训练。
5. **历史验证**：原始 Transformer 论文（Attention Is All You Need）就采用 LayerNorm，之后 BERT、GPT 等主流预训练模型沿用并验证了它的稳定性，已成为事实标准。
