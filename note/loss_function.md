## MSE
nn.MSELoss() —— 均方误差损失
预测值与真实值之差的平方，取平均。常用于回归任务（预测连续值，如房价、评分）。
关键特性：
- 对大误差惩罚重（平方放大差异），对小误差"不敏感"（梯度随误差线性减小）
- 输出和 target 维度必须一致，无需额外处理
- 适合做回归，做分类时它不是最优

## Cross Entropy
nn.CrossEntropyLoss() —— 交叉熵损失
用于分类任务（预测类别标签，如"猫/狗/鸟"）。
关键特性：
- target 是类别索引（Long 类型），不是 one-hot 向量
- 即使预测正确但置信度低，loss 仍然大（鼓励模型更"自信"）——这正是它做分类优于 MSE 的核心原因
- 输出和 target 维度不同：logits 是 [batch, num_classes]，labels 是 [batch]


## 为什么 Cross Entropy 不需要手动 Softmax

`nn.CrossEntropyLoss()` 的输入是**未归一化的 logits（原始打分）**，不需要（也不应该）先过 Softmax。原因：

1. **它内部已经合并了 LogSoftmax + NLLLoss**
   数学上交叉熵 = `-log(softmax(logits)[正确类别])`。框架把这两步合成一个算子直接算，数值更稳定，还省一次显式归一化。

2. **数值稳定性**
   手动先算 Softmax 容易出现上溢/下溢（指数运算）。内部实现用的是 "log-sum-exp" 技巧，直接从 logits 算 log 概率，避免了大数相除。

3. **对梯度更友好**
   合并后的算子反向传播时梯度形式非常简洁：`∂loss/∂logits = softmax(logits) - one_hot(target)`。如果外面再套一层 Softmax，梯度会多绕一圈，既慢又容易梯度消失。

正确用法：
```python
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(logits, targets)   # logits: [batch, num_classes], targets: [batch]
```
错误用法：
```python
loss = loss_fn(torch.softmax(logits, dim=1), targets)  # ❌ 不要再手动 Softmax
```
（如果确实想从概率出发，应该用 `nn.NLLLoss()` 并手动 `log_softmax`，但通常没必要。）

## Loss 和 Evaluation Metric 的区别

**Loss（损失函数）** 是训练时优化器要**最小化**的目标，必须可导；**Metric（评估指标）** 是给人看、用来**衡量模型好坏**的，不一定可导。

| | Loss | Metric |
|---|---|---|
| 作用 | 反向传播、参数更新 | 评估、对比、调参 |
| 要求 | 可导、光滑、对误差敏感 | 直观、符合业务语义 |
| 例子 | MSE、Cross Entropy、MAE | Accuracy、F1、AUC、Precision/Recall |

以分类任务为例：
- **Accuracy** 只关心"预测对不对"，不关心置信度高低。一个 51% 置信度猜对的样本和 99% 置信度猜对的样本，accuracy 一样；
- **Cross Entropy** 同时惩罚"错"和"不自信"。同样猜对，置信度低的 loss 更大（见 notebook 中 confident vs uncertain 的例子）。

所以训练时看 loss、评估时看 metric。两者趋势通常一致（loss 降、accuracy 升），但不一定完全同步——有时 accuracy 已经 plateau，loss 还在缓慢下降（模型变得更自信），这通常无害；反过来 loss 降而 accuracy 不动，可能说明模型只是在过拟合少数难样本。