## 问题：训练 Loss 持续降低，但 Validation Loss 开始上升，说明什么？如何处理？

说明模型出现了**过拟合（Overfitting）**。

含义：模型在训练集上越来越好（甚至把训练数据里的噪声都记住了），但泛化能力在变差——对没见过的数据预测误差越来越大。训练的目标不是"记住训练数据"，而是"学到数据背后的规律"，而 val loss 上升说明学偏了。

以 day43 的实验为例：训练集只有 20 个带噪声的样本，模型却有 128×128 的两层网络（约 1.6 万参数）。参数远多于数据，模型完全有能力把 20 个点（包括噪声）全部穿过，train loss 降到接近 0；但对真实函数 sin(3x) 的拟合反而在第 31 个 epoch 后变差。

处理方法：

1. Early Stopping（早停）：监控 val loss，在其开始上升时停止训练，或保存 val loss 最低时的模型权重。这是最直接控制训练时长的方法。
2. 减小模型容量：减少层数/神经元数量，降低模型的记忆能力，让它只能学规律、背不下噪声。
3. 增加训练数据：更多训练样本让"记住全部数据"变难，迫使模型学习泛化规律；也可以做数据增强。
4. 正则化：典型方法是 Weight Decay（L2 正则）和 Dropout。BatchNorm 可能带来一定的正则化副作用，但不应把它当作通用、可靠的抗过拟合方法。

注意：降低学习率不一定能解决过拟合。学习率主要影响的是优化过程（收敛快慢与稳定性），而不是模型对训练数据的记忆程度；想控制训练时长，直接的手段是 Early Stopping。

判断依据：val loss 是训练过程中"泛化能力"的体检表，train/val loss 曲线的分离程度就是过拟合的严重程度。

## 实验：day43_training_debugging 的训练过程

### 1. 数据准备

- 训练集：20 个样本，x 均匀采样自 [-1, 1]，y = sin(3x) + 0.3×高斯噪声（故意又少又噪，容易过拟合）。
- 验证集：200 个密集采样点，y = sin(3x)，不带噪声，代表真实函数。

### 2. 模型与优化器（故意用大网络制造过拟合）

```python
class OverfitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

model = OverfitModel()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
```

输入 1 维、输出 1 维的回归任务，中间两层 128 个神经元（约 1.6 万参数），对 20 个样本来说是严重超参数量的配置。

### 3. 训练循环（full-batch，无 mini-batch）

每个 epoch 分两段：

训练段：
1. `model.train()`：切换训练模式（对当前模型无实际作用，但属于标准写法）。
2. `optimizer.zero_grad()`：清零梯度（PyTorch 梯度默认累加，不清零会叠加上一轮的旧梯度）。
3. `train_pred = model(x_train)` + `loss_fn`：前向传播，计算 20 个训练点上的 MSE。
4. `train_loss.backward()`：反向传播，计算每个参数的梯度。
5. `optimizer.step()`：Adam 按梯度更新参数。

验证段：
1. `model.eval()`：切换评估模式。
2. `with torch.no_grad():`：关闭梯度——验证不更新参数，不记录计算图，省内存且更快。
3. 在 200 个验证点上计算 val loss。

最后把两个 loss 用 `.item()` 转成标量存入 `train_losses` / `val_losses`，供后续画曲线和诊断。

### 4. 实验结果

- 训练 1500 个 epoch，Best epoch = 31（val loss 最低 0.0359）。
- 最终 train loss = 0.040，val loss = 0.0996。
- 曲线形态：train loss 一路下降趋近 0；val loss 在前 31 轮下降，之后持续回升——教科书式的过拟合曲线。第 31 轮之后训练的每一步都在进一步记忆训练集的噪声，离真实函数越来越远。
- 启示：早停在这里等价于"只用前 31 个 epoch 的模型"，其余 1469 轮全是负收益。
- Training Loss 衡量模型对训练数据的拟合程度；Validation Loss 帮助我们判断模型能否泛化到没见过的数据。