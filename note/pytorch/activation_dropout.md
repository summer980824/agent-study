# Activation 与 Dropout(ReLU / GELU / Dropout)

> 对应 notebook:`notebooks/day39_activation_dropout.ipynb`

## 1. ReLU 与 GELU 对比及适用场景

| | ReLU | GELU |
|---|---|---|
| 公式 | $\max(0, x)$ | $x \cdot \Phi(x)$,$\Phi$ 为标准正态 CDF |
| 负半轴 | 硬截断为 0(有 dead ReLU 风险) | 微小非零输出,不会死神经元 |
| 0 附近 | 折角,不可导 | 平滑,处处可导 |
| 计算开销 | 极小(一次 max) | 较大(erf 或 tanh 近似) |
| 梯度特性 | 正区间梯度恒为 1 | 正区间梯度略小于 1(约 0.83 峰值) |

**适用场景:**

- **ReLU**:CNN 等计算敏感的场景,或模型较小、想压低开销。简单高效,但神经元可能"死亡"。
- **GELU**:现代 Transformer(BERT、GPT 等)的主流选择。平滑 + 负区间保留梯度,在深度残差结构中训练更稳定,实证效果通常优于 ReLU。
- 两者整体形状相似,GELU 可看作 ReLU 的"平滑 + 概率门控"版本。

## 2. Dropout 的随机 Mask 与 inverted dropout 缩放机制

训练时,每个元素以概率 $p$ 被随机置零:

$$\tilde{h} = m \odot h, \quad m_i \sim \text{Bernoulli}(1-p)$$

这带来一个期望值的错位:若只置零不缩放,训练期输出的期望是 $(1-p)h$,而测试期用全部神经元输出为 $h$,两者不匹配。

**Inverted Dropout**(PyTorch 采用):训练时对保留的元素除以 $1-p$ 做缩放,使期望值保持为 $h$:

$$\tilde{h} = \frac{m \odot h}{1-p}, \quad E[\tilde{h}] = h$$

测试时直接恒等输出,什么都不用做。好处是缩放逻辑集中在训练阶段,推理代码干净且高效。

## 3. `train()` 与 `eval()` 的区别

- `model.train()`:启用训练模式 —— Dropout 生效(随机置零 + 缩放),BatchNorm 使用 batch 统计量。
- `model.eval()`:关闭 Dropout(恒等映射),BatchNorm 改用累积的 running 统计量,推理输出变为**确定性**。

关键陷阱:推理时忘了 `model.eval()`,输出每次都会随机变化 —— 这是新手最常见的 bug。

## 4. day39 三个实验的实际观察结果

**实验 1:ReLU vs GELU 输出与梯度曲线**

- 负输入下,ReLU 输出全为 0,而 GELU 输出微小的负数(如 `-0.0040, -0.0047, -0.0054, ...`)—— 验证了 GELU 负半轴不为零、平滑过渡。
- 梯度曲线:ReLU 梯度在 0 处从 0 跳变到 1(折角不可导);GELU 梯度连续平滑,在 $x \approx 0.17$ 附近达到峰值约 0.83。

**实验 2:Dropout 的 mask 与缩放(`nn.Dropout(p=0.5)`,输入全 1)**

- 训练输出均值 `1.014` ≈ 输入均值 `1.0` —— 验证了 inverted dropout 的缩放使期望值保持不变。
- 输出唯一值只有 `{0., 2.}` —— 一半被置零,保留的一半被缩放到 $1/(1-0.5)=2$。
- 两次前向输出不相等(随机 mask);切到 `eval()` 后输出**完全等于输入**(`torch.equal` 通过)。

**实验 3:模型层面的 train/eval(`GELU + Dropout(p=0.25)` 的小网络)**

- `model.train()` 下两次前向输出不相等(dropout 随机性生效)。
- `model.eval()` + `torch.no_grad()` 下两次前向输出完全相等(确定性,`assert` 通过)。

## 5. Transformer 为什么常见 GELU?

1. **平滑无死区**:Transformer 动辄上百层,ReLU 的硬截断容易造成大量死神经元;GELU 负半轴保留微小梯度,深层训练更稳定。
2. **与随机正则的天然联系**:GELU 可解释为"dropout + ReLU 期望形式"的平滑近似($x \cdot P(x + \epsilon > 0)$,$\epsilon \sim N(0,1)$),和 Transformer 中广泛使用的 dropout 理念一致(Hendrycks & Gimpel, 2016)。
3. **实证效果**:BERT、GPT-2/3 等都用 GELU,在 NLP 任务上比 ReLU 有稳定的小幅提升。
4. 计算开销在现代框架下已不是问题(`F.gelu` 有高效的 tanh 近似实现)。
