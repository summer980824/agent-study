# Tensor

就是一个可以存储多维数据的容器，类似于 Java 中的多维数组或 NumPy ndarray，但它额外支持 GPU 计算和自动求导。

在深度学习中，模型接收的数据、模型参数、计算过程中的中间结果，几乎全部都是 Tensor。

## Tensor 和 NumPy array 有什么关系？

关系非常密切，可以概括为一句话：**Tensor 可以理解为"加强版的 NumPy array"**，但二者不能互相替代。

**共同点：**

- 都是存储多维数组（N 维数据）的容器，API 设计高度相似（`shape`、`dtype`、索引、广播、切片等概念完全一致）。
- 底层数据在内存中的布局方式类似，因此可以几乎零拷贝地互相转换。

**关键区别：**

| 维度 | NumPy array | Tensor |
|------|-------------|--------|
| 运行设备 | 只能跑在 CPU 上 | 可以跑在 CPU / GPU / NPU 上，且可无缝迁移 |
| 自动求导 | 不支持 | 支持 `requires_grad`，能记录计算图并自动反向传播 |
| 计算图 | 无 | 支持，是深度学习训练的基础 |
| 生态定位 | 科学计算通用库 | 深度学习框架（PyTorch、TensorFlow 等）的核心数据结构 |

**互相转换：**

```python
import torch
import numpy as np

# NumPy -> Tensor
arr = np.array([1.0, 2.0, 3.0])
t = torch.from_numpy(arr)   # 共享底层内存，修改一个会影响另一个

# Tensor -> NumPy
x = torch.tensor(
    [1.0, 2.0, 3.0],
    requires_grad=True
)

# CPU 上的 Tensor 才能直接转，带有梯度追踪的 Tensor 不能直接调用 .numpy()
arr = x.detach().cpu().numpy()
```

**为什么还需要 Tensor？**

NumPy 只做"数值计算"，而深度学习需要"在 GPU 上大规模并行计算 + 自动求导"。Tensor 把这两件事封装进了和 NumPy 几乎一样的接口里，让熟悉 NumPy 的人几乎零成本上手深度学习框架。

## 什么是 PyTorch Broadcasting？

广播（Broadcasting）是 PyTorch（和 NumPy）做**逐元素运算**时的一种自动扩展机制：当两个 tensor 的 shape 不同时，它不会直接报错，而是先尝试把较小的那个"虚拟地"拉伸复制，凑成和较大者兼容的形状，再做加减乘除。

好处是**不用手写循环、也不真正复制数据**（省内存），代码更简洁。Broadcasting 通常不需要物理复制输入 Tensor 来扩展维度；运算结果仍然会生成相应的输出 Tensor。典型场景：

```python
x = torch.ones(2, 3)
b = torch.tensor([10., 20., 30.])   # shape: (3,)

x + b   # 不需要把 b 手动复制成 2 行，广播自动搞定
```

## Broadcasting 的 Shape 匹配规则

把两个 shape **从右往左对齐**，逐维比较（缺失的维度视为 1），每一维满足以下**任意一条**即可广播：

1. 两个维度**相等**。
2. 其中一个是 **1**（这一维会被拉伸到和另一个一样长）。

如果某一维既不相等、也没有 1，就抛 `RuntimeError`。

例子（对应 notebook `day30_matrix_ops_broadcasting.ipynb`）：

| 例子 | shape | 对齐结果 | 结果 |
|------|-------|----------|------|
| `x + bias` | `(2,3) + (3,)` | `3 == 3` ✓，缺位补 1 拉伸成 2 ✓ | `(2,3)` |
| `x + column` | `(2,3) + (2,1)` | `2 == 2` ✓，`1 → 拉伸成 3` ✓ | `(2,3)` |
| `x + invalid` | `(2,3) + (2,)` | `3 vs 2`，不相等也无 1 ✗ | 报错 |

**一句话总结**：右边对齐、逐维看，"相等或含 1"就能广播，两个都不满足就报错。
