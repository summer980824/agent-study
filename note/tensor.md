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
t2 = torch.tensor([1.0, 2.0, 3.0])
arr2 = t2.numpy()           # CPU 上的 Tensor 才能直接转
```

**为什么还需要 Tensor？**

NumPy 只做"数值计算"，而深度学习需要"在 GPU 上大规模并行计算 + 自动求导"。Tensor 把这两件事封装进了和 NumPy 几乎一样的接口里，让熟悉 NumPy 的人几乎零成本上手深度学习框架。
