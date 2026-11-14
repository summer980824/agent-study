# Dataset 与 DataLoader

对应 notebook：`notebooks/day37_dataset_dataloader.ipynb`

## Dataset 的职责

`torch.utils.data.Dataset` 是数据管线的最底层抽象，代表"一份可索引的数据集"。自定义 Dataset 必须实现两个方法：

- `__len__()`：返回数据集有多少个样本。`DataLoader` 用它计算 batch 数量、shuffle 时确定抽样范围。
- `__getitem__(index)`：给定下标，返回第 `index` 个样本（如 `(x, y)` 元组）。这是取数据的唯一入口。

示例（`day37_dataset_dataloader.ipynb` 中的 `LinearDataset`）：

```python
class LinearDataset(Dataset):
    def __init__(self, x, y):
        if len(x) != len(y):
            raise ValueError('x and y must have the same length')
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]
```

Dataset 只负责"单个样本长什么样、怎么取"，不关心 batch、顺序等上层逻辑。它的价值在于提供统一抽象：无论数据在内存、磁盘图片还是数据库里，对上层来说都长一个样；大数据集还可以在 `__getitem__` 里做按需加载。

## DataLoader 的职责

`torch.utils.data.DataLoader` 包在 Dataset 外面，负责把"单个样本"组织成"训练可用的批次流"：

- **batching**：每次迭代向 Dataset 取 `batch_size` 个索引的样本，自动堆叠成张量。
- **shuffle**：每轮（epoch）重新打乱取数顺序（`shuffle=True`），避免模型学到样本的固定排列规律。
- **iteration**：实现迭代协议，`for batch_x, batch_y in train_loader` 即可逐 batch 遍历整个数据集。

示例：

```python
train_loader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=False)
```

## Dataset 与 DataLoader 的区别

| | Dataset | DataLoader |
|---|---|---|
| 抽象层次 | 数据层：单个样本 | 加载层：样本流 |
| 核心问题 | 第 `i` 个样本是什么 | 样本以什么顺序、什么粒度喂给模型 |
| 关键接口 | `__len__`、`__getitem__` | 迭代器、batch 维度 |
| 类比 | 书架（存书、按编号取书） | 图书管理员（按规则一次抱一摞给你） |

一句话：Dataset 定义"数据是什么"，DataLoader 定义"数据怎么批量地、打乱地流给模型"。

## batch size 如何影响每个 epoch 的更新次数

每个 batch 做一次参数更新，所以一个 epoch 内的更新次数 = batch 数量：

```
更新次数 = ceil(N / batch_size)        # drop_last=False
更新次数 = floor(N / batch_size)       # drop_last=True
```

以 100 个样本为例：

| batch_size | batch 数（更新次数） |
|---|---|
| 100 | 1（full batch） |
| 50 | 2 |
| 16 | 7（最后一批只有 4 个样本） |
| 1 | 100（随机梯度下降 SGD） |

batch size 越小，一个 epoch 内参数更新越频繁（每次更新更接近真实梯度方向、噪声更小方向更随机）；batch size 越大，单次更新的梯度估计越准，但更新次数少、内存占用高。

## 为什么训练中通常用 mini-batch，而不是一次性加载整个数据集

三个原因：

1. **内存限制**：整个数据集（尤其是图像、文本）往往放不进显存/内存。mini-batch 每次只处理一小撮数据，内存开销可控。
2. **计算效率与并行**：GPU 擅长对成块的张量做并行矩阵运算。batch_size=1 时矩阵运算退化成向量运算，硬件利用率低、跑不快；适中的 batch 能同时喂饱并行单元。
3. **训练动力学**：full batch 每次都用全量数据算精确梯度，容易卡在尖锐的局部极小值或鞍点，且每一步计算昂贵；batch_size=1（纯 SGD）噪声太大、收敛不稳。mini-batch 在两者之间取得平衡——梯度估计足够准，又保留一定的随机噪声帮助跳出坏点，实际收敛通常最好。

实践中的取舍：batch size 太大 → 内存溢出、泛化可能变差；太小 → 硬件利用率低、更新噪声大。常见做法是选 GPU 显存能容纳的最大 batch，再相应调大学习率。
