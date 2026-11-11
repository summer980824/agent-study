# nn.Module 是什么

nn.Module 是 PyTorch 中**所有神经网络模块的基类**。一个模型、一层（如 nn.Linear）、甚至你自己写的任何带参数的计算单元，都应该继承它。

它本质上是一个"**参数的容器 + 计算的约定**"：

- **容器**：你挂在 `self` 上的 `nn.Parameter`（张量的子类，`requires_grad=True`）会被**自动登记（register）**到模块里。你的子模块（如 `self.fc = nn.Linear(...)`）也会被自动登记为子模块。
- **约定**：你只需要实现 `forward(self, x)` 定义前向计算，直接 `model(x)` 即可调用（背后走的是 `__call__`，它会先跑注册的 hook，再调你的 `forward`）。

最小例子：

```python
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()          # 必须：初始化 nn.Module 的内部机制
        self.w = nn.Parameter(torch.randn(1))   # 自动登记为参数
        self.fc = nn.Linear(10, 1)               # 自动登记为子模块

    def forward(self, x):
        return self.fc(x) + self.w

model = MyModel()
```

## 为什么 PyTorch 模型通常需要继承 nn.Module？

简单说：手工管理参数和子模块太繁琐、太容易出错，nn.Module 把这套脏活全部自动化了。具体有这些收益：

1. **参数自动登记与收集**
   `nn.Parameter` 和子模块一旦赋值到 `self` 上，就被登记进 `_parameters` / `_modules` 字典。于是 `model.parameters()` 能一次性**递归地**拿到整棵模块树里所有参数（包括子模块的），交给优化器：
   ```python
   optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
   ```
   不用继承的话，你得自己维护一张参数列表，每加一层、每层加个参数都要手动同步。

2. **状态字典 state_dict（保存 / 加载模型）**
   `model.state_dict()` 返回所有参数和 buffer（如 BatchNorm 的 running_mean）的有序字典；`model.load_state_dict(...)` 能精确地把权重装回去。断点续训、模型分享都靠它。手工管理就得自己写序列化。

3. **设备与类型的整体迁移**
   `model.to('cuda')` / `.to(torch.float16)` 会递归地把所有参数和 buffer 一起搬走，不用逐层、逐参数手动 `.cuda()`。现代框架里模型和输入设备不匹配是最常见的报错之一，这一条直接消掉了整类 bug。

4. **训练 / 评估模式切换**
   `model.train()` 和 `model.eval()` 递归设置所有子模块的模式——Dropout 在 eval 下关闭、BatchNorm 在 eval 下用 running 统计量。框架靠这个约定知道当前该走哪条行为分支。

5. **与训练循环和生态无缝集成**
   优化器、`torch.save`、checkpoint、`DistributedDataParallel`、各种 hook（`register_forward_hook`）都默认你提供了 nn.Module 的接口（parameters / state_dict / train / forward）。不继承等于和整套生态脱节，什么都得自己造。

6. **命名与结构清晰**
   参数有层级名字（`fc.weight`、`fc.bias`），打印 `model` 能看到树状结构，调试和日志都方便。

一句话总结：**继承 nn.Module，你只需写 `__init__`（声明有哪些参数/子层）和 `forward`（写清怎么算），参数的收集、迁移、保存、模式切换、反向求导的入口全部交给框架**——这正是 PyTorch "搭积木式"建模型的基础。
