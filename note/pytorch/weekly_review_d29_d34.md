# Week Review: D29-D34

## 1. What I Learned

- Tensor:
- Matrix Operations:
- Autograd:
- nn.Module:
- Loss Function:
- Optimizer:

## 2. What I Can Implement Independently

- [ ] Create and manipulate Tensors
- [ ] Perform matrix operations
- [ ] Implement a simple linear regression
- [ ] Build a model with nn.Module
- [ ] Calculate MSE / CrossEntropy Loss
- [ ] Train a model using SGD / Adam

## 3. Knowledge Gaps

- Topic 1:	为什么 logits (1, 3) 不能与 targets (2,) 直接用于普通 CrossEntropy？
CrossEntropy需要相同数量的样本数
- Topic 2:  Accuracy = 0.5 代表什么？
代表样本预测一半正确
- Topic 3:  为什么只计算 y_pred 不会自动改变模型参数？
计算预测值只需要参数的当前值，是纯粹的函数求值

## 4. Important Lessons

- Why backward() is necessary:
- Why optimizer.step() is necessary:
- Why Adam is not always better than SGD:

## 5. Next Week Focus

- Training Loop
- Dataset / DataLoader
- Embedding
- Activation / Dropout
- LayerNorm
- GPU / Checkpoint


## 问题：从输入数据到模型参数更新，一次神经网络训练迭代涉及哪些关键步骤？
1. Forward Pass：模型根据输入和当前参数计算预测结果。
2. Calculate Loss：计算预测值与真实标签之间的误差。
3. Backward Pass：通过自动微分计算各个参数的梯度。
4. Optimizer Step：根据梯度和优化算法更新模型参数。
5. Zero Grad：清除上一次迭代的梯度，准备下一次计算。