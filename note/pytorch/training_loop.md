## 问题：一个标准 PyTorch Training Loop 包含哪些步骤？为什么需要 zero_grad()？
1. Forward Pass：模型根据输入和当前参数计算预测结果。
2. Calculate Loss：用损失函数计算预测值与真实标签之间的误差。
3. Backward Pass：调用 loss.backward()，通过自动微分计算各参数的梯度并存入 param.grad。
4. Optimizer Step：调用 optimizer.step()，根据梯度和优化算法（SGD / Adam 等）更新模型参数。
5. Zero Grad：调用 optimizer.zero_grad()，清除梯度，准备下一次迭代。

为什么需要 zero_grad()：PyTorch 中参数的梯度默认是累加的（param.grad += 新梯度）。如果不清零，当前轮的梯度会叠加在上一轮的旧梯度上，导致参数更新方向错误、训练发散。每轮迭代开始前必须清零，确保 backward() 计算出的梯度只属于当前 batch。
