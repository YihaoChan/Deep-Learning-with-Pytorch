import torch
from matplotlib import pyplot as plt

# 真实数据
x = torch.rand([500, 1])
y_true = 3 * x + 0.8

# 权重初始化
w = torch.rand([1, 1], requires_grad=True)
b = torch.tensor(1, requires_grad=True, dtype=torch.float32)

# 学习率
learning_rate = 0.01

# 训练
for i in range(5000):
    # 用新的w和b去计算模型输出
    y_predict = torch.matmul(x, w) + b

    # 梯度清零
    if w.grad is not None:
        w.grad.data.zero_()
    if b.grad is not None:
        b.grad.data.zero_()

    # 反向传播
    loss = (y_predict - y_true).pow(2).mean()
    loss.backward()

    # 权重更新
    w.data = w.data - w.grad * learning_rate
    b.data = b.data - b.grad * learning_rate

    print("w, b, loss:", w.item(), b.item(), loss.item())

plt.figure(figsize=(20, 8))

# 真实数据
plt.scatter(x.numpy().reshape(-1), y_true.numpy().reshape(-1))

# 预测数据
y_predict = torch.matmul(x, w) + b
plt.plot(x.numpy().reshape(-1), y_predict.detach().numpy().reshape(-1), c='r')

plt.show()
