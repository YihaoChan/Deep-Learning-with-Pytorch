import torch
import torch.nn as nn
import matplotlib.pyplot as plt

''' y = 3 * x + 0.8 '''

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# 输入数据x和对应标签y
data = torch.rand([500, 1]).to(device)
label = 3 * data + 0.8


# 线性回归就是一个不加激活函数的全连接层
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)

        return out


input_dim = 1
output_dim = 1

model = LinearRegressionModel(input_dim, output_dim)

# CPU / GPU
model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

criterion = nn.MSELoss()

epochs = 5000
for epoch in range(epochs):
    epoch += 1

    # 模型计算结果
    outputs = model(data)

    # 计算损失
    loss = criterion(outputs, label)

    # 梯度清零
    optimizer.zero_grad()

    # 反向传播
    loss.backward()

    # 权重更新
    optimizer.step()

    if epoch % 50 == 0:
        param = list(model.parameters())
        print('loss {}, w {}, b {}'.format(loss.item(), param[0].item(), param[1].item()))

# 测试
model.eval()
prediction = model(data).detach().cpu().numpy()

# 用真实数据和模型预测数据相比较，并计算误差
error = (label.detach().cpu().numpy() - prediction) ** 2

print(error.mean())

torch.save(model, 'model.pkl')

restore_model = torch.load('model.pkl')

print(restore_model.state_dict())

# 真实数据散点图
plt.scatter(data.detach().cpu().numpy(), label.detach().cpu().numpy(), c='r')

# 预测数据图
plt.plot(data.detach().cpu().numpy(), prediction)

plt.show()
