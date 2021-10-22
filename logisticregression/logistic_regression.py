"""
MNIST手写数字数据集分类任务
网络结构：
    输入层 - 图片像素为28 * 28，故为784
    第一个隐藏层 - 196
    第二个隐藏层 - 28
    输出层 - 十个类别，故为10
"""

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import os


def get_dataloader(train=True, batch_size=64):
    """
    1. 0.1307和0.3081是mnist数据集的均值和标准差，是数据集提供方计算好的数据，【且传入的为元组格式！】
    2. transforms.ToTensor() 将图片格式的ndarray转换成形状为(通道数, 高, 宽)的Tensor格式，且除以255归一化到[0, 1]之间
       具体实现过程：
        i. img.tobytes() 将图片转化为内存中的存储格式
        ii. torch.BytesStorage.frombuffer(img.tobytes()) 将字节以流的形式输入，转化为一维的张量
        iii. 对张量进行reshape
        iv. 对张量进行transpose
        v. 将张量中的每个元素除以255
        vi. 输出张量
    """
    transform_fn = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = torchvision.datasets.MNIST(
        root='./datasets/mnist_data',
        train=train,
        download=True,
        transform=transform_fn
    )

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True
    )

    return data_loader


class MnistModel(nn.Module):
    def __init__(self, input_dim, hidden_one_dim, hidden_two_dim, output_dim):
        super(MnistModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_one_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_one_dim, hidden_two_dim)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(hidden_two_dim, output_dim)

        self._init_parameters()

    def forward(self, x):
        # 第一个维度为batch_size
        x = x.view(-1, 28 * 28)

        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)

        return x

    def _init_parameters(self):
        """
        权重初始化
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p)
            else:
                nn.init.constant_(p, 0)


model = MnistModel(784, 196, 28, 10)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = optim.Adam(model.parameters())

criterion = nn.CrossEntropyLoss()


def train(epochs):
    train_loader = get_dataloader(train=True, batch_size=64)

    for epoch in range(epochs):
        for batch_idx, (train_data, train_label) in enumerate(train_loader):
            train_data = train_data.to(device)
            train_label = train_label.to(device)

            train_output = model(train_data)

            loss = criterion(train_output, train_label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print('train---epoch: {}, batch: {}, loss: {}'.format(
                    epoch, batch_idx, loss.item()
                ))

                if not os.path.exists('./model/'):
                    os.mkdir('./model')

                torch.save(model.state_dict(), './model/model.pkl')
                torch.save(optimizer.state_dict(), './model/optimizer.pkl')


def evaluate():
    """
    1. 模型的预测结果维度为[batch_size, 10]
       行数表示一个batch共有多少张图片，列数表示共有多少个类别
    2. 测试集的dataloader的标签维度为[batch_size]，表示共有多少张图片的对应真实标签
    3. 每一行的每一个不同元素表示模型对测试集预测的每个数字分别对应的概率，概率最大者可视为预测结果
    4. 对预测结果取每一行的最大值所在的索引，表示取出每一行的预测结果数字，得到维度为[batch_size]的矩阵
    5. 将4所得的矩阵和标签矩阵相比较，统计相等元素个数，即为模型的正确预测结果个数
    """
    test_loader = get_dataloader(train=False, batch_size=64)

    if os.path.exists('./model/model.pkl'):
        model.load_state_dict(torch.load('./model/model.pkl'))
    else:
        print('No model for evaluation.')
        return

    # 统计预测正确个数
    acc_count = 0

    model.eval()

    for (test_data, test_label) in test_loader:
        test_data = test_data.to(device).data
        test_label = test_label.to(device).data

        test_output = model(test_data).data

        loss = criterion(test_output, test_label)

        test_output_result = test_output.argmax(1)

        # torch.eq比较两个张量后返回一个矩阵，元素相等位置处为1，否则为0
        # 之后进行sum求和，就能得到相等元素个数，即正确预测的数量
        acc_count += torch.eq(test_output_result, test_label).sum()

        print('test---loss: {}'.format(loss.item()))

    # 加上.dataset才能得到总共的测试集数据数量
    print('accuracy: {}%'.format(acc_count / len(test_loader.dataset) * 100))


if __name__ == '__main__':
    train(10)

    evaluate()
