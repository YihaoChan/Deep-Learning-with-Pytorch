"""
变分自编码器生成图片
"""
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os


class Flatten(nn.Module):
    def forward(self, input):
        # [batch_size, 1, 28, 28] -> [batch_size, 784]
        return input.view(input.size(0), 784)


class UnFlatten(nn.Module):
    def forward(self, input):
        # [batch_size, 784] -> [batch_size, 1, 28, 28]
        return input.view(input.size(0), 1, 28, 28)


class VariationalAutoEncoder(nn.Module):
    def __init__(self):
        super(VariationalAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            Flatten(),  # [batch_size, 1, 28, 28] -> [batch_size, 784]
            nn.Linear(784, 256),  # [batch_size, 784] -> [batch_size, 256]
            nn.ReLU()
        )

        self.fc_mean = nn.Linear(256, 20)  # 均值：-> [batch_size, 20]
        self.fc_std = nn.Linear(256, 20)  # 标准差：-> [batch_size, 20]

        self.decoder = nn.Sequential(
            nn.Linear(20, 256),  # [batch_size, 20] -> [batch_size, 256]
            nn.ReLU(),
            nn.Linear(256, 784),  # [batch_size, 256] -> [batch_size, 784]
            UnFlatten()  # [batch_size, 784] -> [batch_size, 1, 28, 28]
        )

    def reparametrization(self, mu, sigma):
        """
        对分布内的标准差与采样自高斯分布的变量进行圈乘，并与均值加和，计算潜在变量
        code = mean + std * epsilon, 其中 epsilon ~ N(0, 1)

        :param mu: 均值
        :param sigma: 标准差
        :return: 潜在变量
        """

        # 采样自高斯分布，维度需与标准差一致
        eps = torch.randn_like(sigma)

        # 标准差和采样自高斯分布进行圈乘，每个像素点的元素分别相乘，所以后续计算loss时要除以像素和batch
        z = mu + sigma * eps  # [batch_size, 20]

        return z

    def forward(self, x):
        """
        :return: out 用于计算重构图片和原始图片之间的误差
        :return: mu, sigma 用于计算KL散度
        """

        # 编码器：输入图片 -> 展开成784维度-> 降维至256
        h = self.encoder(x)

        # bottleneck：256维度 -> 20维度的均值和标准差
        mu, sigma = self.fc_mean(h), self.fc_std(h)

        # 参数重构，组合成潜在变量
        z = self.reparametrization(mu, sigma)

        # 解码器：20维度 -> 784维度 -> 原始图片大小
        out = self.decoder(z)

        return out, mu, sigma


def get_dataloader(train=True, batch_size=64):
    transform_fn = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = torchvision.datasets.MNIST(
        root='./dataset/mnist_data',
        transform=transform_fn,
        train=train,
        download=True
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True
    )

    return dataloader


def cal_kl_divergence(mu, sigma, batch):
    """
    当计算当前分布和高斯分布的KL散度时，计算公式为：½ * (σ^2 + μ^2 - log(σ^2) - 1)
    :param mu: 均值
    :param sigma: 标准差
    :param batch: batch_size
    :return: KL散度
    """
    return 0.5 * torch.sum(
        torch.pow(mu, 2) +
        torch.pow(sigma, 2) -
        torch.log(1e-8 + torch.pow(sigma, 2)) -  # 如果x接近于0，log为趋向负无穷，会导致KL很小(不加小常数会出现nan)
        1
    ) / (batch * 28 * 28)


def train(net, num_epochs, batch_size, optimizer, device):
    net.train()

    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        for batch_idx, (data, _) in enumerate(get_dataloader(train=True, batch_size=batch_size)):
            data = data.to(device)

            output, mu, sigma = net(data)
            output = output.to(device)

            loss_reconstruct = criterion(output, data)
            kl_divergence = cal_kl_divergence(mu, sigma, batch_size)

            loss = loss_reconstruct + kl_divergence

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print('epoch: {}, batch: {}, loss: {}'.format(
                    epoch, batch_idx, loss.item()
                ))

        gen_img = output.cpu().data  # 铺开成64张小图拼在一起，每一张1通道及28*28大小
        save_image(gen_img, './img/gen_img-{}.png'.format(epoch + 1))


if __name__ == '__main__':
    if not os.path.exists('./img'):
        os.mkdir('./img')

    if torch.cuda.is_available():
        dev = torch.device("cuda:0")
        torch.cuda.manual_seed(7)
    else:
        dev = torch.device("cpu")
        torch.manual_seed(7)

    vae = VariationalAutoEncoder().to(dev)

    optimizer_vae = optim.Adam(vae.parameters())

    train(net=vae, num_epochs=20, batch_size=64, optimizer=optimizer_vae, device=dev)
