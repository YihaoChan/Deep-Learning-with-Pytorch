"""
生成图片
"""
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import os
from torchvision.utils import save_image


def to_img(x):
    """
    根据随机向量生成图片
    1. 首先给出一个简单的高维的正态分布的噪声向量，之后通过仿射变换(即xw+b)将其映射到一个更高的维度；
    2. 然后将它重新排列成一个矩形，接着进行卷积、池化、激活函数等处理；
    3. 最后得到了一个与我们输入图片大小一模一样的噪音矩阵，这就是假的图片
    """
    out = 0.5 * x + 1
    out = torch.clamp(out, 0, 1)  # torch.clamp: 把张量压缩到(a, b)区间之间
    out = out.view(-1, 1, 28, 28)  # 和MNIST图片一样：黑白(单通道)，28 * 28，把64张小图拼在一张图中
    return out


def get_dataloader(train=True, batch_size=64):
    """
    批量加载数据
    """
    transform_fn = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = torchvision.datasets.MNIST(
        root='./dataset/mnist_data',
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


class Discrimator(nn.Module):
    def __init__(self):
        super(Discrimator, self).__init__()
        self.dis = nn.Sequential(
            # 对送入判别器的图片，转为一个一维数据，和全0向量或全1向量进行loss
            nn.Linear(28 * 28, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()  # 转为0 - 1之间的概率进行二分类
        )

    def forward(self, x):
        return self.dis(x)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            # 根据输入的随机的100维向量，最终生成28 * 28的图片
            nn.Linear(100, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 28 * 28)
        )

    def forward(self, x):
        return self.gen(x)


def train(num_epochs, generator, discriminator, optimizer_gen, optimizer_dis, device):
    generator.train()
    discriminator.train()

    criterion = nn.BCELoss()

    train_dataloader = get_dataloader(train=True, batch_size=64)

    for epoch in range(num_epochs):
        for batch_idx, (img, _) in enumerate(train_dataloader):
            num_img = img.size(0)
            img = img.view(num_img, -1)

            # 真实数据
            real_data = img.to(device)

            # 真实数据标签全1，假数据标签全0。PS.维度[64]需要转为[64, 1]
            real_label = torch.ones(num_img).unsqueeze(dim=1).to(device)
            fake_label = torch.zeros(num_img).unsqueeze(dim=1).to(device)

            """
            命名：dp:训练判别器阶段；gp:训练生成器阶段
            """

            """训练判别器"""
            dis_real_data = discriminator(real_data).to(device)  # 训练判别器阶段，判别器对于真实数据的输出
            loss_real_dp = criterion(dis_real_data, real_label)  # 训练判别器阶段，判别器对于真实数据的loss

            z_dp = torch.randn(num_img, 100).to(device)  # 训练判别器阶段，生成器的随机输入噪声
            fake_data_dp = generator(z_dp).to(device)  # 训练判别器阶段，生成器生成的假数据
            dis_fake_data_dp = discriminator(fake_data_dp).to(device)  # 训练判别器阶段，判别器对于假数据的输出
            loss_fake_dp = criterion(dis_fake_data_dp, fake_label)  # 训练判别器阶段，判别器对于假数据的loss

            loss_dp = loss_real_dp + loss_fake_dp  # 判别器的总loss

            # 当前阶段训练判别器，根据判别器的loss，对判别器进行参数更新
            optimizer_dis.zero_grad()
            loss_dp.backward()
            optimizer_dis.step()

            """训练生成器"""
            z_gp = torch.randn(num_img, 100).to(device)  # 训练生成器阶段，生成器的随机输入噪声
            fake_data_gp = generator(z_gp).to(device)  # 训练生成器阶段，生成器生成的假数据
            dis_fake_data_gp = discriminator(fake_data_gp).to(device)  # 训练生成器阶段，判别器对于假数据的输出
            loss_gp = criterion(dis_fake_data_gp, real_label)  # 训练生成器阶段，判别器对于假数据的loss

            # 当前阶段训练生成器，根据判别器的loss，对生成器进行参数更新
            optimizer_gen.zero_grad()
            loss_gp.backward()
            optimizer_gen.step()

            if batch_idx % 100 == 0:
                print('epoch: {}, batch: {}, d_loss: {}, g_loss: {}'.format(
                    epoch, batch_idx, loss_dp, loss_gp
                ))

                torch.save(generator.state_dict(), './model/generator.pkl')
                torch.save(discriminator.state_dict(), './model/discriminator.pkl')

                torch.save(optimizer_gen.state_dict(), './model/optimizer_gen.pkl')
                torch.save(optimizer_dis.state_dict(), './model/optimizer_dis.pkl')

        fake_images = to_img(fake_data_gp.cpu().data)  # 生成器生成的假图片
        save_image(fake_images, './img/fake_images-{}.png'.format(epoch + 1))


if __name__ == '__main__':
    if not os.path.exists('./model'):
        os.mkdir('./model')

    if not os.path.exists('./img'):
        os.mkdir('./img')

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    G = Generator().to(dev)
    D = Discrimator().to(dev)

    optim_g = optim.Adam(G.parameters())
    optim_d = optim.Adam(D.parameters())

    train(20, G, D, optim_g, optim_d, dev)
