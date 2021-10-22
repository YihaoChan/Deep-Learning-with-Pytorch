"""
标准自编码器生成图片
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import os
from torch.utils.data import DataLoader
from torchvision.utils import save_image


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 20)
        )

        self.decoder = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 784)
        )

    def forward(self, x):
        x_batch_size = x.size(0)

        x = x.view(x_batch_size, 784)  # [batch_size, 1, 28, 28] -> [batch_size, 784]

        x = self.encoder(x)  # [batch_size, 784] -> [batch_size, 20]

        x = self.decoder(x)  # [batch_size, 20] -> [batch_size, 784]

        return x


def get_dataloader(train=True, batch_size=64):
    transform_fn = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = torchvision.datasets.MNIST(
        root='./dataset/mnist_data',
        transform=transform_fn,
        download=True,
        train=train
    )

    dataloader = DataLoader(
        dataset=dataset,
        shuffle=True,
        batch_size=batch_size
    )

    return dataloader


def train(net, num_epochs, batch_size, optimizer, device):
    net.train()

    train_dataloader = get_dataloader(train=True, batch_size=batch_size)

    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        for batch_idx, (data, _) in enumerate(train_dataloader):
            num_img = data.size(0)
            data = data.view(num_img, -1).to(device)

            output = net(data).to(device)

            loss = criterion(output, data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print('epoch: {}, batch: {}, loss: {}'.format(
                    epoch, batch_idx, loss.item()
                ))

        gen_img = output.cpu().data.view(-1, 1, 28, 28)  # 铺开成64张小图拼在一起，每一张1通道及28*28大小
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

    ae = AutoEncoder().to(dev)

    optimizer_ae = optim.Adam(ae.parameters())

    train(net=ae, num_epochs=20, batch_size=64, optimizer=optimizer_ae, device=dev)
