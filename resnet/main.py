import torch
import torch.nn as nn

"""
expansion: 用于匹配输入和输出的通道数
"""


class BasicBlock(nn.Module):
    expansion = 1  # Basic Block没有升维

    def __init__(self, in_channels, out_channels, stride):
        """
        :param in_channels: 原始输入通道数
        :param out_channels: 64
        """
        super(BasicBlock, self).__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,  # 64
                      kernel_size=3,  # 3*3
                      stride=stride,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels,
                      out_channels=BasicBlock.expansion * out_channels,  # 64
                      kernel_size=3,  # 3*3
                      # shortcut只做了一次stride，所以这里不能再做stride了，因为上面做过了，否则维度就不匹配
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(BasicBlock.expansion * out_channels)
            # 加完再一起ReLU
        )

        self.shortcut = nn.Sequential()

        # 如果stride不为1，那么shortcut和多层网络的输出维度就会不一致，就要对x进行1*1卷积以使维度匹配
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=BasicBlock.expansion * out_channels,  # 64
                          kernel_size=1,  # 1*1用于变换维度
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(BasicBlock.expansion * out_channels)
                # 加完再一起ReLU
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class BottleneckBlock(nn.Module):
    expansion = 4  # 64升到256

    def __init__(self, in_channels, out_channels, stride=1):
        """
        :param in_channels: 原始输入通道数
        :param out_channels: 64
        """
        super(BottleneckBlock, self).__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,  # 64
                      kernel_size=1,  # 1*1
                      # 在3*3做stride，这里不能做
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,  # 64
                      kernel_size=3,  # 3*3
                      stride=stride,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels,
                      out_channels=BottleneckBlock.expansion * out_channels,  # 升到256
                      kernel_size=1,  # 1*1
                      # 在3*3做stride，这里不能做
                      bias=False),
            nn.BatchNorm2d(BottleneckBlock.expansion * out_channels)
            # 加完再一起ReLU
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != BottleneckBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=BottleneckBlock.expansion * out_channels,  # 256
                          kernel_size=1,  # 1*1用于变换维度
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(BottleneckBlock.expansion * out_channels)
                # 加完再一起ReLU
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self, block_type, num_block, n_classes=10):
        """
        :param block_type: basic block or bottleneck block
        :param num_block: 每一个conv_x的残差块要叠几个，即论文中的×3/×4/×6/×3
        """
        super(ResNet, self).__init__()
        self.n_classes = n_classes
        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,  # RGB
                      out_channels=64,
                      kernel_size=7,  # 7*7
                      stride=2,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_x = self._make_layer(block_type, 64, num_block[0], stride=1)
        self.conv3_x = self._make_layer(block_type, 128, num_block[1], stride=2)
        self.conv4_x = self._make_layer(block_type, 256, num_block[2], stride=2)
        self.conv5_x = self._make_layer(block_type, 512, num_block[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))  # 自适应算法能够自动帮助我们计算核的大小和每次移动的步长
        self.fc = nn.Linear(512 * block_type.expansion, n_classes)

    def _make_layer(self, block_type, out_channels, num_block, stride):
        strides = [stride] + [1] * (num_block - 1)
        layers = []

        for stride in strides:
            layers.append(block_type(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block_type.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avg_pool(x)
        x = x.view(x.shape[0], -1)
        output = self.fc(x)
        return output


def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def resnet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def resnet50():
    return ResNet(BottleneckBlock, [3, 4, 6, 3])


def resnet101():
    return ResNet(BottleneckBlock, [3, 4, 23, 3])


def resnet152():
    return ResNet(BottleneckBlock, [3, 8, 36, 3])


if __name__ == '__main__':
    batch_size = 64
    channels = 3
    height = 100
    width = 80
    x = torch.rand(size=(batch_size, channels, height, width))
    model = resnet34()
    output = model(x)
    print(output.shape)
