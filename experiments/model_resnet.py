'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
from random import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from dropblock import DropBlock2D



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.elu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.elu(out)
        return out



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.elu(self.bn1(self.conv1(x)))
        out = F.elu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.elu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, dropout_rate=0.2, input_size=3):
        super().__init__()
        self.in_planes = 64

        self.dropout_rate = dropout_rate

        self.conv1 = nn.Conv2d(input_size, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, dropout_rate=0.2, dropout_mask=None):
        # out = F.elu(self.bn1(self.conv1(x)))
        out = self.maxpool(F.elu(self.bn1(self.conv1(x))))
        out = self.layer1(out)

        if dropout_mask is not None:
            # out = out * dropout_mask(out, dropout_rate, 0)
            out = F.dropout(out, p=self.dropout_rate)
        else:
            out = self.dropout(out)

        out = self.layer2(out)
        if dropout_mask is not None:
            # out = out * dropout_mask(out, dropout_rate, 1)
            out = F.dropout(out, p=self.dropout_rate)
        else:
            out = self.dropout(out)

        out = self.layer3(out)
        if dropout_mask is not None:
            # out = out * dropout_mask(out, dropout_rate, 2)
            out = F.dropout(out, p=self.dropout_rate)
        else:
            out = self.dropout(out)

        out = self.layer4(out)
        out = out.view(out.size(0), -1)
        if dropout_mask is not None:
            # out = out * dropout_mask(out, dropout_rate, 3)
            out = F.dropout(out, p=self.dropout_rate)
        else:
            out = self.dropout(out)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2], input_size=1, dropout_rate=0.1)


def ResNet34(dropout_rate):
    return ResNet(BasicBlock, [3, 4, 6, 3], dropout_rate=dropout_rate)


def ResNet50(input_size=3, dropout_rate=0.15):
    return ResNet(Bottleneck, [3, 4, 6, 3], input_size=input_size, dropout_rate=dropout_rate)


def ResNet50_dropblock(input_size=3, dropout_rate=0.15, block_size=3):
    return ResNetDropblock(Bottleneck, [3, 4, 6, 3], input_size=input_size, dropout_rate=dropout_rate)

def ResNet50_dropchannel(input_size=3, dropout_rate=0.15, block_size=3):
    return ResNetDropchannel(Bottleneck, [3, 4, 6, 3], input_size=input_size, dropout_rate=dropout_rate)


def ResNet50_droplayer(input_size=3, dropout_rate=0.15, block_size=3):
    return ResNetDroplayer([3, 4, 6, 3], input_size=input_size, dropout_rate=dropout_rate)


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())



class ResNetDropblock(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, dropout_rate=0.2, input_size=3, block_size=3):
        super().__init__()
        self.in_planes = 64

        self.dropout_rate = dropout_rate

        self.conv1 = nn.Conv2d(input_size, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.drop_1 = DropBlock2D(block_size=block_size, drop_prob=dropout_rate)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.drop_2 = DropBlock2D(block_size=block_size, drop_prob=dropout_rate)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.drop_3 = DropBlock2D(block_size=block_size, drop_prob=dropout_rate)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.drop_4 = DropBlock2D(block_size=block_size, drop_prob=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _cached_drop(self, x, drop_layer, dropout_mask):
        if dropout_mask is None:
            return drop_layer(x)
        else:
            state = drop_layer.training
            drop_layer.training = True
            out = drop_layer(x)
            drop_layer.training = state
            return out

    def forward(self, x, dropout_rate=0.2, dropout_mask=None, debug=False):
        # out = F.elu(self.bn1(self.conv1(x)))
        out = self.maxpool(F.elu(self.bn1(self.conv1(x))))
        out = self.layer1(out)
        if debug:
            import ipdb; ipdb.set_trace()
        out = self._cached_drop(out, self.drop_1, dropout_mask)

        out = self.layer2(out)
        out = self._cached_drop(out, self.drop_2, dropout_mask)

        out = self.layer3(out)
        out = self._cached_drop(out, self.drop_3, dropout_mask)

        out = self.layer4(out)
        out = self._cached_drop(out, self.drop_4, dropout_mask)
        out = out.view(out.size(0), -1)
        self.embedding = out

        out = self.linear(out)
        return out

# test()

class ResNetDropchannel(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, dropout_rate=0.2, input_size=3, block_size=3):
        super().__init__()
        self.in_planes = 64

        self.dropout_rate = dropout_rate

        self.conv1 = nn.Conv2d(input_size, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.drop_1 = nn.Dropout2d(p=dropout_rate)

        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.drop_2 = nn.Dropout2d(p=dropout_rate)

        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.drop_3 = nn.Dropout2d(p=dropout_rate)

        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.drop_4 = nn.Dropout2d(p=dropout_rate)

        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _cached_drop(self, x, drop_layer, dropout_mask):
        if dropout_mask is None:
            return drop_layer(x)
        else:
            state = drop_layer.training
            drop_layer.training = True
            out = drop_layer(x)
            drop_layer.training = state
            return out

    def forward(self, x, dropout_rate=0.2, dropout_mask=None, debug=False):
        # out = F.elu(self.bn1(self.conv1(x)))
        out = self.maxpool(F.elu(self.bn1(self.conv1(x))))
        out = self.layer1(out)
        if debug:
            import ipdb; ipdb.set_trace()
        out = self._cached_drop(out, self.drop_1, dropout_mask)

        out = self.layer2(out)
        out = self._cached_drop(out, self.drop_2, dropout_mask)

        out = self.layer3(out)
        out = self._cached_drop(out, self.drop_3, dropout_mask)

        out = self.layer4(out)
        out = self._cached_drop(out, self.drop_4, dropout_mask)
        out = out.view(out.size(0), -1)
        self.embedding = out

        out = self.linear(out)
        return out



class DropLayerBottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1, dropout_rate=0.2):
        super().__init__()

        self.dropout_rate = dropout_rate

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        shortcut = self.shortcut(x)
        if random() > self.dropout_rate:
            out = F.elu(self.bn1(self.conv1(x)))
            out = F.elu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
        else:
            out = torch.zeros_like(shortcut).to(shortcut.device)
        out += shortcut
        out = F.elu(out)
        return out



class ResNetDroplayer(nn.Module):
    def __init__(self, num_blocks, num_classes=10, dropout_rate=0.2, input_size=3, block_size=3):
        super().__init__()
        self.in_planes = 64
        block = DropLayerBottleneck

        self.dropout_rate = dropout_rate

        self.conv1 = nn.Conv2d(input_size, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.drop_1 = nn.Dropout2d(p=dropout_rate)

        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.drop_2 = nn.Dropout2d(p=dropout_rate)

        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.drop_3 = nn.Dropout2d(p=dropout_rate)

        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.drop_4 = nn.Dropout2d(p=dropout_rate)

        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _cached_drop(self, x, drop_layer, dropout_mask):
        if dropout_mask is None:
            return drop_layer(x)
        else:
            state = drop_layer.training
            drop_layer.training = True
            out = drop_layer(x)
            drop_layer.training = state
            return out

    def forward(self, x, dropout_rate=0.2, dropout_mask=None, debug=False):
        # out = F.elu(self.bn1(self.conv1(x)))
        out = self.maxpool(F.elu(self.bn1(self.conv1(x))))
        out = self.layer1(out)
        if debug:
            import ipdb; ipdb.set_trace()
        out = self._cached_drop(out, self.drop_1, dropout_mask)

        out = self.layer2(out)
        out = self._cached_drop(out, self.drop_2, dropout_mask)

        out = self.layer3(out)
        out = self._cached_drop(out, self.drop_3, dropout_mask)

        out = self.layer4(out)
        out = self._cached_drop(out, self.drop_4, dropout_mask)
        out = out.view(out.size(0), -1)

        self.embedding = out

        out = self.linear(out)
        return out