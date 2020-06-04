import torch
from torch import nn
from torchvision.models.resnet import ResNet, BasicBlock, conv3x3, conv1x1, Bottleneck
from torch.hub import load_state_dict_from_url

from alpaca.model.cnn import AnotherConv, SimpleConv

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
}


class StrongConv(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        base = 16
        self.conv = nn.Sequential(
            nn.Conv2d(3, base, 3, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.CELU(),
            nn.Conv2d(base, base, 3, padding=1, bias=False),
            nn.CELU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            nn.Conv2d(base, 2*base, 3, padding=1, bias=False),
            nn.BatchNorm2d(2*base),
            nn.CELU(),
            nn.Conv2d(2 * base, 2 * base, 3, padding=1, bias=False),
            nn.CELU(),
            nn.Dropout2d(0.3),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(2*base, 4*base, 3, padding=1, bias=False),
            nn.BatchNorm2d(4*base),
            nn.CELU(),
            nn.Conv2d(4*base, 4*base, 3, padding=1, bias=False),
            nn.CELU(),
            nn.Dropout2d(0.4),
            nn.MaxPool2d(2, 2),
        )
        self.linear_size = 8 * 8 * base
        self.linear = nn.Sequential(
            nn.Linear(self.linear_size, 8*base),
            nn.CELU(),
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(8*base, 10)

    def forward(self, x, dropout_rate=0.5, dropout_mask=None):
        x = self.conv(x)
        x = x.reshape(-1, self.linear_size)
        x = self.linear(x)
        if dropout_mask is None:
            x = self.dropout(x)
        else:
            x = x * dropout_mask(x, dropout_rate, 0)
        return self.fc(x)


class ResNetMasked(ResNet):
    def forward(self, x, dropout_rate=0.3, dropout_mask=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        import ipdb; ipdb.set_trace()
        if dropout_mask is not None:
            x = x * dropout_mask(x, dropout_rate, 0)
        else:
            x = self.dropout(x)
        x = self.fc(x)

        return x


def resnet_masked(pretrained=True, dropout_rate=0.3):
    base = resnet18(pretrained=pretrained)
    base.dropout = nn.Dropout(dropout_rate)
    base.fc = nn.Linear(512, 10)

    return base


def resnet_dropout(pretrained=True, dropout_rate=0.3):
    base = resnet18(pretrained=pretrained)
    base.dropout = nn.Dropout(dropout_rate)
    # base.fc = nn.Linear(512, 10)

    return base



def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNetMasked(block, layers, **kwargs)

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)
