from torch import nn
from alpaca.model.cnn import AnotherConv, SimpleConv


class StrongConv(nn.Module):
    def __init__(self):
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
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(8*base, 10)

    def forward(self, x, dropout_rate=0.5, dropout_mask=None):
        x = self.conv(x)
        x = x.reshape(-1, self.linear_size)
        x = self.linear(x)
        if dropout_mask is None:
            x = self.dropout(x)
        else:
            x = x * dropout_mask(x, 0.3, 0)
        return self.fc(x)