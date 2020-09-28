import numpy as np
import torch
import torchvision
from torchvision import transforms
from torchvision.models import vgg16, resnet18
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



from catalyst.dl import SupervisedRunner
from catalyst.dl.callbacks import AccuracyCallback, EarlyStoppingCallback
from catalyst.utils import set_global_seed

# 1. resnet-18 - и посмотреть результат
# 2. Помогут ли аугментации
# 3. Более модные архитектуры
# 4. Сравнить с кагглом


def load_data():
    transform = transforms.Compose( [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    batch_size = 64

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

    train_set, valid_set = train_test_split(train_set, test_size=5000)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                              shuffle=True, num_workers=8)

    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size,
                                               shuffle=False, num_workers=8)

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                             shuffle=False, num_workers=8)


    return train_loader, valid_loader, test_loader


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def train(train_loader, valid_loader, test_loader):
    model = vgg16(pretrained=True)
    logdir = 'logs/best_cifar'

    loaders = {
        'train': train_loader,
        'valid': valid_loader,
        'test': test_loader
    }

    runner = SupervisedRunner()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    callbacks = [AccuracyCallback(num_classes=10), EarlyStoppingCallback(5)]
    runner.train(
        model=model, criterion=criterion, optimizer=optimizer, loaders=loaders,
        logdir=logdir, num_epochs=50, verbose=True,
        callbacks=callbacks
    )


if __name__ == '__main__':
    train_loader, valid_loader, test_loader = load_data()
    model = train(train_loader, valid_loader, test_loader)

