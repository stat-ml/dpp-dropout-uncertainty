import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.models import vgg19, resnet18, resnet34
from torchvision import transforms

from catalyst.dl import SupervisedRunner
from catalyst.dl.callbacks import AccuracyCallback, EarlyStoppingCallback
from catalyst.utils import set_global_seed

# 1. resnet-18 - и посмотреть результат
# 2. Помогут ли аугментации
# 3. Более модные архитектуры
# 4. Сравнить с кагглом


def load_data():
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    batch_size = 512


    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transform)

    # train_set, valid_set = train_test_split(train_set, test_size=5000)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                              shuffle=True, num_workers=8)

    valid_loader =[]
    # valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size,
    #                                            shuffle=False, num_workers=8)

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=test_transform)
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


def get_model(model_name):
    if model_name == 'vgg':
        return vgg19(pretrained=False)
    elif model_name in ['resnet_18', 'resnet']:
        return resnet34(pretrained=True)


def train(train_loader, valid_loader, test_loader):
    MODEL_NAME = 'resnet'
    model = get_model(MODEL_NAME)
    logdir = f'logs/best_cifar_{MODEL_NAME}'

    loaders = {
        'train': train_loader,
        'valid': test_loader
    }

    runner = SupervisedRunner()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, weight_decay=5e-4, momentum=0.9)
    callbacks = [AccuracyCallback(num_classes=10), EarlyStoppingCallback(100, metric='accuracy01', minimize=False)]

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, factor=0.5, patience=3
    # )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250], gamma=0.1)
    runner.train(
        model=model, criterion=criterion, optimizer=optimizer, loaders=loaders,
        logdir=logdir, num_epochs=400, verbose=True, scheduler=scheduler,
        callbacks=callbacks
    )


    # predictions, labels = [], []
    # model.cuda()
    #
    # for x_batch, y_batch in test_loader:
    #     x_batch = x_batch.cuda()
    #     with torch.no_grad():
    #         preds = torch.argmax(model(x_batch), dim=-1)
    #     predictions.append(preds.cpu().numpy())
    #     labels.append(y_batch.cpu().numpy())
    # print(
    #     'Test accuracy',
    #     accuracy_score(np.concatenate(predictions), np.concatenate(labels))
    # )


if __name__ == '__main__':
    train_loader, valid_loader, test_loader = load_data()
    model = train(train_loader, valid_loader, test_loader)
