from time import time
import os
import os.path as path
import numpy as np
from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# start: 20
# add: 100x10
# val: 50
# test:10_000
# pool: the rest
epochs = 1000
base_patience = 50


def train_model(model, x_train, y_train, x_val, y_val):
    # for e in range(epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-2)

    x_batch, y_batch = torch.Tensor(x_train), torch.LongTensor(y_train)
    x_v_batch = torch.Tensor(x_val)

    train_acc = []
    val_acc = []
    best_accuracy, patience = 0, base_patience
    for e in range(epochs):
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_acc.append(accuracy_score(y_train, np.argmax(outputs.detach(), axis=-1)))
        with torch.no_grad():
            preds = model(x_v_batch)
            acc = accuracy_score(y_val, np.argmax(preds, axis=-1))
            val_acc.append(acc)
            if acc > best_accuracy:
                best_accuracy = acc
                patience = base_patience
            else:
                patience -= 1
                if patience == 0:
                    break

    # plt.plot(train_acc)
    # plt.plot(val_acc)
    # plt.show()


def update_sets(x_train, y_train, x_pool, y_pool, scores):
    pass


def active_train():
    saver = DataSaver('data/.tmp')

    ## Uncomment to load data first time
    # x, y = fetch_openml("mnist_784", return_X_y=True, cache=True)
    # y = y.astype(np.long)
    # saver.save(x, y, 'mnist')

    x, y = saver.load('mnist')

    x = (x.reshape((-1, 1, 28, 28))) / 256

    x_train, x, y_train, y = train_test_split(x, y, stratify=y, train_size=20)
    x_val, x, y_val, y = train_test_split(x, y, stratify=y, train_size=50)
    x_test, x_pool, y_test, y_pool = train_test_split(x, y, stratify=y, train_size=10_000)

    t = time()


    x_t_batch = torch.Tensor(x_test)
    model = CNN()
    results = []
    for rep in range(5):
        print(rep)
        train_model(model, x_train, y_train, x_val, y_val)
        scores = np.random.random(y_pool.shape)
        update_sets(x_train, y_train, x_pool, y_pool, scores)

        with torch.no_grad():
            results.append(accuracy_score(y_test, np.argmax(model(x_t_batch), axis=-1)))

    print(time() - t)
    plt.plot(results)
    plt.show()





class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(2)
        self.dropout_1 = nn.Dropout(0.25)
        self.linear_1 = nn.Linear(14*14*64, 256)
        self.dropout_2 = nn.Dropout(0.5)
        self.linear_2 = nn.Linear(256, 10)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = x.view(-1, 14*14*64)
        x = F.relu(self.linear_1(self.dropout_1(x)))
        x = self.linear_2(self.dropout_2(x))
        return x


class DataSaver:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def save(self, x, y, name=""):
        if not (path.exists(self.data_dir)):
            os.makedirs(self.data_dir)
        np.save(path.join(self.data_dir, f"x_{name}.npy"), x)
        np.save(path.join(self.data_dir, f"y_{name}.npy"), y)

    def load(self, name=""):
        x_path = path.join(self.data_dir, f"x_{name}.npy")
        y_path = path.join(self.data_dir, f"y_{name}.npy")
        assert path.exists(x_path) and path.exists(y_path)
        x_set = np.load(x_path, allow_pickle=True)
        y_set = np.load(y_path, allow_pickle=True)
        return x_set, y_set


if __name__ == '__main__':
    active_train()


#
# model.add(Conv2D(32, kernel_size=(3, 3),
#                  activation='relu',
#                  input_shape=input_shape))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))