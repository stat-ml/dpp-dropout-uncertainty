from time import time
from copy import deepcopy
import pandas as pd
import seaborn as sns
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
from alpaca.uncertainty_estimator.bald import bald
from catalyst.utils import set_global_seed

# start: 20
# add: 100x10
# val: 50
# test:10_000
# pool: the rest
epochs = 50
base_patience = 50
update_size = 10
forward_passes = 100


def train_model(model, x_train, y_train, x_val, y_val):
    # for e in range(epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-2)

    x_batch, y_batch = torch.Tensor(x_train).cuda(), torch.LongTensor(y_train).cuda()
    x_v_batch = torch.Tensor(x_val).cuda()

    train_acc = []
    val_acc = []
    best_accuracy, patience = 0, base_patience
    for e in range(epochs):
        model.train()
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_acc.append(accuracy_score(y_train, np.argmax(outputs.detach().cpu(), axis=-1)))
        with torch.no_grad():
            model.eval()
            preds = model(x_v_batch)
            acc = accuracy_score(y_val, np.argmax(preds.cpu(), axis=-1))
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
    indx = np.argsort(-scores)[:update_size]
    x_train = np.concatenate((x_train, x_pool[indx]))
    y_train = np.concatenate((y_train, y_pool[indx]))
    x_pool = np.delete(x_pool, indx, axis=0)
    y_pool = np.delete(y_pool, indx)
    return x_train, y_train, x_pool, y_pool


def calc_max_prob_uq(model, x_pool):
    model.eval()
    with torch.no_grad():
        x_tensor = torch.Tensor(x_pool).cuda()
        probabilities = torch.softmax(model(x_tensor), axis=-1)
        max_prob = 1 - torch.max(probabilities, axis=-1)[0]
    return max_prob.cpu().numpy()


def rand_uq(model, x_pool):
    return np.random.random(x_pool.shape[0])


def calc_mc_bald(model, x_pool):
    mcd = np.zeros((x_pool.shape[0], forward_passes, 10))

    model.train()
    x_tensor = torch.Tensor(x_pool).cuda()
    with torch.no_grad():
        for n in range(forward_passes):
            logits = model(x_tensor)
            mcd[:, n, :] = logits.cpu().numpy()

    scores = bald(mcd)
    print(scores.shape)
    print(scores[:11])

    return scores


scores_funcs = {
    'random': rand_uq,
    'max_prob': calc_max_prob_uq,
    'bald': calc_mc_bald
}

def active_train():
    saver = DataSaver('data/.tmp')

    ## Uncomment to load data first time
    # x, y = fetch_openml("mnist_784", return_X_y=True, cache=True)
    # y = y.astype(np.long)
    # saver.save(x, y, 'mnist')

    x, y = saver.load('mnist')

    x = (x.reshape((-1, 1, 28, 28))) / 256

    x_train_init, x, y_train_init, y = train_test_split(x, y, stratify=y, train_size=20)
    x_val, x, y_val, y = train_test_split(x, y, stratify=y, train_size=50)
    x_test, x_pool_init, y_test, y_pool_init = train_test_split(x, y, stratify=y, train_size=10_000)
    x_pool_init, _, y_pool_init, _ = train_test_split(x_pool_init, y_pool_init, train_size=5000)

    x_t_batch = torch.Tensor(x_test).cuda()

    results = []
    init_model = CNN()
    for reps in range(2):
        for ue in ['bald', 'random', 'max_prob']:
            set_global_seed(42 + reps)
            x_train, y_train = np.copy(x_train_init), np.copy(y_train_init)
            x_pool, y_pool = np.copy(x_pool_init), np.copy(y_pool_init)
            model = deepcopy(init_model).cuda()

            for iteration in range(20):
                train_model(model, x_train, y_train, x_val, y_val)
                scores = scores_funcs[ue](model, x_pool)
                x_train, y_train, x_pool, y_pool = update_sets(x_train, y_train, x_pool, y_pool, scores)

                with torch.no_grad():
                    model.eval()
                    acc = accuracy_score(y_test, np.argmax(model(x_t_batch).cpu(), axis=-1))
                    results.append((20 + iteration*10, acc, ue))

    df = pd.DataFrame(results, columns=['Set size', 'Accuracy', 'Method'])
    sns.lineplot('Set size', 'Accuracy', data=df, hue='Method')
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
