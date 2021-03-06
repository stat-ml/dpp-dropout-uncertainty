import random

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd

import torch
from torch import nn

from fastai.vision import (ImageDataBunch, Learner, accuracy)
from fastai.callbacks import EarlyStoppingCallback

from alpaca.model.cnn import AnotherConv, SimpleConv
from alpaca.model.resnet import resnet_masked
from alpaca.active_learning.simple_update import update_set
from deprecated.utils import ImageArrayDS
from visual_datasets import prepare_mnist, prepare_cifar, prepare_svhn


"""
Active learning experiment for computer vision tasks (MNIST, CIFAR, SVHN)
"""


SEED = 43
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.set_device(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = 'cuda'
else:
    device = 'cpu'

experiment_label = 'ht'


def main(config):
    # Load data
    x_set, y_set, x_val, y_val, train_tfms = config['prepare_dataset'](config)

    val_accuracy = []
    for _ in range(config['repeats']):  # more repЕсли кнопочный, то есть вот прямо бабушкофон. Ещё его можно забрать из пассажа на фрунзе.eats for robust results
        # Initial data split
        x_set, x_train_init, y_set, y_train_init = train_test_split(x_set, y_set, test_size=config['start_size'], stratify=y_set)
        _, x_pool_init, _, y_pool_init = train_test_split(x_set, y_set, test_size=config['pool_size'], stratify=y_set)

        # Active learning
        for method in config['methods']:
            print(f"== {method} ==")
            x_pool, y_pool = np.copy(x_pool_init), np.copy(y_pool_init)
            x_train, y_train = np.copy(x_train_init), np.copy(y_train_init)

            model = build_model(config['model_type'])
            accuracies = []

            for i in range(config['steps']):
                print(f"Step {i+1}, train size: {len(x_train)}")

                learner = train_classifier(model, config, x_train, y_train, x_val, y_val, train_tfms)
                accuracies.append(learner.recorder.metrics[-1][0].item())

                if i != config['steps'] - 1:
                    x_pool, x_train, y_pool, y_train = update_set(
                        x_pool, x_train, y_pool, y_train, config['step_size'], method=method, model=model)

            records = list(zip(accuracies, range(len(accuracies)), [method] * len(accuracies)))
            val_accuracy.extend(records)

    # Display results
    try:
        plot_metric(val_accuracy, config)
    except:
        import ipdb; ipdb.set_trace()


def train_classifier(model, config, x_train, y_train, x_val, y_val, train_tfms=None):
    loss_func = torch.nn.CrossEntropyLoss()

    if train_tfms is None:
        train_tfms = []
    train_ds = ImageArrayDS(x_train, y_train, train_tfms)
    val_ds = ImageArrayDS(x_val, y_val)
    data = ImageDataBunch.create(train_ds, val_ds, bs=config['batch_size'])

    callbacks = [partial(EarlyStoppingCallback, min_delta=1e-3, patience=config['patience'])]
    learner = Learner(data, model, metrics=accuracy, loss_func=loss_func, callback_fns=callbacks)
    learner.fit(config['epochs'], config['start_lr'], wd=config['weight_decay'])

    return learner


def plot_metric(metrics, config, title=None):
    plt.figure(figsize=(8, 6))
    default_title = f"Validation accuracy, start size {config['start_size']}, "
    default_title += f"step size {config['step_size']}, model {config['model_type']}"
    title = title or default_title
    plt.title(title)

    df = pd.DataFrame(metrics, columns=['Accuracy', 'Step', 'Method'])
    sns.lineplot('Step', 'Accuracy', hue='Method', data=df)
    # plt.legend(loc='upper left')

    filename = f"{experiment_label}_{config['name']}_{config['model_type']}_{config['start_size']}_{config['step_size']}"
    dir = Path(__file__).parent.absolute() / 'data' / 'al'
    file = dir / filename
    plt.savefig(file)
    df.to_csv(dir / (filename + '.csv'))
    # plt.show()


class Model21(nn.Module):
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


def build_model(model_type):
    if model_type == 'conv':
        model = AnotherConv()
    elif model_type == 'resnet':
        model = resnet_masked(pretrained=True)
    elif model_type == 'simple_conv':
        model = SimpleConv()
    elif model_type == 'strong_conv':
        print('piu!')
        model = Model21()
    return model


config_cifar = {
    'val_size': 10_000,
    'pool_size': 15_000,
    'start_size': 7_000,
    'step_size': 50,
    'steps': 50,
    # 'methods': ['random', 'error_oracle', 'max_entropy', *DEFAULT_MASKS, 'ht_dpp'],
    'methods': ['random', 'error_oracle', 'max_entropy', 'mc_dropout', 'ht_dpp'],
    'epochs': 30,
    'patience': 2,
    'model_type': 'strong_conv',
    'repeats': 3,
    'nn_runs': 100,
    'batch_size': 128,
    'start_lr': 5e-4,
    'weight_decay': 0.0002,
    'prepare_dataset': prepare_cifar,
    'name': 'cifar'
}

config_svhn = deepcopy(config_cifar)
config_svhn.update({
    'prepare_dataset': prepare_svhn,
    'name': 'svhn',
    'model_type': 'strong_conv',
    'repeats': 3,
    'epochs': 30,
    'methods': ['random', 'error_oracle', 'max_entropy', 'mc_dropout', 'ht_dpp']
})

config_mnist = deepcopy(config_cifar)
config_mnist.update({
    'start_size': 100,
    'step_size': 20,
    'model_type': 'simple_conv',
    'methods': ['ht_dpp'],
    'prepare_dataset': prepare_mnist,
    'batch_size': 32,
    'name': 'mnist',
    'steps': 50
})

# configs = [config_mnist, config_cifar, config_svhn]

configs = [config_mnist]

if __name__ == '__main__':
    for config in configs:
        print(config)
        main(config)