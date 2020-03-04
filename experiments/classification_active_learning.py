from functools import partial
from copy import deepcopy
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd

import torch

from fastai.vision import (rand_pad, flip_lr, ImageDataBunch, Learner, accuracy)
from fastai.callbacks import EarlyStoppingCallback

from alpaca.model.cnn import AnotherConv, SimpleConv
from alpaca.model.resnet import resnet_masked
from alpaca.dataloader.builder import build_dataset
from alpaca.uncertainty_estimator.masks import DEFAULT_MASKS
from alpaca.active_learning.simple_update import update_set
from utils.fastai import ImageArrayDS

if torch.cuda.is_available():
    torch.cuda.set_device(1)
    device = 'cuda'
else:
    device = 'cpu'


"""
Active learning experiment for computer vision tasks (MNIST, CIFAR, SVHN)
"""


def main(config):
    # Load data
    x_set, y_set, x_val, y_val, train_tfms = config['prepare_dataset'](config)

    val_accuracy = []
    for _ in range(config['repeats']):  # more repeats for robust results
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
    plot_metric(val_accuracy, config)


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

    filename = f"{config['name']}_{config['model_type']}_{config['start_size']}_{config['step_size']}"
    file = Path('data') / 'al' / filename
    plt.savefig(file)
    df.to_csv(filename + '.csv')
    plt.show()


def build_model(model_type):
    if model_type == 'conv':
        model = AnotherConv()
    elif model_type == 'resnet':
        model = resnet_masked(pretrained=True)
    elif model_type == 'simple_conv':
        model = SimpleConv()
    return model


def prepare_cifar(config):
    dataset = build_dataset('cifar_10', val_size=config['val_size'])
    x_set, y_set = dataset.dataset('train')
    x_val, y_val = dataset.dataset('val')

    shape = (-1, 3, 32, 32)
    x_set = ((x_set - 128) / 128).reshape(shape)
    x_val = ((x_val - 128) / 128).reshape(shape)

    train_tfms = [*rand_pad(4, 32), flip_lr(p=0.5)]  # Transformation to augment images

    return x_set, y_set, x_val, y_val, train_tfms


def prepare_mnist(config):
    dataset = build_dataset('mnist', val_size=config['val_size'])
    x_set, y_set = dataset.dataset('train')
    x_val, y_val = dataset.dataset('val')

    shape = (-1, 1, 28, 28)
    x_set = ((x_set - 128) / 128).reshape(shape)
    x_val = ((x_val - 128) / 128).reshape(shape)

    train_tfms = []

    return x_set, y_set, x_val, y_val, train_tfms


def prepare_svhn(config):
    dataset = build_dataset('svhn', val_size=config['val_size'])
    x_set, y_set = dataset.dataset('train')
    x_val, y_val = dataset.dataset('val')
    y_set[y_set == 10] = 0
    y_val[y_val == 10] = 0

    shape = (-1, 32, 32, 3)
    x_set = ((x_set - 128) / 128).reshape(shape)
    x_val = ((x_val - 128) / 128).reshape(shape)
    x_set = np.rollaxis(x_set, 3, 1)
    x_val = np.rollaxis(x_val, 3, 1)

    train_tfms = [*rand_pad(4, 32), flip_lr(p=0.5)]  # Transformation to augment images

    return x_set, y_set, x_val, y_val, train_tfms



cifar_config = {
    'val_size': 10_000,
    'pool_size': 15_000,
    'start_size': 7_000,
    'step_size': 50,
    'steps': 30,
    'methods': ['random', 'error_oracle', 'max_entropy', *DEFAULT_MASKS],
    'epochs': 30,
    'patience': 2,
    'model_type': 'resnet',
    'repeats': 3,
    'nn_runs': 100,
    'batch_size': 128,
    'start_lr': 5e-4,
    'weight_decay': 0.2,
    'prepare_dataset': prepare_cifar,
    'name': 'cifar'
}

svhn_config = deepcopy(cifar_config)
svhn_config.update({
    'prepare_dataset': prepare_svhn,
    'name': 'svhn'
})

mnist_config = deepcopy(cifar_config)
mnist_config.update({
    'start_size': 100,
    'step_size': 20,
    'model_type': 'simple_conv',
    'prepare_dataset': prepare_mnist,
    'batch_size': 32,
    'name': 'mnist'
})

configs = [mnist_config, cifar_config, svhn_config]

if __name__ == '__main__':
    for config in configs:
        main(config)
