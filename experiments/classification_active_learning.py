from argparse import ArgumentParser
from collections import OrderedDict, defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from catalyst.dl import SupervisedRunner
from catalyst.dl.callbacks import AccuracyCallback, EarlyStoppingCallback
from catalyst.utils import set_global_seed

from alpaca.active_learning.simple_update import update_set

from configs import al_config, al_experiments
from utils.visual_datasets import ImageDataset


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('name')
    name = parser.parse_args().name
    config = deepcopy(al_config)
    config.update(al_experiments[name])
    config['name'] = name

    return config


def active_train(config, i):
    set_global_seed(i + 42)
    # Load data
    x_set, y_set, x_val, y_val, train_tfms = config['prepare_dataset'](config)
    print(x_set.shape)

    # Initial data split
    x_set, x_train_init, y_set, y_train_init = train_test_split(x_set, y_set, test_size=config['start_size'], stratify=y_set)
    _, x_pool_init, _, y_pool_init = train_test_split(x_set, y_set, test_size=config['pool_size'], stratify=y_set)

    model_init = config['model_class']().double()
    val_accuracy = []
    # Active learning
    for method in config['methods']:
        model = deepcopy(model_init)
        print(f"== {method} ==")
        logdir = f"logs/al/{config['name']}_{method}_{i}"
        x_pool, y_pool = np.copy(x_pool_init), np.copy(y_pool_init)
        x_train, y_train = np.copy(x_train_init), np.copy(y_train_init)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())
        callbacks = [AccuracyCallback(num_classes=10), EarlyStoppingCallback(config['patience'])]
        runner = SupervisedRunner()

        accuracies = []

        for i in range(config['steps']):
            print(f"Step {i+1}, train size: {len(x_train)}")

            loaders = get_loaders(x_train, y_train, x_val, y_val, config['batch_size'], train_tfms)

            runner.train(
                model, criterion, optimizer, loaders,
                logdir=logdir, num_epochs=config['epochs'], verbose=False,
                callbacks=callbacks
            )

            accuracies.append(runner.state.best_valid_metrics['accuracy01'])

            if i != config['steps'] - 1:
                samples = next(iter(loader(x_pool, y_pool, batch_size=len(x_pool))))[0].cuda()
                print('Samples!', samples.shape)
                x_pool, x_train, y_pool, y_train = update_set(
                    x_pool, x_train, y_pool, y_train, config['step_size'],
                    method=method, model=model, samples=samples)

        records = list(zip(accuracies, range(len(accuracies)), [method] * len(accuracies)))
        val_accuracy.extend(records)

    return val_accuracy


def get_loaders(x_train, y_train, x_val, y_val, batch_size, tfms):
    loaders = OrderedDict({
        'train': loader(x_train, y_train, batch_size, tfms=tfms, train=True),
        'valid': loader(x_val, y_val, batch_size)
    })
    return loaders


# Set initial datas
def loader(x, y, batch_size=128, tfms=None, train=False):
    # ds = TensorDataset(torch.DoubleTensor(x), torch.LongTensor(y))
    ds = ImageDataset(x, y, train=train, tfms=tfms)
    _loader = DataLoader(ds, batch_size=batch_size, num_workers=4, shuffle=train)
    return _loader


def plot_metric(metrics, config, title=None):
    plt.figure(figsize=(8, 6))
    default_title = f"Validation accuracy, start size {config['start_size']}, "
    default_title += f"step size {config['step_size']}"
    title = title or default_title
    plt.title(title)

    df = pd.DataFrame(metrics, columns=['Accuracy', 'Step', 'Method'])
    sns.lineplot('Step', 'Accuracy', hue='Method', data=df)
    # plt.legend(loc='upper left')

    filename = f"ht_{config['name']}_{config['start_size']}_{config['step_size']}"
    dir = Path(__file__).parent.absolute() / 'data' / 'al'
    file = dir / filename
    plt.savefig(file)
    df.to_csv(dir / (filename + '.csv'))
    plt.show()


if __name__ == '__main__':
    config = parse_arguments()
    results = []
    for i in range(config['repeats']):
        accuracies = active_train(config, i)
        results.extend(accuracies)

    print(results)

    plot_metric(results, config)

