from copy import deepcopy
import sys
from functools import partial
from pathlib import Path
import random

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

import torch
import torch.nn.functional as F
from fastai.vision import (ImageDataBunch, Learner, accuracy)
from fastai.callbacks import EarlyStoppingCallback

from alpaca.uncertainty_estimator.masks import DEFAULT_MASKS
from alpaca.uncertainty_estimator import build_estimator
from alpaca.active_learning.simple_update import entropy

from utils.fastai import ImageArrayDS
from classification_active_learning import build_model
from utils.visual_datasets import prepare_cifar, prepare_mnist, prepare_svhn


"""
Experiment to detect errors by uncertainty estimation quantification
It provided on MNIST, CIFAR and SVHN datasets (see config below)
We report results as a boxplot ROC-AUC figure on multiple runs
"""

label = 'detector'

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.set_device(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = 'cuda'
else:
    device = 'cpu'


def benchmark_uncertainty(config):
    results = []
    plt.figure(figsize=(10, 8))
    for i in range(config['repeats']):
        # Load data
        x_set, y_set, x_val, y_val, train_tfms = config['prepare_dataset'](config)

        if len(x_set) > config['train_size']:
            _, x_train, _, y_train = train_test_split(
                x_set, y_set, test_size=config['train_size'], stratify=y_set)
        else:
            x_train, y_train = x_set, y_set

        train_ds = ImageArrayDS(x_train, y_train, train_tfms)
        val_ds = ImageArrayDS(x_val, y_val)
        data = ImageDataBunch.create(train_ds, val_ds, bs=config['batch_size'])

        # Train model
        loss_func = torch.nn.CrossEntropyLoss()
        np.set_printoptions(threshold=sys.maxsize, suppress=True)

        model = build_model(config['model_type'])
        callbacks = [partial(EarlyStoppingCallback, min_delta=1e-3, patience=config['patience'])]
        learner = Learner(data, model, metrics=accuracy, loss_func=loss_func, callback_fns=callbacks)
        learner.fit(config['epochs'], config['start_lr'], wd=config['weight_decay'])

        # Evaluate different estimators
        images = torch.FloatTensor(x_val).to(device)
        probabilities = F.softmax(model(images), dim=1).detach().cpu().numpy()
        predictions = np.argmax(probabilities, axis=-1)

        for name in config['estimators']:
            ue = calc_ue(model, images, probabilities, name, config['nn_runs'])
            mistake = 1 - (predictions == y_val).astype(np.int)

            roc_auc = roc_auc_score(mistake, ue)
            print(name, roc_auc)
            results.append((name, roc_auc))

            if i == config['repeats'] - 1:
                fpr, tpr, thresholds = roc_curve(mistake, ue, pos_label=1)
                plt.plot(fpr, tpr, label=name, alpha=0.8)
                plt.xlabel('FPR')
                plt.ylabel('TPR')

    # Plot the results and generate figures
    dir = Path(__file__).parent.absolute() / 'data' / 'detector'
    plt.title(f"{config['name']} uncertainty ROC")
    plt.legend()
    file = f"var_{label}_roc_{config['name']}_{config['train_size']}_{config['nn_runs']}"
    plt.savefig(dir / file)
    # plt.show()

    df = pd.DataFrame(results, columns=['Estimator type', 'ROC-AUC score'])
    df = df.replace('mc_dropout', 'MC dropout')
    df = df.replace('decorrelating_sc', 'decorrelation')
    df = df[df['Estimator type'] != 'k_dpp_noisereg']
    print(df)
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(left=0.2)

    with sns.axes_style('whitegrid'):
        sns.boxplot(data=df, y='ROC-AUC score', x='Estimator type', ax=ax)

    ax.yaxis.grid(True)
    ax.xaxis.grid(True)

    plt.title(f'{config["name"]} wrong prediction ROC-AUC')

    file = f"var_{label}_boxplot_{config['name']}_{config['train_size']}_{config['nn_runs']}"
    plt.savefig(dir / file)
    df.to_csv(dir / (file + '.csv'))


def calc_ue(model, images, probabilities, estimator_type='max_prob', nn_runs=100):
    if estimator_type == 'max_prob':
        ue = 1 - probabilities[np.arange(len(probabilities)), np.argmax(probabilities, axis=-1)]
    elif estimator_type == 'max_entropy':
        ue = entropy(probabilities)
    else:
        estimator = build_estimator('bald_masked', model, dropout_mask=estimator_type, num_classes=10, nn_runs=nn_runs)
        ue = estimator.estimate(images)
        print(ue[:10])
    return ue


config_mnist = {
    'train_size': 60_000,
    'val_size': 10_000,
    'model_type': 'simple_conv',
    'batch_size': 256,
    'patience': 3,
    'epochs': 100,
    'start_lr': 1e-3,
    'weight_decay': 0.1,
    'reload': False,
    'nn_runs': 150,
    'estimators': DEFAULT_MASKS,
    'repeats': 5,
    'name': 'MNIST',
    'prepare_dataset': prepare_mnist,
}

# # TODO: for debug, remove
# config_mnist.update({
#     'epochs': 2,
#     'estimators': ['decorrelating_sc', 'mc_dropout'],
#     'repeats': 1
# })

config_cifar = deepcopy(config_mnist)
config_cifar.update({
    'train_size': 60_000,
    'val_size': 10_000,
    'model_type': 'resnet',
    'name': 'CIFAR-10',
    'prepare_dataset': prepare_cifar
})

config_svhn = deepcopy(config_mnist)
config_svhn.update({
    'train_size': 60_000,
    'val_size': 10_000,
    'model_type': 'resnet',
    'name': 'SVHN',
    'prepare_dataset': prepare_svhn
})

configs = [config_mnist, config_cifar, config_svhn]


if __name__ == '__main__':
    for config in configs:
        print(config)
        benchmark_uncertainty(config)
