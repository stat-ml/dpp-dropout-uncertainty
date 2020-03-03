from copy import deepcopy
import sys
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd

import torch
import torch.nn.functional as F

from fastai.vision import (ImageDataBunch, Learner, accuracy)
from fastai.callbacks import EarlyStoppingCallback

sys.path.append('..')
from uncertainty_estimator.masks import DEFAULT_MASKS
from utils import ImageArrayDS
from experiment_setup import ROOT_DIR
from pathlib import Path
from experiments.active_learning_mnist import prepare_mnist
from active_learning import build_model, prepare_cifar
from experiments.active_learning_svhn import prepare_svhn
from sklearn.metrics import roc_auc_score, roc_curve
from uncertainty_estimator.masks import build_mask
from experiment_setup import build_estimator
from active_learning.simple_update import entropy
from utils import prepare_fashion_mnist

torch.cuda.set_device(1)
label = 'ratio_3'

config_mnist = {
    'train_size': 50_000,
    'val_size': 5_000,
    'model_type': 'simple_conv',
    'batch_size': 256,
    'patience': 3,
    'epochs': 50,
    'start_lr': 1e-3,
    'weight_decay': 0.1,
    'reload': False,
    'nn_runs': 150,
    # 'estimators': ['max_prob', 'max_entropy', *DEFAULT_MASKS],
    'estimators': DEFAULT_MASKS,
    'repeats': 5,
    'name': 'MNIST',
    'prepare_dataset': prepare_mnist,
    'alternative_dataset': prepare_fashion_mnist,
}

config_cifar = deepcopy(config_mnist)
config_cifar.update({
    'train_size': 50_000,
    'val_size': 5_000,
    'model_type': 'resnet',
    'name': 'CIFAR-10',
    'prepare_dataset': prepare_cifar,
    'alternative_dataset': prepare_svhn,
})

configs = [config_mnist, config_cifar]


def benchmark_ood(config):
    results = []
    plt.figure(figsize=(10, 8))
    for i in range(config['repeats']):
        x_set, y_set, x_val, y_val, train_tfms = config['prepare_dataset'](config)

        if len(x_set) > config['train_size']:
            _, x_train, _, y_train = train_test_split(
                x_set, y_set, test_size=config['train_size'], stratify=y_set)
        else:
            x_train, y_train = x_set, y_set

        _, _, x_alt, y_alt, _ = config['alternative_dataset'](config)

        train_ds = ImageArrayDS(x_train, y_train, train_tfms)
        val_ds = ImageArrayDS(x_val, y_val)
        data = ImageDataBunch.create(train_ds, val_ds, bs=config['batch_size'])

        loss_func = torch.nn.CrossEntropyLoss()
        np.set_printoptions(threshold=sys.maxsize, suppress=True)

        model = build_model(config['model_type'])
        callbacks = [partial(EarlyStoppingCallback, min_delta=1e-3, patience=config['patience'])]
        learner = Learner(data, model, metrics=accuracy, loss_func=loss_func, callback_fns=callbacks)
        learner.fit(config['epochs'], config['start_lr'], wd=config['weight_decay'])

        original_images = torch.FloatTensor(x_val)
        alt_images = torch.FloatTensor(x_alt)
        images = torch.cat((original_images, alt_images)).cuda()
        y = np.array([0]*len(original_images) + [1]*len(alt_images))

        probabilities = F.softmax(model(images), dim=1).detach().cpu().numpy()

        for name in config['estimators']:
            ue = calc_ue(model, images, probabilities, name, config['nn_runs'])

            roc_auc = roc_auc_score(y, ue)
            print(name, roc_auc)
            results.append((name, roc_auc))

            if i == config['repeats'] - 1:
                fpr, tpr, thresholds = roc_curve(y, ue, pos_label=1)
                plt.plot(fpr, tpr, label=name, alpha=0.8)
                plt.xlabel('FPR')
                plt.ylabel('TPR')

    dir = Path(ROOT_DIR) / 'experiments' / 'data' / 'ood'
    plt.title(f"{config['name']} ood detection ROC")
    plt.legend()
    plt.savefig(dir / f"var_{label}_ood_roc_{config['name']}_{config['train_size']}_{config['nn_runs']}")
    # plt.show()

    plt.figure(figsize=(10, 8))
    plt.title(f"{config['name']} ood detection ROC-AUC")
    df = pd.DataFrame(results, columns=['Estimator type', 'ROC-AUC score'])
    sns.boxplot('Estimator type', 'ROC-AUC score', data=df)
    plt.savefig(dir / f"var_{label}_ood_boxplot_{config['name']}_{config['train_size']}_{config['nn_runs']}")
    # plt.show()


def calc_ue(model, images, probabilities, estimator_type='max_prob', nn_runs=100):
    if estimator_type == 'max_prob':
        ue = 1 - probabilities[np.arange(len(probabilities)), np.argmax(probabilities, axis=-1)]
    elif estimator_type == 'max_entropy':
        ue = entropy(probabilities)
    else:
        mask = build_mask(estimator_type)
        estimator = build_estimator('bald_masked', model, dropout_mask=mask, num_classes=10, nn_runs=nn_runs)
        ue = estimator.estimate(images)
        print(ue[:10])

    return ue


if __name__ == '__main__':
    for config in configs:
        print(config)
        benchmark_ood(config)
