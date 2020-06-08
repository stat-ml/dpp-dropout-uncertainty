from copy import deepcopy

import torch

from utils.visual_datasets import prepare_cifar, prepare_mnist, prepare_svhn
from models import SimpleConv, StrongConv, resnet_masked


base_config = {
    'train_size': 50_000,
    'val_size': 10_000,
    'prepare_dataset': prepare_mnist,
    'model_class': SimpleConv,
    'epochs': 50,
    'patience': 3,
    'batch_size': 128,
    'repeats': 3,
    'dropout_rate': 0.5,
}


experiment_config = {
    'mnist': {
        'train_size': 1000,
        'dropout_rate': 0.5,
        'n_models': 20
    },
    'cifar': {
        'prepare_dataset': prepare_cifar,
        'model_class': StrongConv,
        'n_models': 20,
    },
    'imagenet': {
        'dropout_rate': 0.5,
        'repeats': 1,
        'nn_runs': 100
    }
}

experiment_ood_config = {
    'mnist': {
        'train_size': 1000,
        'ood_dataset': prepare_mnist
    },
    'svhn': {
        'prepare_dataset': prepare_svhn,
        'model_class': StrongConv,
        'repeats': 3,
    },
    'cifar': {
        'prepare_dataset': prepare_cifar,
        'model_class': StrongConv,
        'repeats': 3
    }
}


al_config = deepcopy(base_config)
al_config.update({
    'pool_size': 10_000,
    'val_size': 10_000,
    'repeats': 5,
    'methods': ['random', 'mc_dropout', 'decorrelating_sc', 'dpp', 'ht_dpp', 'k_dpp', 'ht_k_dpp'],
    'steps': 50
})

al_experiments = {
    'mnist': {
        'start_size': 200,
        'step_size': 10,
    },
    'cifar': {
        'start_size': 2000,
        'step_size': 30,
        'prepare_dataset': prepare_cifar,
        'model_class': StrongConv,
        'repeats': 3
    },
    'svhn': {
        'start_size': 2000,
        'step_size': 30,
        'prepare_dataset': prepare_svhn,
        'model_class': StrongConv,
        'repeats': 3
    }
}
