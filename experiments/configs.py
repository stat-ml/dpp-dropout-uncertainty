from copy import deepcopy

import torch

from utils.visual_datasets import prepare_cifar, prepare_mnist, prepare_svhn
from models import SimpleConv, StrongConv


base_config = {
    'train_size': 50_000,
    'val_size': 10_000,
    'prepare_dataset': prepare_mnist,
    'model_class': SimpleConv,
    'epochs': 50,
    'patience': 3,
    'batch_size': 128,
    'repeats': 5
}


experiment_config = {
    'mnist': {
        'checkpoint': 'logs/ht/mnist/checkpoints/train.5.pth'
    },
    'svhn': {
        'prepare_dataset': prepare_svhn,
        'model_class': StrongConv,
        'repeats': 5
    },
    'cifar': {
        'prepare_dataset': prepare_cifar,
        'model_class': StrongConv,
        'repeats': 5
    }
}


al_config = deepcopy(base_config)
al_config.update({
    'pool_size': 10_000,
    'repeats': 1,
    'methods': ['random', 'mc_dropout']
})

al_experiments = {
    'mnist': {
        'start_size': 200,
        'step_size': 10,
        'steps': 5
    }
}
