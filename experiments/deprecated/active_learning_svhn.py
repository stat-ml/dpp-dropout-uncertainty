import sys
import torch
import numpy as np
from fastai.vision import (rand_pad, flip_lr)

sys.path.append('..')
from dataloader.builder import build_dataset
from uncertainty_estimator.masks import DEFAULT_MASKS
from active_learning import main

torch.cuda.set_device(1)
torch.backends.cudnn.benchmark = True


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


svhn_config = {
    'repeats': 3,
    'start_size': 5_000,
    'step_size': 50,
    'val_size': 10_000,
    'pool_size': 12_000,
    'steps': 30,
    'methods': ['random', 'error_oracle', 'max_entropy', *DEFAULT_MASKS],
    'epochs': 30,
    'patience': 2,
    'model_type': 'resnet',
    'nn_runs': 100,
    'batch_size': 256,
    'start_lr': 5e-4,
    'weight_decay': 0.2,
    'prepare_dataset': prepare_svhn,
    'name': 'svhn'
}

experiment = 4

if experiment == 2:
    svhn_config['pool_size'] = 5000
    svhn_config['start_size'] = 2000
    svhn_config['step_size'] = 20
    svhn_config['model'] = 'conv'
elif experiment == 3:
    svhn_config['pool_size'] = 20_000
    svhn_config['start_size'] = 10_000
    svhn_config['step_size'] = 100
    svhn_config['model'] = 'resent'
elif experiment == 4:
    svhn_config['pool_size'] = 5_000
    svhn_config['start_size'] = 1_000
    svhn_config['step_size'] = 30
    svhn_config['model'] = 'resent'
    svhn_config['repeats'] = 10


if __name__ == '__main__':
    main(svhn_config)
