import numpy as np

from fastai.vision import rand_pad, flip_lr
from alpaca.dataloader.builder import build_dataset


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


def prepare_fashion_mnist(config):
    dataset = build_dataset('fashion_mnist', val_size=config['val_size'])
    x_set, y_set = dataset.dataset('train')
    x_val, y_val = dataset.dataset('val')
    print(x_set.shape)

    shape = (-1, 1, 28, 28)
    x_set = ((x_set - 128) / 128).reshape(shape)
    x_val = ((x_val - 128) / 128).reshape(shape)

    train_tfms = []

    return x_set, y_set, x_val, y_val, train_tfms
