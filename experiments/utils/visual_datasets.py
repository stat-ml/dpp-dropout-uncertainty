import numpy as np
from torch.utils.data import Dataset
from albumentations import (
    HorizontalFlip, CLAHE, ShiftScaleRotate, Blur, HueSaturationValue,
    RandomBrightnessContrast, Compose, Normalize, ToFloat
)
from albumentations.pytorch.transforms import ToTensorV2
import matplotlib.pyplot as plt

from alpaca.dataloader.builder import build_dataset


# augmentations = Compose([
#     HorizontalFlip(p=0.5),
#     # Blur(blur_limit=1, p=0.2),
#     ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.7),
#     CLAHE(clip_limit=2, tile_grid_size=(2, 2), p=0.3),
#     HueSaturationValue(5, 5, 5, p=0.5),
#     RandomBrightnessContrast(0.05, 0.05, p=0.5),
# ])

augmentations = Compose([])

post_aug = Compose([
    Normalize(),
    ToTensorV2(),
])


def prepare_cifar(config):
    dataset = build_dataset('cifar_10', val_size=config['val_size'])

    x_set, y_set = dataset.dataset('train')
    x_val, y_val = dataset.dataset('val')

    shape = (-1, 3, 32, 32)
    x_set = np.moveaxis(x_set.reshape(shape), 1, 3).astype(np.uint8)
    x_val = np.moveaxis(x_val.reshape(shape), 1, 3).astype(np.uint8)

    # x_set = ((x_set - 128) / 128).reshape(shape)
    # x_val = ((x_val - 128) / 128).reshape(shape)

    train_tfms = augmentations

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
    x_set = x_set.reshape(shape).astype(np.uint8)
    x_val = x_val.reshape(shape).astype(np.uint8)
    train_tfms = augmentations

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


class ImageDataset(Dataset):
    def __init__(self, x, y, train=False, tfms=None):
        self.x = x
        self.y = y
        self.train = train
        self.tfms = tfms

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.load_img(idx), self.y[idx]

    def load_img(self, idx):
        image = self.x[idx]
        if self.tfms:
            image = self.tfms(image=image)['image']
        if image.shape[2] == 3:
            image = post_aug(image=image)['image'].double()
        return image
