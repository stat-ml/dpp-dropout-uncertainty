from time import time
import torch

from datasets import get_data
from model_resnet import ResNet34
from models_resnet_2 import resnet34


def main():
    loaders = get_data('cifar')
    print(loaders)

    model_1 = resnet34()
    model_2 = ResNet34(0.2)

    t = time()
    for x, y in loaders['train']:
        print(x.shape)
        model_1(x)
        break
    print('Time 1', time() - t)

    t = time()
    for x, y in loaders['train']:
        print(x.shape)
        model_2(x)
        break
    print('Time 2', time() - t)


if __name__ == '__main__':
    main()


