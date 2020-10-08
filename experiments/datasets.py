from collections import OrderedDict

import torch
import torchvision
from torchvision import transforms
from sklearn.model_selection import train_test_split



def get_data(dataset_name):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # transforms.Lambda(lambda x: x.double())
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # transforms.Lambda(lambda x: x.double())
    ])
    batch_size = 512

    if dataset_name == 'cifar':
        dataset_class = torchvision.datasets.CIFAR10
        train_set = dataset_class(
            root='./data', train=True, download=True, transform=train_transform
        )
        test_set = dataset_class(
            root='./data', train=False, download=True, transform=test_transform
        )
        test_set, _ = train_test_split(test_set, train_size=2500)
    elif dataset_name == 'svhn':
        dataset_class = torchvision.datasets.SVHN
        train_set = dataset_class(
            root='./data', split='train', download=True, transform=train_transform
        )
        test_set = dataset_class(
            root='./data', split='test', download=True, transform=test_transform
        )
    elif dataset_name == 'lsun':
        dataset_class = torchvision.datasets.LSUN
    elif dataset_name == 'cifar100':
        dataset_class = torchvision.datasets.CIFAR100
    else:
        raise ValueError('Wrong dataset name')


    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=8
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=8
    )

    loaders = OrderedDict({'train': train_loader, 'valid': test_loader})
    return loaders
