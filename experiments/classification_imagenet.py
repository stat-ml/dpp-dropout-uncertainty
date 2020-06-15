from time import time
from collections import defaultdict
import os
import pickle
from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import numpy as np
import torch
from catalyst.utils import set_global_seed

from alpaca.active_learning.simple_update import entropy
from alpaca.uncertainty_estimator import build_estimator

from configs import base_config, experiment_config
from models import resnet_dropout


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--acquisition', '-a', type=str, default='bald')
    parser.add_argument('--dataset-folder', type=str, default='data/imagenet')
    parser.add_argument('--bs', type=int, default=250)
    parser.add_argument('--ood', dest='ood', action='store_true')
    args = parser.parse_args()
    args.name ='imagenet'

    config = deepcopy(base_config)
    config.update(experiment_config[args.name])


    if args.ood:
        args.dataset_folder = 'data/chest'

    for param in ['name', 'acquisition', 'bs', 'dataset_folder', 'ood']:
        config[param] = getattr(args, param)


    return config


def bench_uncertainty(model, val_loader, y_val, acquisition, config, logdir):
    ood_str = '_ood' if config['ood'] else ''
    logfile = logdir / f"ue{ood_str}_{config['acquisition']}.pickle"
    print(logfile)

    probabilities = get_probabilities(model, val_loader).astype(np.single)

    estimators = ['max_prob', 'mc_dropout', 'ht_dpp', 'cov_k_dpp']
    # estimators = ['mc_dropout']

    print(estimators)

    uncertainties = {}
    times = defaultdict(list)
    for estimator_name in estimators:
        print(estimator_name)
        t0 = time()
        ue = calc_ue(model, val_loader, probabilities, config['dropout_rate'], estimator_name, nn_runs=config['nn_runs'], acquisition=acquisition)
        times[estimator_name].append(time() - t0)
        print('time', time() - t0)
        uncertainties[estimator_name] = ue

    record = {
        'y_val': y_val,
        'uncertainties': uncertainties,
        'probabilities': probabilities,
        'estimators': estimators,
        'times': times
    }
    with open(logfile, 'wb') as f:
        pickle.dump(record, f)

    return probabilities, uncertainties, estimators


def calc_ue(model, val_loader, probabilities, dropout_rate, estimator_type='max_prob', nn_runs=150, acquisition='bald'):
    if estimator_type == 'max_prob':
        ue = 1 - np.max(probabilities, axis=-1)
    elif estimator_type == 'max_entropy':
        ue = entropy(probabilities)
    else:
        acquisition_param = 'var_ratio' if acquisition == 'max_prob' else acquisition

        estimator = build_estimator(
            'bald_masked', model, dropout_mask=estimator_type, num_classes=1000,
            nn_runs=nn_runs, keep_runs=False, acquisition=acquisition_param,
            dropout_rate=dropout_rate
        )
        ue = estimator.estimate(val_loader)

    return ue


def get_probabilities(model, loader):
    model.eval().cuda()

    results = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            print(i)
            probabilities = torch.softmax(model(batch.cuda()), dim=-1)
            results.extend(list(probabilities.cpu().numpy()))
    return np.array(results)



image_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class ImageDataset(Dataset):
    def __init__(self, folder):
        self.folder = Path(folder)
        files = sorted(os.listdir(folder))
        self.files = files[:downsample]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.folder / self.files[idx]
        image = Image.open(img_path).convert('RGB')

        image = image_transforms(image).double()

        return image


def get_data(config):
    label_file = Path(config['dataset_folder'])/'val.txt'

    with open(label_file, 'r') as f:
        y_val = np.array([int(line.split()[1]) for line in f.readlines()])[:downsample]

    valid_folder = Path(config['dataset_folder']) / 'valid'

    dataset = ImageDataset(valid_folder)
    val_loader = DataLoader(dataset, batch_size=config['bs'])
    val_loader.shape = (downsample, 3, 224, 224)

    return val_loader, y_val


if __name__ == '__main__':
    downsample = 50_000

    config = parse_arguments()
    print(config)
    set_global_seed(42)
    val_loader, y_val = get_data(config)

    for i in range(config['repeats']):
        set_global_seed(i + 42)
        logdir = Path(f"logs/classification/{config['name']}_{i}")

        model = resnet_dropout(dropout_rate=config['dropout_rate']).double()

        bench_uncertainty(model, val_loader, y_val, config['acquisition'], config, logdir)
