from time import time
import os
import pickle
from argparse import ArgumentParser
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from PIL import Image

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.special import softmax
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from catalyst.dl import SupervisedRunner
from catalyst.dl.callbacks import AccuracyCallback, EarlyStoppingCallback
from catalyst.utils import set_global_seed

from alpaca.active_learning.simple_update import entropy
from alpaca.uncertainty_estimator import build_estimator
from alpaca.uncertainty_estimator.masks import DEFAULT_MASKS

from configs import base_config, experiment_config
from deprecated.classification_active_learning import loader
from models import resnet_dropout


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('--acquisition', '-a', type=str, default='bald')
    parser.add_argument('--dataset-folder', type=str, default='data/imagenet')
    parser.add_argument('--bs', type=int, default=250)
    parser.add_argument('--ood', dest='ood', action='store_true')
    args = parser.parse_args()


    config = deepcopy(base_config)
    config.update(experiment_config[args.name])


    if args.ood:
        args.dataset_folder = 'data/chest'

    for param in ['name', 'acquisition', 'bs', 'dataset_folder', 'ood']:
        config[param] = getattr(args, param)


    return config


def bench_uncertainty(model, val_loader, y_val, acquisition, config, logdir):
    ood_str = '_ood' if config['ood'] else ''
    logfile = logdir / f"ue{ood_str}_{config['acquisition']}{downsample}.pickle"
    print(logfile)

    probabilities = get_probabilities(model, val_loader).astype(np.single)

    estimators = ['max_prob', 'mc_dropout', 'ht_dpp', 'cov_k_dpp']
    # estimators = ['mc_dropout']

    print(estimators)

    uncertainties = {}
    times = {}
    for estimator_name in estimators:
        # try:
        print(estimator_name)
        t0 = time()
        ue = calc_ue(model, val_loader, probabilities, config['dropout_rate'], estimator_name, nn_runs=config['nn_runs'], acquisition=acquisition)
        times[estimator_name] = time() - t0
        print('time', time() - t0)
        uncertainties[estimator_name] = ue
        # except Exception as e:
        #     print(e)

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
        # ue = estimator.estimate(torch.DoubleTensor(datapoints).cuda())
        ue = estimator.estimate(val_loader)
        # if acquisition == 'max_prob':
        #     probs = softmax(estimator.last_mcd_runs(), axis=-1)
        #     probs = np.mean(probs, axis=-2)
        #     ue = 1 - np.max(probs, axis=-1)

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
        # self.files = [file for file in files if file.endswith(".JPEG")][:downsample]
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

        # x_val_tensor = torch.cat([batch for batch in val_loader]).double()

        bench_uncertainty(model, val_loader, y_val, config['acquisition'], config, logdir)
