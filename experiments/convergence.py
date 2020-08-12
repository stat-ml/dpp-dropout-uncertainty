import os
import pickle
from argparse import ArgumentParser
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy.special import softmax
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from catalyst.dl import SupervisedRunner
from catalyst.dl.callbacks import AccuracyCallback, EarlyStoppingCallback
from catalyst.utils import set_global_seed

from alpaca.dataloader.builder import build_dataset
from alpaca.uncertainty_estimator import build_estimator
from alpaca.uncertainty_estimator.masks import DEFAULT_MASKS

from models import SimpleMLP



def train(config, loaders, logdir, checkpoint=None):
    model = config['model_class'](dropout_rate=config['dropout_rate']).double()

    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint)['model_state_dict'])
        model.eval()
    else:
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters())
        callbacks = [EarlyStoppingCallback(config['patience'])]

        runner = SupervisedRunner()
        runner.train(
            model, criterion, optimizer, loaders,
            logdir=logdir, num_epochs=config['epochs'], verbose=True,
            callbacks=callbacks
        )

    return model



def bench_uncertainty(model, model_checkpoint, loaders, x_val, y_val, acquisition, config):
    runner = SupervisedRunner()
    logits = runner.predict_loader(model, loaders['valid'])
    probabilities = softmax(logits, axis=-1)

    if config['acquisition'] in ['bald', 'var_ratio']:
        estimators = ['mc_dropout', 'ht_dpp', 'ht_k_dpp', 'cov_dpp', 'cov_k_dpp']
    elif config['acquisition'] == 'max_prob':
        estimators = ['max_prob', 'mc_dropout', 'ht_dpp', 'ht_k_dpp', 'cov_dpp', 'cov_k_dpp']
    else:
        raise ValueError

    print(estimators)

    uncertainties = {}
    lls = {}
    for estimator_name in estimators:
        # try:
        print(estimator_name)
        ue, ll = calc_ue(model, x_val, y_val, probabilities, estimator_name, nn_runs=config['nn_runs'], acquisition=acquisition)
        uncertainties[estimator_name] = ue
        lls[estimator_name] = ll
        # except Exception as e:
        #     print(e)

    record = {
        'checkpoint': model_checkpoint,
        'y_val': y_val,
        'uncertainties': uncertainties,
        'probabilities': probabilities,
        'logits': logits,
        'estimators': estimators,
        'lls': lls
    }
    with open(logdir / f"ue_{config['acquisition']}.pickle", 'wb') as f:
        pickle.dump(record, f)

    return probabilities, uncertainties, estimators

def parse_arguments(config):
    parser = ArgumentParser()
    parser.add_argument('--name', type=str, default='kin8nm')
    args = parser.parse_args()

    config['name'] = args.name

    return config



config = {
    'model_class': SimpleMLP,
    'epochs': 10_000,
    'patience': 50,
    'batch_size': 128,
    'dropout_rate': 0.5,
    'nn_runs': 100,
    'val_size': 0.2,
    'repeats': 1
}


def loader(x, y, batch_size):
    return DataLoader(
        TensorDataset(torch.DoubleTensor(x), torch.DoubleTensor(y)),
        batch_size=batch_size)


def get_data(config):
    dataset = build_dataset(config['name'], val_split=0)
    x_set, y_set = dataset.dataset('train')
    x_train, x_val, y_train, y_val = train_test_split(x_set, y_set, train_size=config['val_size'])

    scaler = StandardScaler()
    print(x_train[:5])
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    print(x_train[:5])

    loaders = OrderedDict({
        'train': loader(x_train, y_train, config['batch_size']),
        'valid': loader(x_val, y_val, config['batch_size'])
    })
    return loaders, x_train, y_train, x_val, y_val



if __name__ == '__main__':
    config = parse_arguments(config)
    set_global_seed(42)
    loaders, x_train, y_train, x_val, y_val = get_data(config)
    print(y_train[:5])
    for i in range(config['repeats']):
        set_global_seed(i + 42)
        logdir = Path(f"logs/regression/{config['name']}_{i}")
        print(logdir)

        possible_checkpoint = logdir / 'checkpoints' / 'best.pth'
        if os.path.exists(possible_checkpoint):
            checkpoint = possible_checkpoint
        else:
            checkpoint = None

        model = train(config, loaders, logdir, checkpoint)
        x_val_tensor = torch.cat([batch[0] for batch in loaders['valid']])

        # bench_uncertainty(
        #     model, checkpoint, loaders, x_val_tensor, y_val, config['acquisition'], config)

