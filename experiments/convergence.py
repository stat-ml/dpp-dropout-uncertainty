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
from alpaca.uncertainty_estimator.masks import build_mask
from alpaca.uncertainty_estimator.masks import DEFAULT_MASKS
from alpaca.analysis.metrics import uq_ll


from models import SimpleMLP
from convergence_estimators import BasicBernoulliMask, MCDUEMasked, DPPMask




def train(config, loaders, logdir, input_size, checkpoint=None):
    model = config['model_class'](dropout_rate=config['dropout_rate'], input_size=input_size).double().cuda()

    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint)['model_state_dict'])
        model.eval()
    else:
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters())
        callbacks = [EarlyStoppingCallback(config['patience'])]

        runner = SupervisedRunner()
        model.train()
        runner.train(
            model, criterion, optimizer, loaders,
            logdir=logdir, num_epochs=config['epochs'], verbose=True,
            callbacks=callbacks
        )

    return model



def bench_uncertainty(model, loaders, x_val, y_val, config, y_scaler):
    print(len(model.memory))
    model.eval()
    with torch.no_grad():
        predictions = model(x_val)
    # unscale = y_scaler.inverse_transform
    print(len(model.memory))
    errors = np.square(predictions.cpu() - y_val)
    mask = BasicBernoulliMask()
    estimator = MCDUEMasked(model, dropout_mask=mask, nn_runs=config['nn_runs'], dropout_rate=0.5)
    uq = estimator.estimate(x_val)
    print("mcdue", uq_ll(errors, uq))
    print(len(model.memory))

    mask = DPPMask(ht_norm=True)
    estimator = MCDUEMasked(model, dropout_mask=mask, nn_runs=config['nn_runs'])
    uq = estimator.estimate(x_val)
    print("dpp", uq_ll(errors, uq))
    print(len(model.memory))
    memory = np.stack([tensor.cpu() for tensor in model.memory])
    # plot_means(memory)
    # plot_vars(memory)
    return measure_covariance(memory)


def measure_covariance(memory):
    mc_samples = memory[:100]
    mc_corrs = np.corrcoef(np.mean(mc_samples, axis=-1))
    mc_cor = np.mean(mc_corrs)
    dpp_samples = memory[-100:]
    dpp_corrs = np.corrcoef(np.mean(dpp_samples, axis=-1))
    dpp_cor = np.mean(dpp_corrs)
    return mc_cor, dpp_cor


def plot_means(memory):
    true_mean = eval_mean(memory[100:101])
    mc_mem = memory[:100]
    mc_means = [eval_mean(mc_mem[:i]) for i in range(1, 101, 5)]
    dpp_mem = memory[-100:]
    dpp_means = [eval_mean(dpp_mem[:i]) for i in range(1, 101, 5)]
    points = np.array(range(1, 101, 5))

    plt.title(config['name'] + ', narrow MC dropout')
    plt.xlabel('Forward passes')
    plt.ylabel('Mean estimation')
    plt.plot(points, mc_means, label='MC dropout')
    plt.plot(points, dpp_means, label='DPP')
    plt.hlines(true_mean, 1, 100)
    plt.legend()
    plt.show()


def eval_mean(block):
    return np.mean((np.sum(block, axis=-1)))


def eval_variation(block):
    # return np.mean(np.std(np.mean(block, axis=-1)), axis=-1)
    # return np.mean((np.std(block, axis=-1)))
    return np.std(np.mean(np.mean(block, axis=-1), axis=-1))


def plot_vars(memory):
    true_vars = eval_variation(memory[100:101])
    mc_mem = memory[:100]
    mc_vars = [eval_variation(mc_mem[:i]) for i in range(1, 101, 5)]
    dpp_mem = memory[-100:]
    dpp_vars = [eval_variation(dpp_mem[:i]) for i in range(1, 101, 5)]
    points = np.array(range(1, 101, 5))

    plt.title(config['name']+', narrow MC-dropout')
    plt.xlabel('Forward passes')
    plt.ylabel('Variance of means')
    plt.plot(points, mc_vars, label='MC dropout')
    plt.plot(points, dpp_vars, label='DPP')
    plt.hlines(true_vars, 1, 100)
    plt.legend()
    plt.show()


def parse_arguments(config):
    parser = ArgumentParser()
    parser.add_argument('--name', type=str, default='kin8nm')
    args = parser.parse_args()

    config['name'] = args.name

    return config


def loader(x, y, batch_size):
    return DataLoader(
        TensorDataset(torch.DoubleTensor(x), torch.DoubleTensor(y)),
        batch_size=batch_size)


def get_data(config):
    dataset = build_dataset(config['name'], val_split=0)
    x_set, y_set = dataset.dataset('train')
    x_train, x_val, y_train, y_val = train_test_split(x_set, y_set, test_size=config['val_size'])
    print(x_train.shape)
    print(x_val.shape)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)

    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train)
    y_val = y_scaler.transform(y_val)

    loaders = OrderedDict({
        'train': loader(x_train, y_train, config['batch_size']),
        'valid': loader(x_val, y_val, config['batch_size'])
    })
    return loaders, x_train, y_train, x_val, y_val, y_scaler


config = {
    'model_class': SimpleMLP,
    'epochs': 10_000,
    'patience': 50,
    'batch_size': 128,
    'dropout_rate': 0.5,
    'nn_runs': 100,
    'val_size': 0.2,
    'repeats': 5,
    'estimator': 'dpp'
}


if __name__ == '__main__':
    config = parse_arguments(config)
    covs = []
    datasets = [
        'boston_housing', 'concrete', 'energy_efficiency',
        'kin8nm', 'naval_propulsion', 'ccpp', 'red_wine',
        'yacht_hydrodynamics'
    ]
    for name in datasets:
        config['name'] = name
        # set_global_seed(40)
        for i in range(config['repeats']):
            set_global_seed(i + 40)
            loaders, x_train, y_train, x_val, y_val, y_scaler = get_data(config)
            logdir = Path(f"logs/regression/{config['name']}_{i}")
            print(logdir)

            possible_checkpoint = logdir / 'checkpoints' / 'best.pth'
            if os.path.exists(possible_checkpoint):
                checkpoint = possible_checkpoint
            else:
                checkpoint = None

            model = train(config, loaders, logdir, x_train.shape[-1], checkpoint)
            x_val_tensor = torch.cat([batch[0] for batch in loaders['valid']]).cuda()

            cov_mc, cov_dpp = bench_uncertainty(
                model, loaders, x_val_tensor, y_val, config, y_scaler)

            covs.append([name, cov_mc, 'mc_dropout'])
            covs.append([name, cov_dpp, 'dpp'])

    df = pd.DataFrame(covs, columns=['dataset', 'covariance', 'method'])
    sns.boxplot('dataset', 'covariance', hue='method', data=df)
    plt.show()

