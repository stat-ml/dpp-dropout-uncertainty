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
from alpaca.uncertainty_estimator.mcdue import MCDUEMasked
from alpaca.uncertainty_estimator.masks import BasicBernoulliMask, DPPMask, KDPPMask


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


def bench_uncertainty(model, loaders, x_val, y_val, config, y_scaler, metric):
    model.eval()
    with torch.no_grad():
        predictions = model(x_val)
    # unscale = y_scaler.inverse_transform
    errors = np.square(predictions.cpu() - y_val)
    mask = BasicBernoulliMask()
    estimator = MCDUEMasked(model, dropout_mask=mask, nn_runs=config['nn_runs'], dropout_rate=0.9)
    uq_mc = estimator.estimate(x_val)
    print("mcdue", uq_ll(errors, uq_mc))

    mask = DPPMask(ht_norm=True)
    estimator = MCDUEMasked(model, dropout_mask=mask, nn_runs=config['nn_runs'])
    uq_dpp = estimator.estimate(x_val)
    print("dpp", uq_ll(errors, uq_dpp))

    mask = KDPPMask(ht_norm=True)
    estimator = MCDUEMasked(model, dropout_mask=mask, nn_runs=config['nn_runs'], dropout_rate=0.7)
    uq_kdpp = estimator.estimate(x_val)
    print("k-dpp", uq_ll(errors, uq_kdpp))

    mask = DPPMask()
    estimator = MCDUEMasked(model, dropout_mask=mask, nn_runs=config['nn_runs'])
    uq_dpp_noht = estimator.estimate(x_val)
    print("dpp_no_ht", uq_ll(errors, uq_dpp_noht))

    memory = np.stack([tensor.cpu() for tensor in model.memory])

    if metric == 'covariance':
        return measure(memory, covariance)
    elif metric == 'uncertainty':
        return np.mean(uq_mc), np.mean(uq_dpp), np.mean(uq_kdpp), np.mean(uq_dpp_noht)
    elif metric == 'rmse':
        return measure(memory, rmse_ensemble, y_val)
    elif metric == 'll':
        return uq_ll(errors, uq_mc), uq_ll(errors, uq_dpp), uq_ll(errors, uq_kdpp), uq_ll(errors, uq_dpp_noht)



def rmse_ensemble(samples, labels):
    return np.sqrt(np.mean(np.square(np.mean(samples, axis=0) - labels)))


def covariance(samples, _):
    corrs = np.corrcoef(np.mean(samples, axis=-1))
    return np.mean(corrs)


def measure(memory, reduction, dop_arg=None):
    mc_samples = memory[:100]
    dpp_samples = memory[102:202]
    k_dpp_samples = memory[203:303]
    dpp_no_ht_samples = memory[-100:]
    return (reduction(mc_samples, dop_arg),
           reduction(dpp_samples, dop_arg),
           reduction(k_dpp_samples, dop_arg),
           reduction(dpp_no_ht_samples, dop_arg))



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
    return np.std(np.mean(np.mean(block, axis=-1), axis=-1))


def plot_vars(memory):
    true_vars = eval_variation(memory[100:101])
    mc_mem = memory[:100]
    mc_vars = [eval_variation(mc_mem[:i]) for i in range(1, 101, 5)]
    dpp_mem = memory[101:201]
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
    # datasets = [
    #     'boston_housing', 'concrete', 'energy_efficiency',
    #     'kin8nm', 'naval_propulsion', 'ccpp', 'red_wine',
    #     'yacht_hydrodynamics'
    # ]
    # datasets = [
    #     'boston_housing', 'concrete', 'energy_efficiency',
    # ]
    datasets = [
        'kin8nm', 'concrete', 'ccpp'# 'yacht_hydrodynamics'
    ]

    metrics = ['covariance', 'uncertainty', 'rmse', 'll']
    metric = metrics[1]


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

            mc, dpp, kdpp, dpp_noht = bench_uncertainty(
                model, loaders, x_val_tensor, y_val, config, y_scaler, metric)

            covs.append([name, mc, 'mc_dropout'])
            covs.append([name, dpp, 'dpp'])
            covs.append([name, kdpp, 'k-dpp'])
            covs.append([name, dpp_noht, 'dpp_noht'])


    if metric == 'covariance':
        df = pd.DataFrame(covs, columns=['dataset', 'correlation', 'method'])
        plt.title("Correlation on samples")
        sns.boxplot('dataset', 'correlation', hue='method', data=df)
        # plt.ylim(0.91, 1)
        for i in range(len(df['dataset'].unique()) - 1):
            plt.vlines(i + .5, 0.88, 0.98, linestyles='solid', colors='gray', alpha=0.2)

        plt.savefig('data/1_covariance.png', dpi=150)
    elif metric == 'uncertainty':
        df = pd.DataFrame(covs, columns=['dataset', 'uncertainty', 'method'])
        plt.title("Average uncertainty")
        g = sns.boxplot('dataset', 'uncertainty', hue='method', data=df)
        for i in range(len(df['dataset'].unique()) - 1):
            plt.vlines(i + .5, 0.1, 0.3, linestyles='solid', colors='gray', alpha=0.2)

        # plt.legend(loc='bottom left')
        g.legend(loc='bottom left')

        plt.savefig('data/2_uncertainty.png', dpi=150)
    elif metric == 'rmse':
        df = pd.DataFrame(covs, columns=['dataset', 'rmse', 'method'])
        plt.title("RMSE")
        sns.boxplot('dataset', 'rmse', hue='method', data=df)
        for i in range(len(df['dataset'].unique()) - 1):
            plt.vlines(i + .5, 0.15, 0.4, linestyles='solid', colors='gray', alpha=0.2)
        plt.savefig('data/3_rmse.png', dpi=150)
    elif metric == 'll':
        df = pd.DataFrame(covs, columns=['dataset', 'll', 'method'])
        plt.title("Log-likelihood")
        sns.boxplot('dataset', 'll', hue='method', data=df)
        plt.ylim(0.5, 1.4)
        for i in range(len(df['dataset'].unique()) - 1):
            plt.vlines(i + .5, 0.6, 1.2, linestyles='solid', colors='gray', alpha=0.2)
        plt.savefig('data/4_ll.png', dpi=150)
    else:
        raise ValueError

    # plt.show()


