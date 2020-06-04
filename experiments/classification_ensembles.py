import os
import pickle
from argparse import ArgumentParser
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from catalyst.dl import SupervisedRunner
from catalyst.dl.callbacks import AccuracyCallback, EarlyStoppingCallback
from catalyst.utils import set_global_seed

from alpaca.uncertainty_estimator.bald import bald as bald_score

from configs import base_config, experiment_config
from deprecated.classification_active_learning import loader
from classification_ue_ood import get_data as get_data_ood
from classification_ue import get_data, train


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('--acquisition', '-a', type=str, default='bald')
    parser.add_argument('--ood', dest='ood', action='store_true')
    args = parser.parse_args()

    config = deepcopy(base_config)
    config.update(experiment_config[args.name])
    config['name'] = args.name
    config['acquisition'] = args.acquisition
    config['ood'] = args.ood

    return config


def bench_uncertainty(ensemble, x_val_tensor, y_val, config):
    probabilities, max_prob, bald = ensemble.estimate(x_val_tensor)
    uncertainties = {
        'ensemble_max_prob': np.array(max_prob),
        'ensemble_bald': np.array(bald),
    }
    record = {
        'y_val': y_val,
        'uncertainties': uncertainties,
        'probabilities': probabilities,
        'estimators': list(uncertainties.keys()),
    }

    ood_str = '_ood' if config['ood'] else ''
    with open(ensemble.logdir / f"ue_ensemble{ood_str}.pickle", 'wb') as f:
        pickle.dump(record, f)


class Ensemble:
    def __init__(self, logbase, n_models, config, loaders, start_i=0):
        self.models = []
        self.n_models = n_models
        self.logdir = Path(f"{logbase}/{config['name']}_{start_i}")

        for i in range(start_i, start_i + n_models):
            set_global_seed(i + 42)
            logdir = Path(f"{logbase}/{config['name']}_{i}")
            print(logdir)

            possible_checkpoint = logdir / 'checkpoints' / 'best.pth'
            if os.path.exists(possible_checkpoint):
                checkpoint = possible_checkpoint
            else:
                checkpoint = None

            self.models.append(train(config, loaders, logdir, checkpoint))

    def estimate(self, X_pool):
        mcd_realizations = torch.zeros((len(X_pool), self.n_models, 10))

        with torch.no_grad():
            for i, model in enumerate(self.models):
                model.cuda()
                prediction = model(X_pool.cuda())
                prediction = prediction.to('cpu')
                # mcd_realizations.append(prediction)
                mcd_realizations[:, i, :] = torch.softmax(prediction, dim=-1)


        # mcd_realizations = torch.cat(mcd_realizations, dim=0)
        probabilities = mcd_realizations.mean(dim=1)
        max_class = torch.argmax(probabilities, dim=-1)
        max_prob_ens = mcd_realizations[np.arange(len(X_pool)), :, max_class]
        max_prob_ue = -max_prob_ens.mean(dim=-1) + 1
        bald = bald_score(mcd_realizations.numpy())
        return probabilities, max_prob_ue, bald


if __name__ == '__main__':
    config = parse_arguments()
    set_global_seed(42)
    logbase = 'logs/classification_ensembles'
    n_models = config['n_models']

    if config['ood']:
        loaders, ood_loader, x_train, y_train, x_val, y_val, x_ood, y_ood = get_data_ood(config)
        x_ood_tensor = torch.cat([batch[0] for batch in ood_loader])
        for j in range(config['repeats']):
            ensemble = Ensemble(logbase, n_models, config, loaders, j * n_models)
            bench_uncertainty(ensemble, x_ood_tensor, y_val, config)
    else:
        loaders, x_train, y_train, x_val, y_val = get_data(config)
        x_val_tensor = torch.cat([batch[0] for batch in loaders['valid']])

        for j in range(config['repeats']):
            ensemble = Ensemble(logbase, n_models, config, loaders, j * n_models)
            bench_uncertainty(ensemble, x_val_tensor, y_val, config)

