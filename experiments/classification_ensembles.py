import os
import pickle
from argparse import ArgumentParser
from collections import OrderedDict, defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
from alpaca.uncertainty_estimator.bald import bald as bald_score

from configs import base_config, experiment_config
from classification_active_learning import loader


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('--acquisition', '-a', type=str, default='bald')
    parser.add_argument('--covariance', dest='covariance', action='store_true')
    args = parser.parse_args()


    config = deepcopy(base_config)
    config.update(experiment_config[args.name])
    config['name'] = args.name
    config['acquisition'] = args.acquisition
    config['covariance'] = args.covariance

    return config


def train(config, loaders, logdir, checkpoint=None):
    model = config['model_class'](dropout_rate=config['dropout_rate']).double()

    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint)['model_state_dict'])
        model.eval()
    else:
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())
        callbacks = [AccuracyCallback(num_classes=10), EarlyStoppingCallback(config['patience'])]

        runner = SupervisedRunner()
        runner.train(
            model, criterion, optimizer, loaders,
            logdir=logdir, num_epochs=config['epochs'], verbose=True,
            callbacks=callbacks
        )

    return model


def bench_uncertainty(ensemble, x_val_tensor, y_val):
    probabilities, max_prob, bald = ensemble.estimate(x_val_tensor)
    uncertainties = {
        'ensemble_max_prob': max_prob,
        'ensemble_bald': bald,
    }
    record = {
        'y_val': y_val,
        'uncertainties': uncertainties,
        'probabilities': probabilities,
        'estimators': list(uncertainties.keys()),
    }

    with open(ensemble.logdir / f"ue_ensemble.pickle", 'wb') as f:
        pickle.dump(record, f)
    #
    # return probabilities, uncertainties, estimators


def log_likelihood(probabilities, y):
    try:
        ll = np.mean(np.log(probabilities[np.arange(len(probabilities)), y]))
    except FloatingPointError:
        import ipdb; ipdb.set_trace()
    return ll




def get_data(config):
    x_train, y_train, x_val, y_val, train_tfms = config['prepare_dataset'](config)

    if len(x_train) > config['train_size']:
        x_train, _, y_train, _ = train_test_split(
            x_train, y_train, train_size=config['train_size'], stratify=y_train
        )

    loaders = OrderedDict({
        'train': loader(x_train, y_train, config['batch_size'], tfms=train_tfms, train=True),
        'valid': loader(x_val, y_val, config['batch_size'])
    })
    return loaders, x_train, y_train, x_val, y_val


class Ensemble:
    def __init__(self, logbase, n_models, start_i=0):
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
    loaders, x_train, y_train, x_val, y_val = get_data(config)
    print(y_train[:5])
    x_val_tensor = torch.cat([batch[0] for batch in loaders['valid']])

    config['repeats'] = 3
    logbase = 'logs/classification_ensembles'
    n_models = config['n_models']
    for j in range(config['repeats']):
        ensemble = Ensemble(logbase, n_models, j * n_models)
        bench_uncertainty(ensemble, x_val_tensor, y_val)

    #     aucs = misclassfication_detection(y_val, probabilities, uncertainties, estimators)
    #     rocaucs.extend(aucs)
    #
    # df = pd.DataFrame(rocaucs, columns=['Estimator', 'ROC-AUCs'])
    # df.to_csv(f"logs/{config['name']}_{config['acquisition']}.csv")
    # plt.figure(figsize=(9, 6))
    # plt.title(f"Error detection for {config['name']}")
    # sns.boxplot('Estimator', 'ROC-AUCs', data=df)
    # plt.savefig(f"data/ed/{config['name']}_bn.png", dpi=150)
    # plt.show()