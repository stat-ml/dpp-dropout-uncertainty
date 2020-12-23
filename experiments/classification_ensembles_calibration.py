import os
import pickle
from argparse import ArgumentParser
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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
from alpaca import calibrator


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('--acquisition', '-a', type=str, default='bald')
    parser.add_argument('--calibrator_type', '-c', type=str, default='temperature')
    args = parser.parse_args()

    config = deepcopy(base_config)
    config.update(experiment_config[args.name])
    config['calibrator_type'] = args.calibrator_type
    config['name'] = args.name

    return config


def bench_uncertainty(ensemble, x_val_tensor, y_val, config, metric):
    probabilities, bald = ensemble.estimate(x_val_tensor, metric)
    record = {
        'y_val': y_val,
        'ue': bald,
        'probabilities': probabilities,
        'metric': metric
    }

    # ood_str = '_ood' if config['ood'] else ''
    # with open(ensemble.logdir / f"ue_ensemble{ood_str}.pickle", 'wb') as f:
    #     pickle.dump(record, f)
    return record


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

    def estimate(self, X_pool, metric):
        mcd_realizations = torch.zeros((len(X_pool), self.n_models, 10))

        with torch.no_grad():
            for i, model in enumerate(self.models):
                model.cuda()
                prediction = model(X_pool.cuda())
                prediction = prediction.to('cpu')
                if metric != 'bald_calibrated':
                    prediction = torch.softmax(prediction, dim=-1)
                mcd_realizations[:, i, :] = prediction

        # mcd_realizations = torch.cat(mcd_realizations, dim=0)
        probabilities = mcd_realizations.mean(dim=1)
        if metric == 'max_prob':
            max_class = torch.argmax(probabilities, dim=-1)
            max_prob_ens = mcd_realizations[np.arange(len(X_pool)), :, max_class]
            ue = -max_prob_ens.mean(dim=-1) + 1
        elif metric == 'max_prob_single':
            max_class = torch.argmax(probabilities, dim=-1)
            max_prob_ens = mcd_realizations[np.arange(len(X_pool)), :1, max_class]
            ue = -max_prob_ens.mean(dim=-1) + 1
        elif metric in ['bald', 'bald_calibrated']:
            ue = bald_score(mcd_realizations.numpy())
        return probabilities, ue


def plot_metric(all_results, title):
    print('plotting')
    # bins = np.concatenate(([0, 0.01, 0.02, 0.03, 0.04, 0.05], np.arange(0.1, 1, 0.1)))
    splits = np.linspace(0.5, 1, 20)

    data = []

    y_val = all_results[0]['y_val']
    for results in all_results:
        probs = results['probabilities'].numpy()
        preds = np.argmax(probs, axis=1)
        correct_answers = (preds == y_val).astype(np.int)
        ue_ind = np.argsort(results['ue'])
        metric = results['metric']
        for split in splits:
            indices = ue_ind[:int(len(ue_ind)*split)]
            accuracy = np.sum(correct_answers[indices] / len(indices))
            data.append((split, accuracy, metric))

    df = pd.DataFrame(data, columns=('Split', 'Accuracy', 'Metric'))
    plt.figure(dpi=200)
    sns.lineplot(x='Split', y='Accuracy', data=df, hue='Metric')
    plt.title(title)
    plt.show()


def calibrate(ensemble, x_val_tensor, y_val, calibrator_factory):
    calibrated_models = []
    for model in ensemble.models:
        model.cuda()
        with torch.no_grad():
            model.cpu()
            logits = model(x_val_tensor.cpu())
        calibr = calibrator_factory(model)
        calibr.scaling(logits, torch.LongTensor(y_val), lr=1e-3, max_iter=500)
        calibrated_models.append(calibr)

    ensemble.models = calibrated_models


calibrators = {
    'temperature': lambda m: calibrator.ModelWithTempScaling(m),
    'vector': lambda m: calibrator.ModelWithVectScaling(m, n_classes=10),
    'matrix': lambda m: calibrator.ModelWithMatrScaling(m, n_classes=10)
}

if __name__ == '__main__':
    config = parse_arguments()
    set_global_seed(42)
    logbase = 'logs/classification_ensembles'
    n_models = config['n_models']

    all_results = []
    loaders, x_train, y_train, x_val, y_val = get_data(config)
    x_val_tensor = torch.cat([batch[0] for batch in loaders['valid']])

    for j in range(config['repeats']):
        ensemble = Ensemble(logbase, n_models, config, loaders, j * n_models)
        for metric in ['bald', 'max_prob', 'max_prob_single']:
            results = bench_uncertainty(ensemble, x_val_tensor, y_val, config, metric)
            all_results.append(results)
        probs = results['probabilities'].numpy()
        ece = calibrator.compute_ece(20, probs, results['y_val'], len(probs))
        print('ece before', ece)

        calibrate(ensemble, x_val_tensor, y_val, calibrators[config['calibrator_type']])
        results = bench_uncertainty(ensemble, x_val_tensor, y_val, config, 'bald_calibrated')
        probs = results['probabilities'].numpy()
        ece = calibrator.compute_ece(20, probs, results['y_val'], len(probs))
        print('ece after', ece)
        all_results.append(results)

    plot_metric(all_results, f"Scaled ensemble ({config['calibrator_type']}) on {config['name']}")
