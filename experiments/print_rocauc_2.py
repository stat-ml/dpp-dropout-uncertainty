from collections import defaultdict
import pickle
import os
import argparse

from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from alpaca.uncertainty_estimator.bald import bald, bald_normed
from scipy.special import softmax


def estimator_name(estimator):
    return {
        'max_prob': 'Max probability',
        'mc_dropout': 'MC dropout',
        'ht_decorrelating': 'decorrelation',
        'ht_dpp': 'DPP',
        'ht_k_dpp': 'k-DPP',
        'cov_dpp': 'DPP (cov)',
        'cov_k_dpp': 'k-DPP (cov)',
        'ensemble_max_prob': 'Ensemble (max prob)',
        'ensemble_bald': 'Ensemble (bald)',
        'ensemble_var_ratio': 'Ensemble (var ratio)',
    }[estimator]


def process_file(file_name, accumulation):
    with open(file_name, 'rb') as f:
        record = pickle.load(f)

    probs = record['probabilities']
    predictions = np.argmax(probs, axis=-1)
    errors = record['y_val'] != np.array(predictions)

    mcd = record['mcd']


    uq = roc_auc_score(errors,record['uncertainties']['max_prob'])
    accumulation.append((uq, "max_prob"))

    methods = ['mc_dropout', 'ht_dpp']
    for acquisition in ['bald', 'var_ratio', 'max_prob', 'mp_variation']:
        for method in methods:
            if method == 'max_prob':
                ue = record['uncertainties'][method]
            else:
                ue = aquisition(mcd[method], acquisition)
            uq = roc_auc_score(errors, ue)
            accumulation.append((uq, f"{method}_{acquisition}"))


def print_rocauc():
    name = 'cifar'

    accumulation = []
    for i in range(5):
        file_name = f'logs/classification_better/{name}_{i}/ue_bald.pickle'
        process_file(file_name, accumulation)

        # methods = ['mc_dropout', 'ht_dpp', 'ht_k_dpp']
        # file_name = f'logs/classification/{name}_{i}/ue_var_ratio.pickle'
        # process_file(file_name, accumulation, methods, acquisition='var_ratio')
        #
        # methods = ['mc_dropout', 'ht_dpp', 'ht_k_dpp']
        # file_name = f'logs/classification/{name}_{i}/ue_bald.pickle'
        # process_file(file_name, accumulation, methods, acquisition='bald')
        #
        # methods = ['ensemble_max_prob']
        # file_name = f'logs/classification/{name}_{i}/ue_ensemble.pickle'
        # process_file(file_name, accumulation, methods, acquisition='')


    df = pd.DataFrame(accumulation, columns=['value', 'method'])
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='method', y='value', data=df)
    plt.title(f"{name.upper()}, error detection ROC AUC")
    plt.xticks(rotation=20)
    plt.show()


def aquisition(mcd_runs, method):
    if method == 'var_ratio':
        predictions = np.argmax(mcd_runs, axis=-1)
        # count how many time repeats the strongest class
        mode_count = lambda preds : np.max(np.bincount(preds))
        modes = [mode_count(point) for point in predictions]
        ue = 1 - np.array(modes) / mcd_runs.shape[1]
        return ue
    elif method == 'var_soft':
        probabilities = softmax(mcd_runs, axis=-1)
        ue = np.mean(np.std(probabilities, axis=-2), axis=-1)
        return ue
    elif method == 'bald':
        print('bald')
        return bald(mcd_runs)
    elif method == 'bald_normed':
        print('normed bald')
        return bald_normed(mcd_runs)
    elif method == 'max_prob':
        n = len(mcd_runs)
        probs = softmax(mcd_runs, axis=-1)
        mean_probs = np.mean(probs, axis=1)
        max_probs = np.max(mean_probs, -1)
        return 1 - max_probs
    elif method == 'mp_variation':
        n = len(mcd_runs)
        probs = softmax(mcd_runs, axis=-1)
        mean_probs = np.mean(probs, axis=1)
        preds = np.argmax(mean_probs, -1)
        max_probs = mean_probs[np.arange(n), preds]
        variation = np.std(probs[np.arange(n), :, preds], axis=1)
        return 1 - max_probs + variation
    else:
        raise ValueError


if __name__ == '__main__':
    print_rocauc()

