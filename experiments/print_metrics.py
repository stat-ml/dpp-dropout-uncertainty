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

name = 'svhn'
metric = ['auc', 'rejection'][1]


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


def process_file(file_name, ood_name, accumulation):
    with open(file_name, 'rb') as f:
        record = pickle.load(f)

    with open(ood_name, 'rb') as f:
        ood_record = pickle.load(f)

    sub = 300
    probs = record['probabilities']
    predictions = np.argmax(probs, axis=-1)
    errors = (record['y_val'] != np.array(predictions))[:sub]
    ood_errors = np.ones(ood_record['mcd']['max_prob'].shape[0])[:sub]
    errors = np.concatenate((errors, ood_errors))


    mcd = record['mcd']
    ood_mcd = ood_record['mcd']

    import ipdb; ipdb.set_trace()

    start = 0.2
    ue = np.concatenate((
        record['uncertainties']['max_prob'][:sub],
        ood_record['uncertainties']['max_prob'][:sub]
    ))

    idx = np.argsort(ue)
    for fraction in np.arange(start, 1.01, 0.003):
        part_size = int(fraction * len(idx))
        part = errors[idx][:part_size]
        accuracy = 1 - sum(part) / len(part)
        accumulation.append((fraction, accuracy, 'max_prob'))


    methods = ['mc_dropout', 'ht_dpp', 'ht_k_dpp']
    # for acquisition in ['bald', 'var_ratio', 'max_prob', 'mp_variation']:
    for acquisition in ['bald', 'mp_variation']:
        for method in methods:
            ue = np.concatenate((
                aquisition(mcd[method], acquisition)[:sub],
                aquisition(ood_mcd[method], acquisition)[:sub]
            ))

            idx = np.argsort(ue)
            for fraction in np.arange(start, 1.01, 0.003):
                part_size = int(fraction * len(idx))
                part = errors[idx][:part_size]
                accuracy = 1 - sum(part) / len(part)
                accumulation.append((fraction, accuracy, f"{method}_{acquisition}"))


def print_rocauc():
    accumulation = []

    for i in range(5):
        file_name = f'logs/classification_better/{name}_{i}/ue.pickle'
        ood_name = f'logs/classification_better/{name}_{i}/ue_ood.pickle'
        process_file(file_name, ood_name, accumulation)


    df = pd.DataFrame(accumulation, columns=['Fraction', 'Accuracy', 'method'])
    plt.figure(figsize=(12, 8))
    sns.lineplot(x='Fraction', y='Accuracy', data=df, hue='method')
    plt.title(f"{name.upper()}, accuracy by partly rejection, 50% OOD")
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
        return bald(mcd_runs)
    elif method == 'bald_normed':
        return bald_normed(mcd_runs)
    elif method == 'max_prob':
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

