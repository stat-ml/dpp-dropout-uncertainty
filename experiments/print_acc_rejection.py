from collections import defaultdict
import pickle
import os
import argparse

from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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


def process_file(file_name, accumulation, methods, acquisition):
    with open(file_name, 'rb') as f:
        record = pickle.load(f)

    probs = record['probabilities']
    predictions = np.argmax(probs, axis=-1)
    correct_predictions = record['y_val'] == np.array(predictions)

    for method in methods:
        uq = record['uncertainties'][method]
        idx = np.argsort(uq)
        for fraction in np.arange(0.5, 1.01, 0.01):
            part_size = int(fraction * len(idx))
            part = correct_predictions[idx][:part_size]
            accuracy = sum(part) / len(part)
            accumulation.append((fraction, accuracy, method))


def add_random(accumulation, file_name):
    with open(file_name, 'rb') as f:
        record = pickle.load(f)

    probs = record['probabilities']
    predictions = np.argmax(probs, axis=-1)
    correct_predictions = record['y_val'] == np.array(predictions)
    idx = np.random.permutation(len(predictions))
    for fraction in np.arange(0.5, 1.01, 0.01):
        part_size = int(fraction * len(idx))
        part = correct_predictions[idx][:part_size]
        accuracy = sum(part) / len(part)
        accumulation.append((fraction, accuracy, 'random'))


def print_acc_rejection():
    name = 'mnist'

    accumulation = []
    for i in range(3):
        methods = ['max_prob', 'mc_dropout', 'ht_dpp', 'ht_k_dpp']
        file_name = f'logs/classification/{name}_{i}/ue_max_prob.pickle'
        process_file(file_name, accumulation, methods, acquisition='max_prob')
        add_random(accumulation, file_name)

        methods = ['mc_dropout', 'ht_dpp', 'ht_k_dpp']
        file_name = f'logs/classification/{name}_{i}/ue_var_ratio.pickle'
        process_file(file_name, accumulation, methods, acquisition='var_ratio')

        methods = ['mc_dropout', 'ht_dpp', 'ht_k_dpp']
        file_name = f'logs/classification/{name}_{i}/ue_bald.pickle'
        process_file(file_name, accumulation, methods, acquisition='bald')

        methods = ['ensemble_max_prob']
        file_name = f'logs/classification/{name}_{i}/ue_ensemble.pickle'
        process_file(file_name, accumulation, methods, acquisition='')

    df = pd.DataFrame(accumulation, columns=['Fraction', 'Accuracy', 'method'])
    plt.figure(figsize=(12, 8))
    sns.lineplot(x='Fraction', y='Accuracy', data=df, hue='method')
    plt.title(f"{name.upper()}, accuracy by partly rejection")
    plt.xticks(rotation=20)
    plt.show()


if __name__ == '__main__':
    print_acc_rejection()
