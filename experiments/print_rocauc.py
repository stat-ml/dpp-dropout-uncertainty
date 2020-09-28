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
    errors = record['y_val'] != np.array(predictions)
    for method in methods:
        uq = roc_auc_score(errors, record['uncertainties'][method])
        accumulation.append((uq, f"{method}_{acquisition}"))


def print_rocauc():
    name = 'cifar'

    # with open('logs/classification/mnist_3/ue.pickle', 'rb') as f:
    #     record = pickle.load(f)
    # import ipdb; ipdb.set_trace()

    accumulation = []
    for i in range(3):
        methods = ['max_prob', 'mc_dropout', 'ht_dpp', 'ht_k_dpp']
        file_name = f'logs/classification/{name}_{i}/ue_max_prob.pickle'
        process_file(file_name, accumulation, methods, acquisition='max_prob')

        methods = ['mc_dropout', 'ht_dpp', 'ht_k_dpp']
        file_name = f'logs/classification/{name}_{i}/ue_var_ratio.pickle'
        process_file(file_name, accumulation, methods, acquisition='var_ratio')

        methods = ['mc_dropout', 'ht_dpp', 'ht_k_dpp']
        file_name = f'logs/classification/{name}_{i}/ue_bald.pickle'
        process_file(file_name, accumulation, methods, acquisition='bald')


    df = pd.DataFrame(accumulation, columns=['value', 'method'])
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='method', y='value', data=df)
    plt.title(f"{name.upper()}, error detection ROC AUC")
    plt.xticks(rotation=20)
    plt.show()


if __name__ == '__main__':
    print_rocauc()
