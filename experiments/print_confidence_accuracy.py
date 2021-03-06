import pickle
import os
import argparse

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


def print_confidence():
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str)
    parser.add_argument('--acquisition', '-a', type=str, default='bald')
    parser.add_argument('--covariance', dest='covariance', action='store_true')
    args = parser.parse_args()

    args.repeats = {'mnist': 3, 'cifar': 3, 'imagenet': 1}[args.name]

    acc_conf = []
    count_conf = []

    covariance_str = '_covar' if args.covariance else ''
    acquisition_str = args.acquisition

    for i in range(args.repeats):
        file_name = f'logs/classification/{args.name}_{i}/ue_{acquisition_str}.pickle'
        if os.path.exists(file_name):
            process_file(file_name, args.acquisition, acc_conf, count_conf, ['mc_dropout', 'ht_dpp', 'cov_k_dpp', 'cov_dpp', 'ht_k_dpp', 'max_prob'])

        if args.acquisition == 'max_prob':
            ensemble_method = f"ensemble_{args.acquisition}"
            ensemble_file = f'logs/classification/{args.name}_{i}/ue_ensemble.pickle'
            if os.path.exists(ensemble_file):
                process_file(ensemble_file, args.acquisition, acc_conf, count_conf, [ensemble_method])


    plt.rcParams.update({'font.size': 13})
    plt.rc('grid', linestyle="--")
    plt.figure(figsize=(7, 5))

    if args.acquisition == 'bald':
        metric = 'UE'
    else:
        metric = 'Confidence'

    plt.title(f"{metric}-accuracy {args.name} {args.acquisition}  {covariance_str}")
    df = pd.DataFrame(acc_conf, columns=[f'{metric} level', 'Accuracy', 'Estimator'])
    sns.lineplot(f'{metric} level', 'Accuracy', data=df, hue='Estimator')
    # plt.subplots_adjust(right=0.7)
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    sign = '<' if args.acquisition == 'bald' else '<'
    plt.ylabel(fr"Accuracy for samples, {metric} {sign} $\tau$")
    plt.xlabel(r"$\tau$")
    plt.grid()

    plt.savefig(f"data/conf_accuracy_{args.name}_{args.acquisition}{covariance_str}", dpi=150)
    plt.show()


def process_file(file_name, acquisition, acc_conf, count_conf, methods):
    if acquisition == 'bald':
        process_file_bald(file_name, acquisition, acc_conf, count_conf, methods)
        return

    with open(file_name, 'rb') as f:
        record = pickle.load(f)

    prediction = np.argmax(np.array(record['probabilities']), axis=-1)
    is_correct = (prediction == record['y_val']).astype(np.int)

    bins = np.concatenate((np.arange(0, 1, 0.1), [0.98, 0.99, 0.999]))

    for estimator in methods:
        if estimator not in record['uncertainties'].keys():
            continue

        ue = record['uncertainties'][estimator]
        print(estimator)
        ue = ue / max(ue)

        for confidence_level in bins:
            point_confidences = 1 - ue
            bin_correct = is_correct[point_confidences > confidence_level]
            if len(bin_correct) > 0:
                accuracy = sum(bin_correct) / len(bin_correct)
            else:
                accuracy = None

            acc_conf.append((confidence_level, accuracy, estimator_name(estimator)))
            count_conf.append((confidence_level, len(bin_correct), estimator_name(estimator)))


def process_file_bald(file_name, acquisition, acc_conf, count_conf, methods):
    with open(file_name, 'rb') as f:
        record = pickle.load(f)

    prediction = np.argmax(np.array(record['probabilities']), axis=-1)
    is_correct = (prediction == record['y_val']).astype(np.int)

    if 'mnist' in file_name:
        bins = np.concatenate(([0, 0.01, 0.02, 0.03, 0.04, 0.05], np.arange(0.1, 1.4, 0.1)))
    elif 'cifar' in file_name:
        bins = np.concatenate(([0, 0.01, 0.02, 0.03, 0.04, 0.05], np.arange(0.1, 1, 0.1)))
    else:
        bins = np.concatenate(([0.05], np.arange(0.3, 3, 0.3)))

    for estimator in methods:
        if estimator not in record['uncertainties'].keys():
            continue
        ue = record['uncertainties'][estimator]

        for ue_level in bins:
            bin_correct = is_correct[ue < ue_level]

            if len(bin_correct) > 0:
                accuracy = sum(bin_correct) / len(bin_correct)
            else:
                accuracy = None

            acc_conf.append((ue_level, accuracy, estimator_name(estimator)))
            count_conf.append((ue_level, len(bin_correct), estimator_name(estimator)))


if __name__ == '__main__':
    print_confidence()
