import pickle
import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# if estimator == 'max_prob':
#     estimator_name = estimator
# else:
#     estimator_name = f"{estimator}_{args.acquisition}"


def estimator_name(estimator):
    return {
        'max_prob': 'Max probability',
        'mc_dropout': 'MC dropout',
        'ht_decorrelating': 'decorrelation',
        'ht_dpp': 'DPP',
        'ht_k_dpp': 'k-DPP',
        'cov_k_dpp': 'k-DPP',
        'ensemble_max_prob': 'Ensemble',
        'ensemble_bald': 'Ensemble'
    }[estimator]



def print_confidence():
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str)
    parser.add_argument('repeats', type=int)
    parser.add_argument('--acquisition', '-a', type=str, default='bald')
    parser.add_argument('--covariance', dest='covariance', action='store_true')
    parser.add_argument('--resnet', dest='resnet', action='store_true')
    args = parser.parse_args()

    acc_conf = []
    count_conf = []

    covariance_str = '_covar' if args.covariance else ''
    resnet_str = '_resnet' if args.resnet else ''
    acquisition_str = 'bald' if args.acquisition == 'bald_n' else args.acquisition

    for i in range(args.repeats):
        # ensemble_file = f'logs/classification/{args.name}_{i}/ue_ensemble.pickle'
        # if os.path.exists(ensemble_file):
        #     process_file(ensemble_file, args.acquisition, acc_conf, count_conf, ['ensemble_max_prob'])

        file_name = f'logs/classification{resnet_str}/{args.name}_{i}/ue_{acquisition_str}.pickle'
        if os.path.exists(file_name):
            process_file(file_name, args.acquisition, acc_conf, count_conf, ['mc_dropout', 'ht_dpp'])

        file_name = f'logs/classification{resnet_str}/{args.name}_{i}/ue_{acquisition_str}_covar.pickle'
        if os.path.exists(file_name):
            process_file(file_name, args.acquisition, acc_conf, count_conf, ['cov_k_dpp'])

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

    plt.savefig(f"data/conf{resnet_str}_accuracy_{args.name}_{args.acquisition}{covariance_str}", dpi=150)
    plt.show()


def process_file(file_name, acquisition, acc_conf, count_conf, methods):
    if acquisition == 'bald':
        process_file_bald(file_name, acquisition, acc_conf, count_conf, methods)
        return
    print(file_name)

    with open(file_name, 'rb') as f:
        record = pickle.load(f)

    prediction = np.argmax(np.array(record['probabilities']), axis=-1)
    is_correct = (prediction == record['y_val']).astype(np.int)

    bins = np.concatenate((np.arange(0, 1, 0.1), [0.98, 0.99, 0.999]))

    # for estimator in record['estimators']:
    for estimator in record['uncertainties'].keys():
        if estimator not in methods:
            continue

        ue = record['uncertainties'][estimator]

        print(estimator)
        print(min(ue), max(ue))

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
    print(file_name)

    with open(file_name, 'rb') as f:
        record = pickle.load(f)

    prediction = np.argmax(np.array(record['probabilities']), axis=-1)
    is_correct = (prediction == record['y_val']).astype(np.int)

    # bins = np.concatenate((np.arange(0, 1, 0.1), [0.98, 0.99, 0.999]))
    max_ue = 1.4 if 'mnist' in file_name else 1

    ## find max ue
    # max_ue = 0
    # for estimator in record['uncertainties'].keys():
    #     if estimator not in methods:
    #         continue
    #     max_ue = max(max_ue, max(record['uncertainties'][estimator]))


    bins = np.concatenate(([0, 0.01, 0.02, 0.03, 0.04, 0.05], np.arange(0.1, max_ue, 0.1)))
    print(bins)
    # for estimator in record['estimators']:
    for estimator in record['uncertainties'].keys():
        if estimator not in methods:
            continue
        print(estimator)

        ue = record['uncertainties'][estimator]

        print(estimator)
        print(min(ue), max(ue))

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
