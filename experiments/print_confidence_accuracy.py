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
        'ensemble_max_prob': 'Ensemble'
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
        file_name = f'logs/classification{resnet_str}/{args.name}_{i}/ue_{acquisition_str}{covariance_str}.pickle'
        process_file(file_name, args.acquisition, acc_conf, count_conf)
        ensemble_file = f'logs/classification/{args.name}_{i}/ue_ensemble.pickle'
        process_file(ensemble_file, args.acquisition, acc_conf, count_conf)


    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=(9, 5))


    # plt.subplot(1, 2, 1)
    plt.title(f"Confidence-accuracy {args.name} {args.acquisition}  {covariance_str}")
    df = pd.DataFrame(acc_conf, columns=['Confidence level', 'Accuracy', 'Estimator'])
    sns.lineplot('Confidence level', 'Accuracy', data=df, hue='Estimator')
    plt.subplots_adjust(right=0.7)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylabel(r"Accuracy for samples, confidence > $\tau$")
    plt.xlabel(r"$\tau$")
    # plt.savefig(f"data/conf_accuracy_{args.name}_{args.acquisition}", dpi=150)

    # plt.subplot(1, 2, 2)
    # # plt.figure(figsize=(8, 6))
    # plt.title(f"Confidence-count {args.name} {args.acquisition} {covariance_str}")
    # df = pd.DataFrame(count_conf, columns=['Confidence level', 'Count', 'Estimator'])
    # sns.lineplot('Confidence level', 'Count', data=df, hue='Estimator')

    plt.savefig(f"data/conf{resnet_str}_accuracy_{args.name}_{args.acquisition}{covariance_str}", dpi=150)
    plt.show()


def process_file(file_name, acquisition, acc_conf, count_conf):
    print(file_name)

    with open(file_name, 'rb') as f:
        record = pickle.load(f)

    prediction = np.argmax(np.array(record['probabilities']), axis=-1)
    is_correct = (prediction == record['y_val']).astype(np.int)

    # bins = np.concatenate((np.arange(0, 0.9, 0.1), np.arange(0.9, 1, 0.01)))
    bins = np.concatenate((np.arange(0, 1, 0.1), [0.97]))

    for estimator in record['estimators']:
        if estimator not in ['mc_dropout', 'max_prob', 'ht_dpp', 'ensemble_max_prob']:
            print('********************', estimator)
            continue

        print(record['uncertainties'].keys())
        ue = record['uncertainties'][estimator]

        if acquisition == 'bald':
            ue = ue / max(ue)
        # print(estimator)
        # print(min(ue), max(ue))

        for confidence_level in bins:
            point_confidences = 1 - ue
            bin_correct = is_correct[point_confidences > confidence_level]
            if len(bin_correct) > 0:
                accuracy = sum(bin_correct) / len(bin_correct)
            else:
                accuracy = None

            acc_conf.append((confidence_level, accuracy, estimator_name(estimator)))
            count_conf.append((confidence_level, len(bin_correct), estimator_name(estimator)))


if __name__ == '__main__':
    print_confidence()
