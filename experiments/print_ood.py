import pickle
import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from print_confidence_accuracy import estimator_name


parser = argparse.ArgumentParser()
parser.add_argument('name', type=str)
parser.add_argument('--acquisition', '-a', type=str, default='bald')
args = parser.parse_args()
args.repeats = {'mnist': 3, 'cifar': 3, 'imagenet': 1}[args.name]


def process_file(file_name, count_conf, args, methods):

    with open(file_name, 'rb') as f:
        record = pickle.load(f)

    if args.acquisition == 'bald':
        process_file_bald(record, count_conf, args, methods)
        return

    bins = np.concatenate((np.arange(0, 1, 0.1), [0.98, 0.99, 0.999]))
    for estimator in methods:
        if estimator not in record['uncertainties'].keys():
            continue
        ue = record['uncertainties'][estimator]

        print(estimator)
        print(min(ue), max(ue))
        if args.acquisition == 'bald':
            ue = ue / max(ue)

        for confidence_level in bins:
            point_confidences = 1 - ue
            level_count = np.sum(point_confidences > confidence_level)
            count_conf.append((confidence_level, level_count, estimator_name(estimator)))


def process_file_bald(record, count_conf, args, methods):
    # bins = np.concatenate((np.arange(0, 1, 0.1), [0.98, 0.99, 0.999]))

    max_ue = {
        'mnist': 1.4, 'cifar': 1, 'imagenet': 1.1
    }[args.name]

    bins = np.concatenate(([0,  0.02, 0.04, 0.06], np.arange(0.1, max_ue, 0.02)))

    for estimator in methods:
        if estimator not in record['uncertainties'].keys():
            continue
        ue = record['uncertainties'][estimator]

        print(estimator)
        if args.acquisition == 'bald':
            ue = ue / max(ue)

        for ue_level in bins:
            level_count = np.sum(ue < ue_level)
            count_conf.append((ue_level, level_count, estimator_name(estimator)))


count_conf = []

for i in range(args.repeats):
    file_name = f'logs/classification/{args.name}_{i}/ue_ood_{args.acquisition}.pickle'
    if args.acquisition == 'max_prob':
        methods = ['mc_dropout', 'ht_dpp', 'cov_k_dpp', 'cov_dpp', 'ht_k_dpp', 'max_prob']
    else:
        methods = ['mc_dropout', 'ht_dpp', 'cov_k_dpp', 'cov_dpp', 'ht_k_dpp']


    process_file(file_name, count_conf, args, methods)

    if args.acquisition == 'max_prob':
        ensemble_method = f"ensemble_{args.acquisition}"
        file_name = f'logs/classification/{args.name}_{i}/ue_ensemble_ood.pickle'
        if os.path.exists(file_name):
            process_file(file_name, count_conf, args, [ensemble_method])



if args.acquisition == 'bald':
    metric = 'UE'
else:
    metric = 'Confidence'

plt.rcParams.update({'font.size': 13})
plt.rc('grid', linestyle="--")
plt.figure(figsize=(7, 5))
plt.title(f"{metric}-count for OOD {args.name} {args.acquisition}")
df = pd.DataFrame(count_conf, columns=[f'{metric} level', 'Count', 'Estimator'])
sns.lineplot(f'{metric} level', 'Count', data=df, hue='Estimator')
plt.subplots_adjust(left=0.15)
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

sign = '<' if args.acquisition == 'bald' else '>'
plt.ylabel(rf"Number of samples, {metric} {sign} $\tau$")
plt.xlabel(rf"$\tau$")
plt.grid()
plt.savefig(f"data/conf_ood_{args.name}_{args.acquisition}", dpi=150)
plt.show()

