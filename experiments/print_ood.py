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
parser.add_argument('repeats', type=int)
parser.add_argument('--acquisition', '-a', type=str, default='bald')
parser.add_argument('--resnet', dest='resnet', action='store_true')
args = parser.parse_args()

print(args)

resnet_str = '_resnet' if args.resnet else ''
acquisition_str = 'bald' if args.acquisition == 'bald_n' else args.acquisition


def process_file(file_name, count_conf, args, methods):
    print(file_name)

    with open(file_name, 'rb') as f:
        record = pickle.load(f)

    if args.acquisition == 'bald':
        process_file_bald(count_conf, record, methods)
        return

    bins = np.concatenate((np.arange(0, 1, 0.1), [0.98, 0.99, 0.999]))
    for i, estimator in enumerate(record['estimators']):
        if estimator not in methods:
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


def process_file_bald(count_conf, record, methods):
    # bins = np.concatenate((np.arange(0, 1, 0.1), [0.98, 0.99, 0.999]))

    max_ue = 1.4 if 'mnist' in file_name else 1
    bins = np.concatenate(([0,  0.02, 0.04, 0.06], np.arange(0.1, max_ue, 0.1)))

    for i, estimator in enumerate(record['estimators']):
        if estimator not in methods:
            continue
        ue = record['uncertainties'][estimator]

        print(estimator)
        print(min(ue), max(ue))
        if args.acquisition == 'bald':
            ue = ue / max(ue)

        for ue_level in bins:
            level_count = np.sum(ue < ue_level)
            count_conf.append((ue_level, level_count, estimator_name(estimator)))


count_conf = []

for i in range(args.repeats):
    ensemble_method = f"ensemble_{args.acquisition}"
    file_name = f'logs/classification/{args.name}_{i}/ue_ensemble_ood.pickle'
    if os.path.exists(file_name):
        process_file(file_name, count_conf, args, [ensemble_method])

    file_name = f'logs/classification{resnet_str}/{args.name}_{i}/ue_ood_{acquisition_str}.pickle'
    process_file(file_name, count_conf, args, ['max_prob', 'mc_dropout', 'ht_dpp', 'ht_k_dpp', 'cov_dpp', 'cov_k_dpp'])


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
plt.savefig(f"data/conf_ood{resnet_str}_{args.name}_{args.acquisition}", dpi=150)
plt.show()

